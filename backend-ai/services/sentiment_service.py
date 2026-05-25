"""Sentiment Analysis Service — fine-tuned BERT with Groq fallback.

Primary: fine-tuned BertForSequenceClassification (3 classes).
Fallback: Groq LLM (llama-3.3-70b-versatile) when BERT is slow or unavailable.

BERT is better for language-specific fine-tuned classification.
Groq fallback ensures the service never blocks or times out.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# pyrefly: ignore [missing-import]
import httpx

# Optional heavy deps — fall back to Groq when unavailable
try:
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    _HAS_TORCH = False

from config import settings

logger = logging.getLogger(__name__)

# ✅ ADDED: Import error handler
from services.error_handling import SentimentErrorHandler

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.3-70b-versatile"
_BERT_TIMEOUT = 1.0  # seconds — switch to Groq if BERT takes longer (keep <2s SLO)
_BATCH_TIMEOUT = 8.0  # seconds — switch to Groq if batch takes longer


@dataclass
class SentimentResult:
    """Immutable result returned by the sentiment predictor."""
    label: str
    confidence: float
    scores: dict[str, float]


class SentimentService:
    """Thread-safe, lazy-loaded sentiment predictor.

    Primary: fine-tuned BERT (fast for repeated texts via cache).
    Fallback: Groq LLM when BERT is slow or unavailable.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device: str = "cpu"
        self._loaded = False
        self._load_lock = threading.Lock()
        self._cache: dict[str, tuple[float, SentimentResult]] = {}
        self._cache_ttl = 300
        if _HAS_TORCH:
            try:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"
        else:
            logger.warning("torch/transformers not available — Groq-only mode")
            self._loaded = True

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if not _HAS_TORCH:
                self._loaded = True
                return

            model_path = Path(settings.SENTIMENT_MODEL_PATH)
            if not model_path.exists():
                logger.warning("BERT model not found at %s — switching to Groq fallback", model_path)
                self._loaded = True
                return

            logger.info("Loading sentiment BERT from %s (device=%s) ...", model_path, self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self._model.to(self._device)
            self._model.eval()
            logger.info("BERT model loaded successfully")
            self._loaded = True

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _get_cached(self, key: str) -> SentimentResult | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, result = entry
        if time.monotonic() - ts > self._cache_ttl:
            del self._cache[key]
            return None
        return result

    def _set_cached(self, key: str, result: SentimentResult) -> None:
        self._cache[key] = (time.monotonic(), result)

    @staticmethod
    def _validate_input(text: str) -> str:
        if text is None:
            return ""
        text = text.strip()
        if len(text) > 2048:
            text = text[:2048]
        return text

    # ------------------------------------------------------------------
    # BERT inference
    # ------------------------------------------------------------------

    def _predict_bert(self, text: str) -> SentimentResult:
        if not _HAS_TORCH or self._model is None or self._tokenizer is None:
            return self._groq_fallback([text])[0]

        inputs = self._tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = probabilities[0].cpu().tolist()
        predicted_idx = int(probabilities.argmax(dim=-1).item())
        predicted_label = LABEL_MAP.get(predicted_idx, "neutral")
        confidence = round(probs[predicted_idx], 4)
        scores = {LABEL_MAP[i]: round(p, 4) for i, p in enumerate(probs)}
        return SentimentResult(label=predicted_label, confidence=confidence, scores=scores)

    def _predict_bert_batch(self, texts: list[str]) -> list[SentimentResult]:
        if not _HAS_TORCH or self._model is None or self._tokenizer is None:
            return self._groq_fallback(texts)

        inputs = self._tokenizer(
            texts, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        results = []
        for i in range(len(texts)):
            probs = probabilities[i].cpu().tolist()
            predicted_idx = int(probabilities[i].argmax().item())
            predicted_label = LABEL_MAP.get(predicted_idx, "neutral")
            confidence = round(probs[predicted_idx], 4)
            scores = {LABEL_MAP[j]: round(p, 4) for j, p in enumerate(probs)}
            results.append(SentimentResult(label=predicted_label, confidence=confidence, scores=scores))
        return results

    # ------------------------------------------------------------------
    # Groq fallback (LLM-based)
    # ------------------------------------------------------------------

    def _groq_fallback(self, texts: list[str]) -> list[SentimentResult]:
        """Use Groq LLM as fallback when BERT is unavailable or slow."""
        if not settings.GROQ_API_KEY:
            return [self._keyword_fallback(t) for t in texts]

        if len(texts) == 1:
            return self._groq_single(texts[0])
        return self._groq_batch(texts)

    def _groq_single(self, text: str) -> list[SentimentResult]:
        system_prompt = (
            "Classify the following feedback as exactly one of: negative, neutral, or positive.\n"
            'Return ONLY valid JSON: {"label": "negative|neutral|positive", "confidence": 0.0-1.0, "scores": {"negative": 0.0, "neutral": 0.0, "positive": 0.0}}'
        )
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(
                    _GROQ_URL,
                    headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": _GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Classify: {text}"},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 64,
                    },
                )
            if resp.status_code == 200:
                import json
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                parsed = json.loads(raw)
                label = parsed.get("label", "neutral")
                if label not in LABEL_MAP.values():
                    label = "neutral"
                confidence = float(parsed.get("confidence", 0.5))
                scores = parsed.get("scores", {})
                for k in LABEL_MAP.values():
                    scores.setdefault(k, 0.0)
                return [SentimentResult(label=label, confidence=confidence, scores=scores)]
        except Exception as exc:
            logger.warning("Groq single fallback failed: %s", exc)
        return [self._keyword_fallback(text)]

    def _groq_batch(self, texts: list[str]) -> list[SentimentResult]:
        system_prompt = (
            "You are a sentiment classifier. For each piece of feedback, classify it as exactly one of: negative, neutral, or positive.\n"
            "Return ONLY a valid JSON list, no explanation:\n"
            '[{"label": "negative|neutral|positive", "confidence": 0.0, "scores": {"negative": 0.0, "neutral": 0.0, "positive": 0.0}}]'
        )
        user_prompt = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    _GROQ_URL,
                    headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": _GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 512,
                    },
                )
            if resp.status_code == 200:
                import json
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    parsed = [parsed]
                results = []
                for item in parsed:
                    label = item.get("label", "neutral")
                    if label not in LABEL_MAP.values():
                        label = "neutral"
                    confidence = float(item.get("confidence", 0.5))
                    scores = item.get("scores", {})
                    for k in LABEL_MAP.values():
                        scores.setdefault(k, 0.0)
                    results.append(SentimentResult(label=label, confidence=confidence, scores=scores))
                while len(results) < len(texts):
                    results.append(SentimentResult(label="neutral", confidence=0.5, scores={"negative": 0.2, "neutral": 0.6, "positive": 0.2}))
                return results[: len(texts)]
        except Exception as exc:
            logger.warning("Groq batch fallback failed: %s", exc)
        return [self._keyword_fallback(t) for t in texts]

    @staticmethod
    def _keyword_fallback(text: str) -> SentimentResult:
        """Keyword-based fallback when all else fails."""
        txt = (text or "").lower()
        positive = any(
            w in txt for w in [
                "good", "great", "excellent", "love", "happy", "amazing",
                "ممتاز", "جيد", "رائع", "ممتازة", "جيدة", "شكرا", "thanks",
                "helpful", "supportive", "مفيد", "منظم", "واضح", "مخلص",
            ]
        )
        negative = any(
            w in txt for w in [
                "bad", "terrible", "hate", "sad", "poor", "disappointed",
                "سيئ", "ضعيف", "سيئة", "ضعيفة", "أسوأ", "غير منظم",
                "not helpful", "bad experience", "waste of time",
            ]
        )
        if positive and not negative:
            return SentimentResult(label="positive", confidence=0.8, scores={"positive": 0.8, "neutral": 0.15, "negative": 0.05})
        if negative and not positive:
            return SentimentResult(label="negative", confidence=0.8, scores={"positive": 0.05, "neutral": 0.15, "negative": 0.8})
        return SentimentResult(label="neutral", confidence=0.6, scores={"positive": 0.2, "neutral": 0.6, "negative": 0.2})

    # ------------------------------------------------------------------
    # Public API with BERT + Groq fallback
    # ------------------------------------------------------------------

    def predict(self, text: str) -> SentimentResult:
        """Analyse text — BERT first, Groq if slow/unavailable."""
        text = self._validate_input(text)
        if not text:
            return SentimentResult(label="neutral", confidence=1.0, scores={"positive": 0.0, "neutral": 1.0, "negative": 0.0})

        cache_key = text.lower()
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._ensure_loaded()

        # Try BERT with timeout and error handling
        if _HAS_TORCH and self._model is not None:
            try:
                t0 = time.perf_counter()
                result = self._predict_bert(text)
                elapsed = (time.perf_counter() - t0) * 1000  # Convert to ms

                # If slow, only use fallback when confidence is low.
                # Avoid overriding correct predictions purely due to latency.
                if elapsed > 2000:  # > 2 seconds = slow
                    logger.warning(f"Sentiment analysis slow: {elapsed:.0f}ms (threshold: 2000ms)")
                    try:
                        # Keep result if model is confident; otherwise use handler
                        if getattr(result, "confidence", 0.0) >= 0.7:
                            logger.debug("Slow BERT result retained due to high confidence: %.2f", result.confidence)
                        else:
                            logger.debug("Slow BERT result low-confidence (%.2f) — using slow handler", getattr(result, "confidence", 0.0))
                            result = SentimentErrorHandler.handle_slow_sentiment(text, elapsed)
                    except Exception:
                        result = SentimentErrorHandler.handle_slow_sentiment(text, elapsed)

                self._set_cached(cache_key, result)
                return result
            except Exception as e:
                # ✅ ADDED: Handle BERT failure
                logger.error(f"BERT prediction failed: {e}", exc_info=True)
                # Try Groq fallback
                try:
                    result = self._groq_fallback([text])[0]
                    self._set_cached(cache_key, result)
                    return result
                except Exception as e2:
                    logger.error(f"Groq fallback also failed: {e2}", exc_info=True)
                    return SentimentErrorHandler.handle_failed_sentiment(e, language="ar" if any(c in text for c in "ابجد") else "en")

        # Try Groq with error handling
        try:
            result = self._groq_fallback([text])[0]
            self._set_cached(cache_key, result)
            return result
        except Exception as e:
            # ✅ ADDED: Final fallback when everything fails
            logger.error(f"Sentiment prediction completely failed: {e}", exc_info=True)
            return SentimentErrorHandler.handle_failed_sentiment(e, language="ar" if any(c in text for c in "ابجد") else "en")

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Predict sentiment for a batch — BERT first, Groq if slow/unavailable."""
        if not texts:
            return []

        texts = [self._validate_input(t) for t in texts]
        results: list[SentimentResult | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if not text:
                results[i] = SentimentResult(label="neutral", confidence=1.0, scores={"positive": 0.0, "neutral": 1.0, "negative": 0.0})
                continue
            cache_key = text.lower()
            cached = self._get_cached(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return [r for r in results if r is not None]

        self._ensure_loaded()

        # Try BERT with timeout
        if _HAS_TORCH and self._model is not None:
            try:
                t0 = time.perf_counter()
                bert_results = self._predict_bert_batch(uncached_texts)
                elapsed = time.perf_counter() - t0
                if elapsed > _BATCH_TIMEOUT:
                    logger.info("BERT batch slow (%.2fs for %d texts) — logging for monitoring", elapsed, len(uncached_texts))
                for idx, result in zip(uncached_indices, bert_results):
                    self._set_cached(uncached_texts[uncached_indices.index(idx)].lower(), result)
                    results[idx] = result
                return [r for r in results if r is not None]
            except Exception as exc:
                logger.warning("BERT batch failed, falling back to Groq: %s", exc)

        # Fall back to Groq
        groq_results = self._groq_fallback(uncached_texts)
        for idx, result in zip(uncached_indices, groq_results):
            self._set_cached(uncached_texts[uncached_indices.index(idx)].lower(), result)
            results[idx] = result

        return [r for r in results if r is not None]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def health(self) -> dict:
        """Return model health status."""
        return {
            "loaded": self._loaded,
            "device": self._device,
            "model_path": str(settings.SENTIMENT_MODEL_PATH),
            "labels": list(LABEL_MAP.values()),
            "bert_available": _HAS_TORCH and self._model is not None,
            "groq_fallback_enabled": bool(settings.GROQ_API_KEY),
        }


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------
sentiment_service = SentimentService()