"""Sentiment Analysis API routes.

Endpoints:
    POST /api/v1/sentiment/predict              — Single text prediction
    POST /api/v1/sentiment/predict-batch         — Batch prediction (up to 32 texts)
    GET  /api/v1/sentiment/mentor-summary/{id}   — Mentor feedback summary with satisfaction rate
    GET  /api/v1/sentiment/health                — Model health check

These endpoints are designed to be called by:
  1. The .NET backend (Mentora-backend) when feedback is submitted
  2. The frontend for real-time sentiment preview
  3. Internal AI services for analytics
"""

import logging
import time
import re

# pyrefly: ignore [missing-import]
import httpx

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException

from config import settings
from database.db import database
from schemas import (
    FeedbackBreakdown,
    MentorFeedbackSummaryResponse,
    SentimentRequest,
    SentimentResponse,
    SentimentBatchRequest,
    SentimentBatchResponse,
)
from services.sentiment_service import sentiment_service

logger = logging.getLogger(__name__)
router = APIRouter()

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _calculate_smart_satisfaction_rate(feedbacks_df, sentiment_results):
    """Calculate satisfaction rate combining star rating with sentiment.

    Star rating (1-5) normalized to 0-1 is the base.
    Sentiment label adjusts the score: negative reduces it, neutral nudges it.
    """
    if len(sentiment_results) == 0:
        return 0.0

    total = 0.0
    count = 0

    for idx, (comment, sentiment) in enumerate(sentiment_results):
        if idx >= len(feedbacks_df):
            break
        try:
            row = feedbacks_df.iloc[idx]
            rating = float(row.get("Rating", 3)) if "Rating" in feedbacks_df.columns else 3.0
            rating_norm = min(max(rating / 5.0, 0), 1)  # 0-1

            if sentiment.label == "positive":
                score = rating_norm  # full stars = full satisfaction
            elif sentiment.label == "negative":
                score = rating_norm - 0.3  # negative sentiment cuts satisfaction
            else:  # neutral
                score = 0.4 + (rating_norm * 0.4)  # neutral centers around 40-80%

            total += max(score, 0)
            count += 1
        except Exception:
            continue

    if count == 0:
        return 0.0

    return round(total / count * 100, 1)


def _clean_feedback_text(text: str) -> str:
    """تنقيح وتحسين نص الفيدباك قبل استخدامه.
    
    يقوم بـ:
    - إزالة الفراغات الزائدة
    - تنسيق الترقيم والعلامات
    - إصلاح الأخطاء الشائعة
    - تحسين التنسيق العام
    """
    if not text or not str(text).strip():
        return ""
    
    text = str(text).strip()
    
    # إزالة الفراغات الزائدة والأسطر الفارغة
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    
    # إزالة النقاط المكررة
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'،{2,}', '،', text)
    
    # إصلاح المسافات حول علامات الترقيم
    text = re.sub(r'\s+([,.!?،؛:])', r'\1', text)
    text = re.sub(r'([،؛:])\s*', r'\1 ', text)
    
    # تنسيق الأقواس
    text = re.sub(r'\s+\(', ' (', text)
    text = re.sub(r'\)\s+', ') ', text)
    
    # إزالة أحرف غريبة أو أرقام معزولة
    text = re.sub(r'^[\d\-\*\•]+\s*', '', text)
    
    # التأكد من الحد الأدنى من الطول
    if len(text) < 10:
        return ""
    
    # التأكد من أن النص يبدأ برأس مال (أول حرف أو رقم)
    if text and not text[0].isalnum():
        text = text.lstrip()
    
    return text[:500]  # تحديد الحد الأقصى للطول


@router.post("/sentiment/predict", response_model=SentimentResponse)
async def predict_sentiment(payload: SentimentRequest) -> SentimentResponse:
    """Analyse a single text and return sentiment label + confidence.

    Called by the .NET backend ``FeedbackService`` after a user submits
    feedback, or by any client that needs sentiment classification.

    Request body::

        { "text": "This mentor was incredibly helpful and supportive!" }

    Response::

        {
            "label": "positive",
            "confidence": 0.9823,
            "scores": { "negative": 0.0052, "neutral": 0.0125, "positive": 0.9823 }
        }
    """
    t0 = time.perf_counter()

    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty")

    try:
        result = sentiment_service.predict(payload.text.strip())
    except FileNotFoundError as exc:
        logger.error("Sentiment model not found: %s", exc)
        raise HTTPException(status_code=503, detail="Sentiment model not loaded") from exc
    except Exception as exc:
        logger.exception("Sentiment prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    elapsed = time.perf_counter() - t0
    logger.info(
        "Sentiment [%s] confidence=%.4f (%.3fs) text=%s",
        result.label, result.confidence, elapsed,
        payload.text[:80],
    )

    return SentimentResponse(
        label=result.label,
        confidence=result.confidence,
        scores=result.scores,
    )


@router.post("/sentiment/predict-batch", response_model=SentimentBatchResponse)
async def predict_sentiment_batch(payload: SentimentBatchRequest) -> SentimentBatchResponse:
    """Analyse multiple texts in a single request (max 32).

    Useful for bulk analysis of historical feedback data.
    """
    if not payload.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    if len(payload.texts) > 32:
        raise HTTPException(status_code=400, detail="Maximum 32 texts per batch request")

    t0 = time.perf_counter()

    try:
        results = sentiment_service.predict_batch(
            [t.strip() for t in payload.texts if t.strip()]
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded") from exc
    except Exception as exc:
        logger.exception("Batch sentiment prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    elapsed = time.perf_counter() - t0
    logger.info("Sentiment batch: %d texts (%.3fs)", len(results), elapsed)

    return SentimentBatchResponse(
        results=[
            SentimentResponse(label=r.label, confidence=r.confidence, scores=r.scores)
            for r in results
        ],
        count=len(results),
    )


@router.get("/sentiment/mentor-summary/{mentor_id}", response_model=MentorFeedbackSummaryResponse)
async def mentor_feedback_summary(mentor_id: str) -> MentorFeedbackSummaryResponse:
    """Generate a comprehensive feedback summary for a specific mentor.

    This endpoint:
    1. Fetches all feedback comments for the mentor from the database
    2. Runs sentiment analysis on each comment using the BERT model
    3. Calculates the satisfaction rate (% positive)
    4. Uses the LLM to generate a human-readable summary sentence

    Response example::

        {
            "mentor_id": "ABC-123",
            "mentor_name": "Ahmed Ali",
            "satisfaction_rate": 87.5,
            "average_rating": 4.3,
            "breakdown": { "positive": 14, "neutral": 2, "negative": 0, "total": 16 },
            "summary": "المتدربون أجمعوا على أن المرشد ممتاز في الشرح والتواصل، مع إشادة خاصة بخبرته في AI.",
            "top_positive_themes": ["شرح ممتاز", "تواصل فعال", "خبرة عالية"],
            "top_negative_themes": []
        }
    """
    t0 = time.perf_counter()

    # 1. Fetch mentor name
    mentor_df = database.run_query_df(
        """
        SELECT CONCAT(u.first_name, ' ', u.last_name) AS name
        FROM mentor_profile mp
        INNER JOIN users u ON u.user_id = mp.user_id
        WHERE mp.user_id = :mid
        """,
        {"mid": mentor_id},
    )
    mentor_name = mentor_df.iloc[0]["name"] if not mentor_df.empty else "Unknown"

    # 2. Fetch all feedbacks for this mentor
    feedbacks_df = database.run_query_df(
        """
        SELECT f.Rating, f.Comment
        FROM feedbacks f
        WHERE f.MentorProfileId = :mid
        ORDER BY f.CreatedAt DESC
        """,
        {"mid": mentor_id},
    )

    if feedbacks_df.empty:
        return MentorFeedbackSummaryResponse(
            mentor_id=mentor_id,
            mentor_name=mentor_name,
            summary="لا توجد تقييمات لهذا المرشد بعد.",
        )

    # 3. Run sentiment analysis on comments (batch)
    comments = [
        str(c).strip()
        for c in feedbacks_df["Comment"].dropna().tolist()
        if str(c).strip()
    ]

    positive_count = 0
    neutral_count = 0
    negative_count = 0
    positive_comments = []
    negative_comments = []
    sentiment_results_with_ratings = []  # لاستخدامها في حساب satisfaction_rate الذكي

    if comments:
        # Process in batches of 32
        all_results = []
        for i in range(0, len(comments), 32):
            batch = comments[i : i + 32]
            try:
                results = sentiment_service.predict_batch(batch)
                all_results.extend(zip(batch, results))
            except Exception as exc:
                logger.warning("Batch sentiment failed for mentor %s: %s", mentor_id, exc)
                # Fallback to individual predictions
                for text in batch:
                    try:
                        r = sentiment_service.predict(text)
                        all_results.append((text, r))
                    except Exception:
                        pass

        for text, result in all_results:
            sentiment_results_with_ratings.append((text, result))
            if result.label == "positive":
                positive_count += 1
                cleaned = _clean_feedback_text(text)
                if cleaned:
                    positive_comments.append(cleaned)
            elif result.label == "negative":
                negative_count += 1
                cleaned = _clean_feedback_text(text)
                if cleaned:
                    negative_comments.append(cleaned)
            else:
                neutral_count += 1

    total = positive_count + neutral_count + negative_count
    
    # استخدام الحساب الذكي للـ satisfaction_rate
    satisfaction_rate = _calculate_smart_satisfaction_rate(feedbacks_df, sentiment_results_with_ratings)

    # Average star rating
    avg_rating = 0.0
    if "Rating" in feedbacks_df.columns:
        avg_rating = round(float(feedbacks_df["Rating"].astype(float).mean()), 2)

    # 4. Generate AI summary sentence using LLM
    summary = ""
    top_positive = []
    top_negative = []

    if comments and settings.GROQ_API_KEY:
        # Build context for the LLM
        sample_positive = positive_comments[:5]
        sample_negative = negative_comments[:3]

        prompt = f"""أنت محلل تقييمات مرشدين. حلل التقييمات التالية واستخرج ملخص ونقاط رئيسية.

القواعد المهمة جداً:
- summary: جملة عربية واحدة وصفية بين 8 و 20 كلمة تلخص رأي المتدربين في المرشد. يجب أن تكون جملة كاملة مفيدة وليست كلمتين فقط.
  أمثلة صحيحة:
  - "المتدربون أشادوا بقدرة المرشد على الشرح الواضح والتواصل الفعال معهم"
  - "المرشد متعاون وصبور في الشرح لكن بعض المتدربين لاحظوا تأخر في الرد"
  - "تجربة إيجابية بشكل عام مع إشادة بخبرة المرشد التقنية وأسلوبه في التوجيه"
  أمثلة خاطئة (ممنوعة): "مينتور ممتاز", "تجربة جيدة", "كويس"
- positive_themes: صفات قصيرة فقط (2-3 كلمات). أمثلة: "شرح واضح", "تواصل ممتاز", "خبرة عالية"
- negative_themes: صفات قصيرة فقط (2-3 كلمات). أمثلة: "تأخر في الرد", "غير منظم"
- لو الكلمة في التقييم غريبة أو غير واضحة، حولها لكلمة معروفة وواضحة
- أقصى حد: 3 positive_themes و 2 negative_themes. بدون تكرار.

التقييمات الإيجابية:
{chr(10).join(f'- {c[:150]}' for c in sample_positive) if sample_positive else 'لا توجد'}

التقييمات السلبية:
{chr(10).join(f'- {c[:150]}' for c in sample_negative) if sample_negative else 'لا توجد'}

أجب بـ JSON فقط بهذا الشكل بالظبط:
{{"summary": "جملة وصفية كاملة من 8-20 كلمة",
  "positive_themes": ["صفة1", "صفة2"],
  "negative_themes": ["صفة1"]}}"""

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    _GROQ_URL,
                    headers={
                        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 200,
                    },
                )
            if resp.status_code == 200:
                import json
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                try:
                    if "```" in raw:
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    parsed = json.loads(raw.strip())
                    summary = parsed.get("summary", "")
                    raw_pos = parsed.get("positive_themes", [])
                    raw_neg = parsed.get("negative_themes", [])
                    # Deduplicate, keep short, max 3 positive / 2 negative
                    seen_pos = set()
                    top_positive = []
                    for t in raw_pos:
                        t = str(t).strip().rstrip(".")
                        if t and t not in seen_pos and len(top_positive) < 3:
                            top_positive.append(t)
                            seen_pos.add(t)
                    seen_neg = set()
                    top_negative = []
                    for t in raw_neg:
                        t = str(t).strip().rstrip(".")
                        if t and t not in seen_neg and len(top_negative) < 2:
                            top_negative.append(t)
                            seen_neg.add(t)
                except json.JSONDecodeError:
                    summary = raw[:200]
        except Exception as exc:
            logger.warning("LLM summary generation failed for mentor %s: %s", mentor_id, exc)

    # Fallback: Build summary from data (themes, rating) instead of hardcoded text
    if not summary:
        parts = []
        if avg_rating >= 4.5:
            parts.append(f"تقييم ممتاز ({avg_rating:.1f}/5)")
        elif avg_rating >= 4.0:
            parts.append(f"تقييم عالي ({avg_rating:.1f}/5)")
        elif avg_rating >= 3.0:
            parts.append(f"تقييم جيد ({avg_rating:.1f}/5)")
        
        if positive_count > 0:
            parts.append(f"{positive_count} تقييمات إيجابية")
        if top_positive:
            parts.append(f"نقاط قوية: {', '.join(top_positive[:2])}")
        if top_negative:
            parts.append(f"نقاط للتحسين: {', '.join(top_negative[:1])}")
        
        summary = " | ".join(parts) if parts else f"المرشد {mentor_name} - {positive_count}/{total} تقييمات إيجابية"

    elapsed = time.perf_counter() - t0
    logger.info(
        "Mentor summary [%s] %s: satisfaction=%.1f%% avg=%.1f total=%d (%.2fs)",
        mentor_id, mentor_name, satisfaction_rate, avg_rating, total, elapsed,
    )

    return MentorFeedbackSummaryResponse(
        mentor_id=mentor_id,
        mentor_name=mentor_name,
        satisfaction_rate=satisfaction_rate,
        average_rating=avg_rating,
        breakdown=FeedbackBreakdown(
            positive=positive_count,
            neutral=neutral_count,
            negative=negative_count,
            total=total,
        ),
        summary=summary,
        top_positive_themes=top_positive,
        top_negative_themes=top_negative,
    )


@router.get("/sentiment/health")
def sentiment_health() -> dict:
    """Check sentiment model status."""
    return sentiment_service.health()
