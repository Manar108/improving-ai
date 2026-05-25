"""Sentiment Analysis Service Test — validates predict + predict_batch.

Usage:
    cd backend-ai
    python test_sentiment.py
"""

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_sentiment")

from services.sentiment_service import sentiment_service


def _print_result(label: str, result) -> None:
    print(f"\n  [{label}]")
    print(f"    Label:      {result.label}")
    print(f"    Confidence: {result.confidence:.4f}")
    print(f"    Scores:     {result.scores}")


def run_sentiment_tests() -> None:
    print("\n" + "=" * 60)
    print("  SENTIMENT ANALYSIS TEST")
    print("=" * 60)

    # --- Single predictions ---
    test_cases = [
        ("Positive (EN)", "great mentor, very helpful and knowledgeable"),
        ("Positive (AR)", "المرشد كان ممتاز والبرنامج مفيد جداً"),
        ("Negative (EN)", "terrible experience, not helpful at all"),
        ("Negative (AR)", "المرشد كان سيئ جداً ومش مفيد"),
        ("Neutral (EN)", "the program was okay, nothing special"),
        ("Neutral (AR)", "البرنامج عادي"),
        ("Empty/Edge", ""),
        ("Long text", "great " * 500),
    ]

    for label, text in test_cases:
        t0 = time.perf_counter()
        result = sentiment_service.predict(text)
        elapsed = (time.perf_counter() - t0) * 1000
        _print_result(label, result)
        print(f"    Time:       {elapsed:.1f}ms")

    # --- Batch prediction ---
    print("\n" + "-" * 60)
    print("  BATCH PREDICTION")
    print("-" * 60)
    batch_texts = [
        "amazing mentor, learned so much!",
        "worst experience ever, waste of time",
        "it was an average session",
    ]
    t0 = time.perf_counter()
    batch_results = sentiment_service.predict_batch(batch_texts)
    elapsed = (time.perf_counter() - t0) * 1000
    for text, result in zip(batch_texts, batch_results):
        print(f"  • '{text[:40]}...' → {result.label} ({result.confidence:.2%})")
    print(f"\n  Batch time: {elapsed:.1f}ms for {len(batch_texts)} items")

    # --- Health check ---
    print("\n" + "-" * 60)
    print("  HEALTH CHECK")
    print("-" * 60)
    health = sentiment_service.health()
    for k, v in health.items():
        print(f"  • {k}: {v}")

    print("\n" + "=" * 60)
    print("  ALL SENTIMENT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_sentiment_tests()
