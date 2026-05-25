"""Test script for the refactored intent classification system.

Tests both keyword fallback (synchronous) and LLM-based (async) classification.
"""

import asyncio
import sys
import logging

# Configure logging to see debug output
logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

from services.intent_service import detect_intent, detect_intent_async, VALID_INTENTS

# ────────────────────────────────────────────────────────────────────
# Test cases: (input, expected_intent)
# ────────────────────────────────────────────────────────────────────

TEST_CASES = [
    # Required test cases from the spec
    ("هات فيديو python", "materials_request"),
    ("عايز roadmap للـ AI", "roadmap_request"),
    ("ايه machine learning", "general_question"),
    ("مش عارف أرفع التاسك", "support_request"),
    ("عايز أشتكي من المرشد", "complaint"),
    ("رشحلي mentor", "ask_mentor_recommendation"),
    ("مين أحسن mentor في AI", "find_mentor"),

    # Additional coverage tests
    ("hello", "greeting"),
    ("السلام عليكم", "greeting"),
    ("الجو عامل ايه", "off_topic"),
    ("tell me a joke", "off_topic"),
    ("مش فاهم التاسك", "task_help"),
    ("إزاي أسلم التاسك", "submit_task"),
    ("الديدلاين امتى", "submit_task"),
    ("هل المنصة مجانية", "faq"),
    ("how to register?", "faq"),
    ("give me videos about React", "materials_request"),
    ("كورسات AI", "materials_request"),
    ("شرحلي OOP", "general_question"),
    ("what is REST API", "general_question"),
    ("المرشد وحش", "complaint"),
    ("mentor is rude", "complaint"),
    ("مش بيرفع", "support_request"),
    ("not working", "support_request"),
    ("recommend mentors for me", "ask_mentor_recommendation"),
    ("اقترح عليا مرشدين مناسبين", "ask_mentor_recommendation"),
    ("best mentor in web dev", "find_mentor"),
    ("كيف أبدأ أتعلم web dev", "roadmap_request"),
]


def test_keyword_fallback():
    """Test the keyword-based fallback classifier."""
    print("\n" + "=" * 70)
    print("  KEYWORD FALLBACK TESTS (synchronous)")
    print("=" * 70)

    passed = 0
    failed = 0

    for text, expected in TEST_CASES:
        result = detect_intent(text)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status}  '{text[:45]:<45}' → {result:<25} (expected: {expected})")

    print(f"\n  Results: {passed}/{passed + failed} passed")
    return failed == 0


async def test_llm_classifier():
    """Test the LLM-based classifier (requires API key)."""
    print("\n" + "=" * 70)
    print("  LLM CLASSIFIER TESTS (async)")
    print("=" * 70)

    passed = 0
    failed = 0

    for text, expected in TEST_CASES:
        result = await detect_intent_async(text)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status}  '{text[:45]:<45}' → {result:<25} (expected: {expected})")

    print(f"\n  Results: {passed}/{passed + failed} passed")
    return failed == 0


def test_valid_intents():
    """Verify all 13 intents are registered."""
    print("\n" + "=" * 70)
    print("  VALID INTENTS CHECK")
    print("=" * 70)

    expected_intents = {
        "greeting", "find_mentor", "ask_mentor_recommendation",
        "ask_program_recommendation", "recommendation_explanation",
        "task_help", "submit_task", "roadmap_request",
        "materials_request", "faq",
        "complaint", "support_request", "general_question", "off_topic",
    }

    missing = expected_intents - VALID_INTENTS
    extra = VALID_INTENTS - expected_intents

    if missing:
        print(f"  ❌ Missing intents: {missing}")
    if extra:
        print(f"  ⚠️  Extra intents: {extra}")
    if not missing and not extra:
        print(f"  ✅ All {len(VALID_INTENTS)} intents registered correctly")

    for intent in sorted(VALID_INTENTS):
        print(f"     • {intent}")

    return len(missing) == 0


async def main():
    print("\n🧪 Intent Classification Test Suite")
    print("=" * 70)

    # Test 1: Valid intents
    intents_ok = test_valid_intents()

    # Test 2: Keyword fallback
    keywords_ok = test_keyword_fallback()

    # Test 3: LLM classifier
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "keywords":
        llm_ok = True
        print("\n  ⏭️  Skipping LLM tests (keywords-only mode)")
    else:
        llm_ok = await test_llm_classifier()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Valid Intents:    {'✅ PASS' if intents_ok else '❌ FAIL'}")
    print(f"  Keyword Fallback: {'✅ PASS' if keywords_ok else '❌ FAIL'}")
    if mode != "keywords":
        print(f"  LLM Classifier:   {'✅ PASS' if llm_ok else '❌ FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
