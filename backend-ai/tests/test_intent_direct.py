"""Direct intent classification test — bypasses API, tests keyword fallback + LLM.

Usage:
    cd backend-ai
    python test_intent_direct.py
"""

import asyncio
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_intent_direct")

from services.intent_service import detect_intent, detect_intent_async, _keyword_fallback


TESTS = [
    # (message, expected_intent, notes)
    # --- Greeting ---
    ("مرحبا", "greeting", ""),
    ("hi there", "greeting", ""),
    ("السلام عليكم", "greeting", ""),
    ("hey", "greeting", ""),
    ("صباح الخير", "greeting", ""),

    # --- Find mentor (browse/search) ---
    ("عايز mentor في AI", "find_mentor", ""),
    ("I need a mentor for web development", "find_mentor", ""),
    ("ابغى حد يعلمني Python", "find_mentor", ""),
    ("ممكن mentor في الـ data science", "find_mentor", ""),
    ("best mentor in AI", "find_mentor", ""),
    ("كام مرشد في البرمجة", "find_mentor", ""),

    # --- Ask mentor recommendation (personalized) ---
    ("رشحلي مرشدين", "ask_mentor_recommendation", ""),
    ("مين أحسن مرشد؟", "find_mentor", ""),
    ("recommend me mentors", "ask_mentor_recommendation", ""),
    ("أريد توصيات مرشدين", "ask_mentor_recommendation", ""),
    ("suggest mentors based on my profile", "ask_mentor_recommendation", ""),

    # --- Task help ---
    ("مش فاهم التاسك", "task_help", ""),
    ("help me with the assignment", "task_help", ""),
    ("التاسك صعبه", "task_help", ""),
    ("can you explain this task?", "general_question", "Contains 'explain' → general_question (concept explanation merged)"),

    # --- Submit task ---
    ("إزاي أسلم التاسك؟", "submit_task", ""),
    ("how to submit my task?", "submit_task", ""),
    ("فين أرفع الشغل؟", "submit_task", ""),
    ("الديدلاين امتى؟", "submit_task", ""),

    # --- Roadmap request ---
    ("عايز roadmap للـ AI", "roadmap_request", ""),
    ("give me a learning path for backend", "roadmap_request", ""),
    ("خريطة طريق لتعلم Python", "roadmap_request", ""),
    ("أزاي أبدأ أتعلم web development?", "roadmap_request", ""),

    # --- Materials request ---
    ("هات فيديو python", "materials_request", ""),
    ("عايز كورسات AI", "materials_request", ""),

    # --- Concept explanation (merged into general_question) ---
    ("ايه machine learning", "general_question", ""),
    ("what is machine learning?", "general_question", ""),
    ("شرحلي OOP", "general_question", ""),
    ("explain REST APIs", "general_question", ""),
    ("ايه الفرق بين Python و Java؟", "general_question", ""),

    # --- FAQ ---
    ("مدة البرنامج قد ايه؟", "faq", ""),
    ("how do I register?", "faq", ""),
    ("هل المنصة مجانية؟", "faq", ""),
    ("ازاي أسجل؟", "faq", ""),

    # --- Complaint ---
    ("عايز أشتكي من المرشد", "complaint", ""),
    ("المرشد وحش", "complaint", ""),
    ("mentor is rude", "complaint", ""),
    ("عايز أقدم شكوى", "complaint", ""),

    # --- Support request ---
    ("مش عارف أرفع التاسك", "support_request", ""),
    ("الموقع فيه error", "support_request", ""),
    ("can't login", "support_request", ""),

    # --- General question ---
    ("what programming language should I learn first", "general_question", ""),
    ("tips for job interviews", "general_question", ""),
    ("كيف أحسن مستواي في البرمجة", "general_question", ""),

    # --- Off-topic ---
    ("الجو عامل ايه؟", "off_topic", ""),
    ("مين رئيس مصر؟", "off_topic", ""),
    ("tell me a joke", "off_topic", ""),
    ("اعملي شعر", "off_topic", ""),
]


def test_keyword_fallback():
    print("\n" + "=" * 70)
    print("  KEYWORD FALLBACK TEST (no LLM)")
    print("=" * 70 + "\n")

    passed = 0
    failed = 0
    errors = []

    for msg, expected, notes in TESTS:
        actual = _keyword_fallback(msg)
        ok = actual == expected
        icon = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1
            errors.append((msg, expected, actual, notes))
        print(f"  {icon}  {expected:30} {'✓' if ok else '→ ' + actual:30}  {msg}")

    total = passed + failed
    pct = (passed / total * 100) if total else 0
    print(f"\n{'='*70}")
    print(f"  Results: {passed}/{total} passed ({pct:.0f}%)")
    if errors:
        print(f"\n  Failed tests:")
        for msg, expected, actual, notes in errors:
            print(f"    • '{msg}' — expected {expected}, got {actual}")
            if notes:
                print(f"      Note: {notes}")
    print(f"{'='*70}\n")
    return failed == 0


async def test_llm_intent():
    print("\n" + "=" * 70)
    print("  LLM INTENT CLASSIFICATION TEST")
    print("=" * 70 + "\n")

    passed = 0
    failed = 0
    errors = []

    for msg, expected, notes in TESTS:
        t0 = time.perf_counter()
        actual = await detect_intent_async(msg)
        elapsed = (time.perf_counter() - t0) * 1000
        ok = actual == expected
        icon = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1
            errors.append((msg, expected, actual, notes, elapsed))
        print(f"  {icon}  {expected:30} {'✓' if ok else '→ ' + actual:30}  {msg}  ({elapsed:.0f}ms)")

    total = passed + failed
    pct = (passed / total * 100) if total else 0
    print(f"\n{'='*70}")
    print(f"  Results: {passed}/{total} passed ({pct:.0f}%)")
    if errors:
        print(f"\n  Failed tests:")
        for msg, expected, actual, notes, elapsed in errors:
            print(f"    • '{msg}' — expected {expected}, got {actual}  ({elapsed:.0f}ms)")
            if notes:
                print(f"      Note: {notes}")
    print(f"{'='*70}\n")
    return failed == 0


if __name__ == "__main__":
    kw_ok = test_keyword_fallback()
    llm_ok = asyncio.run(test_llm_intent())
    exit_code = 0 if (kw_ok and llm_ok) else 1
    print(f"\n  Final: keyword={'PASS' if kw_ok else 'FAIL'}, llm={'PASS' if llm_ok else 'FAIL'}")
    exit(exit_code)
