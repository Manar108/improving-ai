#!/usr/bin/env python3
"""Quick intent classification test script."""

import sys
import requests

sys.stdout.reconfigure(encoding="utf-8")

URL = "http://localhost:8088/api/v1/chat"

TESTS = [
    # (message, expected_intent)
    # Greetings
    ("مرحبا", "greeting"),
    ("hi there", "greeting"),
    ("السلام عليكم", "greeting"),
    ("hey", "greeting"),
    ("صباح الخير", "greeting"),

    # Find mentor
    ("عايز mentor في AI", "find_mentor"),
    ("I need a mentor for web development", "find_mentor"),
    ("ابغى حد يعلمني Python", "find_mentor"),
    ("ممكن mentor في الـ data science", "find_mentor"),

    # Mentor recommendations
    ("رشحلي مرشدين", "ask_mentor_recommendation"),
    ("مين أحسن مرشد؟", "ask_mentor_recommendation"),
    ("recommend me mentors", "ask_mentor_recommendation"),
    ("أريد توصيات مرشدين", "ask_mentor_recommendation"),

    # Task help
    ("مش فاهم التاسك", "task_help"),
    ("help me with the assignment", "task_help"),
    ("التاسك صعبه", "task_help"),
    ("can you explain this task?", "task_help"),

    # Submit task
    ("إزاي أسلم التاسك؟", "submit_task"),
    ("how to submit my task?", "submit_task"),
    ("فين أرفع الشغل؟", "submit_task"),
    ("الديدلاين امتى؟", "submit_task"),

    # Roadmap request
    ("عايز roadmap للـ AI", "roadmap_request"),
    ("give me a learning path for backend", "roadmap_request"),
    ("خريطة طريق لتعلم Python", "roadmap_request"),
    ("أزاي أبدأ أتعلم web development?", "roadmap_request"),

    # FAQ
    ("مدة البرنامج قد ايه؟", "faq"),
    ("how do I register?", "faq"),
    ("هل المنصة مجانية؟", "faq"),
    ("ازاي أسجل؟", "faq"),

    # General question
    ("ايه الفرق بين Python و Java؟", "general_question"),
    ("what is machine learning?", "general_question"),
    ("شرحلي OOP", "general_question"),
    ("explain REST APIs", "general_question"),

    # Off-topic
    ("الجو عامل ايه؟", "off_topic"),
    ("مين رئيس مصر؟", "off_topic"),
    ("tell me a joke", "off_topic"),
    ("اعملي شعر", "off_topic"),
]


def main():
    passed = 0
    failed = 0
    errors = []

    print(f"\n{'='*70}")
    print(f"  Intent Classification Test Suite — {len(TESTS)} tests")
    print(f"{'='*70}\n")

    for msg, expected in TESTS:
        try:
            r = requests.post(URL, json={"message": msg}, timeout=30)
            data = r.json()
            actual = data.get("intent", "?")
            ok = actual == expected
            icon = "✅" if ok else "❌"
            if ok:
                passed += 1
            else:
                failed += 1
                errors.append((msg, expected, actual))
            print(f"  {icon}  {expected:30} {'✓' if ok else '→ ' + actual:30}  {msg}")
        except Exception as e:
            failed += 1
            errors.append((msg, expected, f"ERROR: {e}"))
            print(f"  💥  {expected:30} {'ERROR':30}  {msg}")

    total = passed + failed
    pct = (passed / total * 100) if total else 0

    print(f"\n{'='*70}")
    print(f"  Results: {passed}/{total} passed ({pct:.0f}%)")
    if errors:
        print(f"\n  Failed tests:")
        for msg, expected, actual in errors:
            print(f"    • '{msg}' — expected {expected}, got {actual}")
    print(f"{'='*70}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
