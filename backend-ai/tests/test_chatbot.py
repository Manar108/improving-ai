"""Simple local chatbot test — run interactively in the terminal.

Usage:
    cd mentorship-ai-assistant-mvp/backend-ai
    python test_chatbot.py
"""

import asyncio
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_chatbot")

# ---------------------------------------------------------------------------
# Import services (same ones the API uses)
# ---------------------------------------------------------------------------
from services.intent_service import intent_service
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.recommendation_service import recommendation_service
from services.search_service import search_service
from services.user_context_service import user_context_service


async def _handle_question(question: str, user_id: str = "") -> dict:
    """Process a single question through the chatbot pipeline.

    Returns a dict with intent, language, answer, results, warnings, and elapsed time.
    """
    warnings: list[str] = []
    t0 = time.perf_counter()

    # --- Language & Intent ---
    language = intent_service.detect_language(question)
    intent = intent_service.detect_intent(question)

    answer = ""
    results: list[dict] = []

    try:
        if intent == "greeting":
            answer = (
                "Welcome to the Mentorship Platform! 😊\nI can help you with:\n  🔍 Finding a mentor\n  📋 Personalized mentor recommendations\n  📝 Task help & guidance\n  🗺️ Learning roadmaps\n  ❓ Platform questions\n\nWhat do you need?"
                if language == "en"
                else "أهلاً بك في منصة الإرشاد! 😊\nأقدر أساعدك في:\n  🔍 البحث عن مرشد\n  📋 ترشيح مرشدين مناسبين\n  📝 مساعدة في التاسكات\n  🗺️ خرائط طريق للتعلم\n  ❓ أسئلة عن المنصة\n\nاكتبلي ايه اللي محتاجه!"
            )

        elif intent == "ask_mentor_recommendation":
            recs = await recommendation_service.get_recommendations(user_id=user_id or "")
            answer = "Your best mentor recommendations are ready 🎯" if language == "en" else "تم تجهيز أفضل الترشيحات لك 🎯"
            results = recs

        elif intent == "materials_request":
            search_result = await search_service.find_materials(query=question, language=language)
            answer = search_result.get("summary", "")
            results = search_result.get("results", [])

        elif intent == "faq":
            user_ctx = {}
            if user_id:
                user_ctx = user_context_service.get_user_context(user_id)
            answer = rag_service.answer_platform_question(question, language, user_context=user_ctx, intent=intent)
            if answer and ("not available" in answer.lower() or "غير متوفرة" in answer):
                answer = await llm_service.chat_general(question, language)

        elif intent == "find_mentor":
            user_ctx = {}
            if user_id:
                user_ctx = user_context_service.get_user_context(user_id)
            answer = rag_service.answer_platform_question(question, language, user_context=user_ctx, intent=intent)

        elif intent == "task_help":
            answer = await llm_service.chat_task_help(question, language)

        elif intent == "submit_task":
            answer = await llm_service.chat_submit_task(question, language)

        elif intent == "roadmap_request":
            search_result = await search_service.find_materials(query=question, language=language)
            materials = search_result.get("results", [])
            llm_roadmap = await llm_service.chat_roadmap(question, language)
            summary = search_result.get("summary", "")
            if summary and materials:
                answer = f"{llm_roadmap}\n\n---\n\n📚 {summary}"
            else:
                answer = llm_roadmap
            results = materials

        elif intent == "complaint":
            answer = await llm_service.chat_complaint(question, language)

        elif intent == "support_request":
            answer = await llm_service.chat_support(question, language)

        elif intent == "off_topic":
            answer = llm_service.chat_off_topic(language)

        else:
            answer = await llm_service.chat_general(question, language)
            if not answer:
                warnings.append("LLM fallback returned empty response")

    except Exception as exc:
        warnings.append(f"Error: {exc}")
        answer = f"Error occurred: {exc}"

    elapsed = time.perf_counter() - t0

    return {
        "intent": intent,
        "language": language,
        "answer": answer,
        "results": results,
        "warnings": warnings,
        "response_time": round(elapsed, 2),
    }


def _print_result(result: dict) -> None:
    """Pretty-print a chatbot test result."""
    print("\n" + "=" * 60)
    print(f"  Intent:        {result['intent']}")
    print(f"  Language:      {result['language']}")
    print(f"  Response Time: {result['response_time']} sec")
    if result["warnings"]:
        for w in result["warnings"]:
            print(f"  ⚠ Warning:    {w}")
    print("-" * 60)
    print(f"  Answer:\n    {result['answer'][:500]}")
    if result["results"]:
        print("-" * 60)
        print("  Results:")
        for i, r in enumerate(result["results"][:10], 1):
            if isinstance(r, dict):
                # Materials / recommendations
                title = r.get("title", r.get("mentor_name", r.get("label", "")))
                extra = r.get("url", r.get("link", r.get("domain", r.get("value", ""))))
                score = r.get("score", r.get("reason", ""))
                print(f"    {i}. {title}  —  {extra}  ({score})")
            else:
                print(f"    {i}. {r}")
    print("=" * 60)


def run_chat_test() -> None:
    """Interactive chatbot test loop.

    Type a question and see the full pipeline response.
    Type 'exit' or 'quit' to stop.
    """
    print("\n" + "=" * 60)
    print("  MENTORSHIP CHATBOT — Local Test")
    print("  Type a question, or 'exit' to quit.")
    print("=" * 60)

    user_id = input("\n  Enter user_id (or press Enter to skip): ").strip()

    while True:
        question = input("\n> Enter question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not question:
            print("  (empty question, try again)")
            continue

        result = asyncio.run(_handle_question(question, user_id))
        _print_result(result)


if __name__ == "__main__":
    run_chat_test()
