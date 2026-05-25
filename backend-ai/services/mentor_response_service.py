"""Mentor Response Service — generates mentor-specific responses.

Reuses the existing LLM and RAG infrastructure for lightweight generation.
Provides mentor-focused responses for FAQs, analytics summaries, workflow help, etc.
"""

from services import rag_service
import logging
from typing import Optional

from services.llm_service import llm_service
from services.search_service import search_service
from services.mentor_context_service import mentor_context_service

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Mentor-specific greetings
# ─────────────────────────────────────────────────────────────────────

MENTOR_GREETINGS = {
    "ar": (
        "أهلاً بك! 👋\n"
        "أقدر أساعدك في:\n"
        "  📊 إحصائيات برامجك\n"
        "  📚 موارد تعليمية للمنتيز\n"
        "  ❓ أسئلة عن المنصة\n"
        "  🎯 إدارة المنتيز والتطبيقات\n\n"
        "ازاي أساعدك؟"
    ),
    "en": (
        "Welcome Mentor! 👋\n"
        "I can help you with:\n"
        "  📊 Your program statistics\n"
        "  📚 Teaching materials & resources\n"
        "  ❓ Platform questions\n"
        "  🎯 Managing mentees & applications\n\n"
        "How can I assist?"
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Response generators
# ─────────────────────────────────────────────────────────────────────

async def respond_to_greeting(language: str) -> str:
    """Generate greeting response."""
    return MENTOR_GREETINGS.get(language, MENTOR_GREETINGS["en"])


async def respond_to_faq(message: str, language: str, mentor_context: dict) -> str:
    """Answer FAQs about platform features and how-to questions.
    
    Uses RAG first (search FAQ database), then falls back to LLM.
    """
    try:
        # Try RAG first (FAQ knowledge base)
        answer = await rag_service.answer_platform_question(
            message,
            language,
            user_context={"role": "mentor"},
            intent="faq",
        )
        if answer and "not available" not in answer.lower():
            return answer
    except Exception as e:
        logger.warning("FAQ RAG failed: %s", e)

    # Fallback to LLM with mentor-specific system prompt
    mentor_faq_prompt = """
You are a mentor assistant answering questions about the mentorship platform.

Answer questions about:
- How to create programs and publish them
- How to review mentee submissions
- How to give feedback to mentees
- How mentorship sessions work
- Platform features for mentors
- How the application process works
- How to contact mentees
- How the rating system works

Keep answers practical, clear, and action-oriented for mentors.
Use the same language as the user (Arabic or English).
"""

    try:
        answer = await llm_service.chat_with_system_prompt(
            message,
            language,
            system_prompt=mentor_faq_prompt,
        )
        return answer
    except Exception as e:
        logger.error("FAQ LLM failed: %s", e)
        return (
            "معذرة، حدثت مشكلة في معالجة سؤالك. جرّب لاحقاً 😊"
            if language == "ar"
            else "Sorry, I encountered an error. Please try again later 😊"
        )


async def respond_to_materials_request(message: str, language: str) -> tuple[str, list]:
    """Generate teaching materials suggestions.
    
    Returns: (answer_text, materials_list)
    
    Materials can include:
    - Interview questions
    - Coding exercises
    - Project ideas
    - Quiz templates
    - Assignment ideas
    """
    # Use the existing search service (reuse from mentee chatbot)
    search_result = await search_service.find_materials(query=message, language=language)

    materials = []
    if search_result.get("results"):
        materials = [
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "kind": _classify_material_type(r.get("link", ""), r.get("title", "")),
                "source": r.get("source", ""),
            }
            for r in search_result.get("results", [])
        ]

    summary = search_result.get("summary", "")
    if not summary:
        summary = (
            "لقد جمعت لك موارد تعليمية مناسبة 📚"
            if language == "ar"
            else "I've gathered teaching materials for you 📚"
        )

    return summary, materials


async def respond_to_mentor_analytics(message: str, language: str, mentor_id: str) -> str:
    """Generate analytics summary for mentor.
    
    Queries mentor's programs and mentees, then summarizes with LLM.
    """
    try:
        # Load mentor context and programs
        mentor_ctx = mentor_context_service.get_mentor_context(mentor_id)
        programs = mentor_context_service.get_mentor_programs(mentor_id, limit=5)
        mentees = mentor_context_service.get_mentor_active_mentees(mentor_id, limit=10)

        # Build summary
        summary_data = f"""
Mentor: {mentor_ctx.get('first_name')} {mentor_ctx.get('last_name')}
Programs: {mentor_ctx.get('program_count')}
Active Mentees: {len(mentees)}
Average Rating: {mentor_ctx.get('average_rating')}/5 ({mentor_ctx.get('total_reviews')} reviews)

Programs:
"""
        for p in programs:
            summary_data += f"- {p['title']}: {p['active_mentees']} active mentees, {p['applications_count']} applications\n"

        # Use LLM to create a natural response based on the data
        analytics_prompt = f"""
The mentor asked about their statistics/analytics. Here's their data:

{summary_data}

Answer their question based on this data. If they asked about a specific metric (e.g., "how many mentees?"),
provide that directly. If they asked for a general overview, summarize the key highlights.

Use the same language as their question (Arabic or English).
Be concise and data-driven. Highlight key achievements and areas.
"""

        answer = await llm_service.chat_with_system_prompt(
            message,
            language,
            system_prompt=analytics_prompt,
        )
        return answer

    except Exception as e:
        logger.error("Analytics generation failed: %s", e)
        return (
            "معذرة، حدثت مشكلة في استرجاع الإحصائيات 😊"
            if language == "ar"
            else "Sorry, I encountered an error retrieving your analytics 😊"
        )


async def respond_to_mentor_workflow_help(message: str, language: str, mentor_id: str) -> str:
    """Help mentor manage mentorships, mentees, applications, communication.
    
    Examples:
    - How to contact mentees
    - How to structure sessions
    - How to review applications
    - How to give feedback
    """
    try:
        mentor_ctx = mentor_context_service.get_mentor_context(mentor_id)
        programs = mentor_context_service.get_mentor_programs(mentor_id, limit=5)
        pending_apps = mentor_context_service.get_mentor_pending_applications(mentor_id, limit=5)

        # Build context
        context_data = f"""
Mentor: {mentor_ctx.get('first_name')} {mentor_ctx.get('last_name')}
Current Programs: {len(programs)}
Pending Applications: {len(pending_apps)}

If asking about applications, here are recent pending ones:
"""
        for app in pending_apps:
            context_data += f"- {app['mentee_name']} applied to '{app['program_title']}' on {app['applied_at']}\n"

        workflow_prompt = f"""
The mentor is asking for help managing their mentorship workflow.
Here's their current context:

{context_data}

Answer their question about managing mentees, handling applications, structuring sessions, or communicating with mentees.
Provide practical, actionable guidance.

Use the same language as their question (Arabic or English).
Keep it concise and focused.
"""

        answer = await llm_service.chat_with_system_prompt(
            message,
            language,
            system_prompt=workflow_prompt,
        )
        return answer

    except Exception as e:
        logger.error("Workflow help failed: %s", e)
        return (
            "معذرة، حدثت مشكلة في توليد المساعدة 😊"
            if language == "ar"
            else "Sorry, I encountered an error. Please try again 😊"
        )


async def respond_to_off_topic(language: str) -> str:
    """Politely reject off-topic messages."""
    return (
        "يرجى الالتزام بأسلوب محترم في التواصل. "
        "المنصة مصممة لمساعدتك في الإرشاد والتدريس 😊"
        if language == "ar"
        else "Please keep the conversation respectful and platform-focused. "
        "I'm here to help with mentorship and teaching 😊"
    )


async def respond_to_general_question(message: str, language: str, history: Optional[list] = None) -> str:
    """Answer general educational questions (mentor's own learning)."""
    try:
        answer = await llm_service.chat_general(message, language, history=history)
        return answer
    except Exception as e:
        logger.error("General question failed: %s", e)
        return (
            "معذرة، حدثت مشكلة في معالجة سؤالك 😊"
            if language == "ar"
            else "Sorry, I encountered an error 😊"
        )


# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────

def _classify_material_type(link: str, title: str) -> str:
    """Classify material URL into a type for frontend cards.
    
    Reused from mentee chatbot (see chat.py _classify_link).
    """
    lowered = (link + " " + title).lower()

    if any(s in lowered for s in ["youtube", "youtu.be", "vimeo", "udemy"]):
        return "videos"
    if any(s in lowered for s in ["coursera", "edx", "pluralsight"]):
        return "courses"
    if "github" in lowered:
        return "projects"
    if any(s in lowered for s in ["docs", "documentation"]):
        return "docs"
    if any(s in lowered for s in ["medium", "dev.to", "tutorial", "guide"]):
        return "articles"

    return "articles"


# ─────────────────────────────────────────────────────────────────────
# Service class
# ─────────────────────────────────────────────────────────────────────

class MentorResponseService:
    """Generates mentor-specific responses."""

    @staticmethod
    async def greet(language: str) -> str:
        return await respond_to_greeting(language)

    @staticmethod
    async def faq(message: str, language: str, mentor_context: dict) -> str:
        return await respond_to_faq(message, language, mentor_context)

    @staticmethod
    async def materials(message: str, language: str) -> tuple[str, list]:
        return await respond_to_materials_request(message, language)

    @staticmethod
    async def analytics(message: str, language: str, mentor_id: str) -> str:
        return await respond_to_mentor_analytics(message, language, mentor_id)

    @staticmethod
    async def workflow_help(message: str, language: str, mentor_id: str) -> str:
        return await respond_to_mentor_workflow_help(message, language, mentor_id)

    @staticmethod
    async def off_topic(language: str) -> str:
        return await respond_to_off_topic(language)

    @staticmethod
    async def general_question(message: str, language: str, history: Optional[list] = None) -> str:
        return await respond_to_general_question(message, language, history)


# Singleton
mentor_response_service = MentorResponseService()
