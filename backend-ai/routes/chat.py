"""Chat route — routes user messages through intent classification to the
correct handler (mentor search, recommendations, task help, roadmap,
materials, explanations, FAQ, complaints, support, general Q&A,
or off-topic rejection).
"""

import asyncio
import logging
import time

# pyrefly: ignore [missing-import]
from fastapi import APIRouter

from config import settings
from schemas import ChatRequest, ChatResponse
from services.intent_service import intent_service
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.recommendation_service import recommendation_service
from services.search_service import search_service
from services.user_context_service import user_context_service
from services.program_recommendation_service import program_recommendation_service

# ✅ ADDED: Import error handlers and valid intents
from services.error_handling import ChatbotErrorHandler, SentimentErrorHandler
from services.intent_service import VALID_INTENTS

logger = logging.getLogger(__name__)
router = APIRouter()


# ────────────────────────────────────────────────────────────────────
# Greeting messages
# ────────────────────────────────────────────────────────────────────

_GREETINGS = {
    "ar": (
        "أهلاً بك في منصة الإرشاد! 😊\n"
        "أقدر أساعدك في:\n"
        "  🔍 البحث عن مرشد\n"
        "  📋 ترشيح مرشدين مناسبين\n"
        "  📝 مساعدة في التاسكات\n"
        "  🗺️ خرائط طريق للتعلم\n"
        "  ❓ أسئلة عن المنصة\n\n"
        "اكتبلي ايه اللي محتاجه!"
    ),
    "en": (
        "Welcome to the Mentorship Platform! 😊\n"
        "I can help you with:\n"
        "  🔍 Finding a mentor\n"
        "  📋 Personalized mentor recommendations\n"
        "  📝 Task help & guidance\n"
        "  🗺️ Learning roadmaps\n"
        "  ❓ Platform questions\n\n"
        "What do you need?"
    ),
}


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _classify_link(link: str, title: str) -> str:
    """Classify a URL into a material kind for the frontend cards.
    
    Returns: "videos", "courses", "docs", "articles", "projects"
    """
    lowered = (link + " " + title).lower()
    
    # Video platforms
    if any(s in lowered for s in ["youtube", "youtu.be", "vimeo", "udemy", "lynda"]):
        return "videos"
    
    # Course platforms
    if any(s in lowered for s in ["coursera", "edx", "pluralsight", "skillshare", "treehouse"]):
        return "courses"
    
    # GitHub/Project repositories
    if "github" in lowered and any(s in lowered for s in ["project", "repo", "example", "template", "starter"]):
        return "projects"
    
    # Official documentation
    if any(s in lowered for s in ["docs", "documentation", "developer.mozilla", "learn.microsoft", "devdocs", "official"]):
        return "docs"
    
    # Learning platforms
    if any(s in lowered for s in ["freecodecamp", "codecademy", "w3schools", "tutorialspoint"]):
        return "courses"
    
    # Blog articles & tutorials
    if any(s in lowered for s in ["medium", "dev.to", "hashnode", "blog", "article", "tutorial", "guide"]):
        return "articles"
    
    # Q&A sites
    if "stackoverflow" in lowered:
        return "articles"
    
    # Default to article
    return "articles"


# ────────────────────────────────────────────────────────────────────
# Main chat endpoint
# ────────────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    language = intent_service.detect_language(payload.message)

    # ── Step 1: Classify intent via LLM ──
    if settings.GROQ_API_KEY:
        try:
            intent = await intent_service.detect_intent_async(payload.message)
            # ✅ ADDED: Validate intent is in valid set
            if intent not in VALID_INTENTS:
                logger.warning(f"Invalid intent returned: {intent}, using general_question")
                intent = "general_question"
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            # ✅ ADDED: Better error handling with fallback
            intent = ChatbotErrorHandler.handle_intent_service_error()
            if not isinstance(intent, str) or intent not in VALID_INTENTS:
                intent = "general_question"
    else:
        intent = "general_question"

    # ── Step 2: Load user context if available ──
    user_ctx = {}
    if payload.user_id:
        user_ctx = user_context_service.get_user_context(payload.user_id)
        if user_ctx.get("user_id"):
            logger.info(
                "User context loaded: %s %s | role=%s | domain=%s",
                user_ctx.get("first_name"), user_ctx.get("last_name"),
                user_ctx.get("role"), user_ctx.get("domain_name"),
            )

    # Normalize and trim `history` to avoid sending very large payloads to the LLM.
    # Keep only the last N messages and truncate long messages to save tokens/time.
    history = getattr(payload, "history", None) or []
    MAX_HISTORY_MESSAGES = 6
    MAX_MESSAGE_CHARS = 1000
    trimmed_history: list[dict] = []
    for h in history[-MAX_HISTORY_MESSAGES:]:
        if isinstance(h, dict):
            role = h.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            content = h.get("content", "") or ""
        else:
            role = "user"
            content = str(h)
        if len(content) > MAX_MESSAGE_CHARS:
            # Keep the tail of the message (most recent content)
            content = content[-MAX_MESSAGE_CHARS:]
        trimmed_history.append({"role": role, "content": content})
    if len(trimmed_history) != len(history):
        logger.debug("Trimmed history from %d to %d messages", len(history), len(trimmed_history))
    payload.history = trimmed_history

    # ── Step 3: Route to handler ──

    # ─── GREETING ───
    if intent == "greeting":
        elapsed = time.perf_counter() - t0
        logger.info("Chat [greeting] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=_GREETINGS.get(language, _GREETINGS["en"]),
        )

    # ─── OFF-TOPIC ───
    if intent == "off_topic":
        elapsed = time.perf_counter() - t0
        # Check if message contains profanity/abuse → give firmer response
        _profanity_markers = [
            "fuck", "shit", "bitch", "asshole", "bastard",
            "stfu", "wtf", "idiot", "stupid", "useless", "dumb", "trash", "suck",
            "يلعن",
            "يا حمار", "يا غبي", "كلب", "حيوان", "ولاد الوسخة",
            "اخرس", "يا واطي", "يا قذر", "ابن الكلب",
            "زبالة", "تافه", "يا عبيط",
        ]
        msg_lower = payload.message.lower()
        is_abusive = any(w in msg_lower for w in _profanity_markers)
        if is_abusive:
            logger.warning("Chat [off_topic/abuse] (%.2fs): %s", elapsed, payload.message[:80])
            abuse_answer = (
                "يرجى الالتزام بأسلوب محترم في التواصل. "
                "المنصة مصممة لمساعدتك في التعلم والإرشاد 😊"
                if language == "ar"
                else "Please keep the conversation respectful. "
                "This platform is here to help you with learning and mentorship 😊"
            )
        else:
            abuse_answer = llm_service.chat_off_topic(language)
        logger.info("Chat [off_topic] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=abuse_answer,
        )

    # ─── FIND MENTOR (search from database) ───
    if intent == "find_mentor":
        # Use the RAG service to search for mentors in the database
        answer = await rag_service.answer_platform_question(payload.message, language, user_context=user_ctx, intent=intent)
        elapsed = time.perf_counter() - t0
        logger.info("Chat [find_mentor] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── ASK MENTOR RECOMMENDATION (AI-powered recommendations) ───
    if intent == "ask_mentor_recommendation":
        recommendations = await recommendation_service.get_recommendations(user_id=payload.user_id)
        answer = "تم تجهيز أفضل الترشيحات لك 🎯" if language == "ar" else "Your best mentor recommendations are ready 🎯"
        elapsed = time.perf_counter() - t0
        logger.info("Chat [ask_mentor_recommendation] %d results (%.2fs)", len(recommendations), elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="recommendation",
            answer=answer,
            recommendations=recommendations,
        )

    # ─── ASK PROGRAM RECOMMENDATION (program suggestions) ───
    if intent == "ask_program_recommendation":
        # Reuse the program recommendation service; map program items to RecommendationItem-like shape
        program_recs = await program_recommendation_service.get_recommendations(user_id=payload.user_id, top_k=10)
        
        # ── NEW (May 2026): Handle empty recommendations with friendly fallback ──
        if not program_recs:
            if language == "ar":
                answer = (
                    "للأسف، لم أجد برامج متاحة حالياً تتطابق مع ملفك الشخصي.\n\n"
                    "قد يكون السبب:\n"
                    "• جميع البرامج المتاحة قد امتلأت 📋\n"
                    "• انتهت آجال التقديم للبرامج المناسبة ⏰\n"
                    "• لم تقدم على أي برامج بعد 🔍\n\n"
                    "يمكنك المحاولة لاحقاً أو البحث عن برامج جديدة."
                )
            else:
                answer = (
                    "Sorry, I couldn't find any active programs matching your profile right now.\n\n"
                    "This might be because:\n"
                    "• Available programs have reached capacity 📋\n"
                    "• Program application deadlines have passed ⏰\n"
                    "• You may not have applied to any programs yet 🔍\n\n"
                    "Try checking back later or search for new programs."
                )
            elapsed = time.perf_counter() - t0
            logger.info("Chat [ask_program_recommendation] no eligible programs found (%.2fs)", elapsed)
            return ChatResponse(
                language=language,
                intent=intent,
                response_type="text",
                answer=answer,
            )
        
        mapped = []
        for p in program_recs:
            # Build short reason from signal fields for chat display
            parts = []
            if p.get("target_level_pass"):
                if p.get("minimum_requirement_exact_match"):
                    parts.append("exact level match")
                elif p.get("minimum_requirement_above_minimum"):
                    parts.append("exceeds minimum level")
            if p.get("education_level_pass"):
                parts.append("education qualified")
            cov = float(p.get("requirement_coverage_score", 0) or 0)
            if cov >= 0.5:
                parts.append(f"{cov:.0%} skill coverage")
            
            # ── NEW (May 2026): Add deadline information to reason ──
            days_left = int(p.get("days_until_deadline", 999) or 999)
            if days_left < 999:
                if days_left <= 0:
                    parts.append("⏰ closing soon")
                elif days_left == 1:
                    parts.append("⏰ 1 day left")
                elif days_left <= 7:
                    parts.append(f"⏰ {days_left} days left")
                else:
                    parts.append(f"⏰ {days_left} days left")
            
            reason = "; ".join(parts).capitalize() + "." if parts else ""

            mapped.append({
                "mentor_id": str(p.get("mentor_id", "")),
                "mentor_name": p.get("mentor_name", ""),
                "domain": p.get("domain", ""),
                "score": float(p.get("pred_score", p.get("score", 0) or 0)),
                "match_percentage": int(p.get("match_percentage", 75) or 75),
                "reason": reason,
            })
        
        answer = "لقد جمعت لك برامج مناسبة 🎯" if language == "ar" else "I've gathered suitable programs for you 🎯"
        elapsed = time.perf_counter() - t0
        logger.info("Chat [ask_program_recommendation] %d results (%.2fs)", len(mapped), elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="recommendation",
            answer=answer,
            recommendations=mapped,
        )

    # ─── RECOMMENDATION EXPLANATION (follow-up: "why this mentor?") ───
    if intent == "recommendation_explanation":
        answer = recommendation_service.explain_recommendation(
            user_id=payload.user_id,
            user_message=payload.message,
            language=language,
        )
        elapsed = time.perf_counter() - t0
        logger.info("Chat [recommendation_explanation] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── TASK HELP ───
    if intent == "task_help":
        answer = await llm_service.chat_task_help(payload.message, language, history=payload.history)
        elapsed = time.perf_counter() - t0
        logger.info("Chat [task_help] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── SUBMIT TASK ───
    if intent == "submit_task":
        answer = await llm_service.chat_submit_task(payload.message, language)
        elapsed = time.perf_counter() - t0
        logger.info("Chat [submit_task] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── ROADMAP REQUEST ───
    if intent == "roadmap_request":
        # Run search and LLM generation in PARALLEL — they don't depend on each other
        search_task = search_service.find_materials(query=payload.message, language=language)
        roadmap_task = llm_service.chat_roadmap(payload.message, language, history=payload.history)
        search_result, llm_roadmap = await asyncio.gather(search_task, roadmap_task)

        materials = [
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "kind": _classify_link(r.get("link", ""), r.get("title", "")),
                "source": r.get("source", ""),
                "reason": r.get("reason", ""),
            }
            for r in search_result.get("results", [])
        ]

        # Combine: LLM roadmap text + found materials
        if materials:
            summary = search_result.get("summary", "")
            if summary:
                combined_answer = f"{llm_roadmap}\n\n---\n\n📚 {summary}"
            else:
                combined_answer = llm_roadmap
        else:
            combined_answer = llm_roadmap

        elapsed = time.perf_counter() - t0
        logger.info("Chat [roadmap_request] %d materials (%.2fs)", len(materials), elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="roadmap" if materials else "text",
            answer=combined_answer,
            materials=materials,
        )

    # ─── MATERIALS REQUEST (videos, articles, courses) ───
    if intent == "materials_request":
        search_result = await search_service.find_materials(query=payload.message, language=language)
        materials = [
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "kind": _classify_link(r.get("link", ""), r.get("title", "")),
                "source": r.get("source", ""),
                "reason": r.get("reason", ""),
            }
            for r in search_result.get("results", [])
        ]
        answer = search_result.get("summary", "")
        if not answer:
            answer = "تم تجميع المواد التعليمية المناسبة 📚" if language == "ar" else "Here are the learning materials I found 📚"
        elapsed = time.perf_counter() - t0
        logger.info("Chat [materials_request] %d results (%.2fs)", len(materials), elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="materials" if materials else "text",
            answer=answer,
            materials=materials,
        )

    # ─── FAQ ───
    if intent == "faq":
        answer = await rag_service.answer_platform_question(payload.message, language, user_context=user_ctx, intent=intent)
        # If RAG can't answer, fallback to LLM instead of returning "unavailable"
        if answer and ("not available" in answer.lower() or "غير متوفرة" in answer):
            answer = await llm_service.chat_general(payload.message, language, history=payload.history)
        elapsed = time.perf_counter() - t0
        logger.info("Chat [%s] (%.2fs)", intent, elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── COMPLAINT (mentor behavior reports) ───
    if intent == "complaint":
        answer = await llm_service.chat_complaint(payload.message, language, history=payload.history)
        elapsed = time.perf_counter() - t0
        logger.info("Chat [complaint] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── SUPPORT REQUEST (technical problems) ───
    if intent == "support_request":
        answer = await llm_service.chat_support(payload.message, language, history=payload.history)
        elapsed = time.perf_counter() - t0
        logger.info("Chat [support_request] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── MENTOR ANALYTICS (mentor-specific: view stats) ───
    if intent == "mentor_analytics":
        answer = (
            "يمكنك عرض تحليلاتك من لوحة المنتيز الرئيسية 📊\n"
            "ستجد هناك عدد المتدربين، الجلسات المكتملة، والتقييمات"
            if language == "ar"
            else "You can view your analytics from the mentor dashboard 📊\n"
            "You'll find the number of mentees, completed sessions, and ratings"
        )
        elapsed = time.perf_counter() - t0
        logger.info("Chat [mentor_analytics] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── MENTOR WORKFLOW HELP (mentor-specific: workflow/communication) ───
    if intent == "mentor_workflow_help":
        answer = (
            "للتواصل مع المتدربين 💬:\n"
            "1. استخدم رسائل المنصة المباشرة\n"
            "2. جدول الجلسات متاح في التقويم\n"
            "3. يمكنك إضافة ملاحظات بعد كل جلسة"
            if language == "ar"
            else "To communicate with your mentees 💬:\n"
            "1. Use the platform's direct messaging\n"
            "2. Session schedule is in your calendar\n"
            "3. You can add notes after each session"
        )
        elapsed = time.perf_counter() - t0
        logger.info("Chat [mentor_workflow_help] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── GENERAL QUESTION (fallback with guardrails) ───
    answer = await llm_service.chat_general(payload.message, language, history=payload.history)
    elapsed = time.perf_counter() - t0
    logger.info("Chat [general_question] (%.2fs)", elapsed)
    return ChatResponse(
        language=language,
        intent="general_question",
        response_type="text",
        answer=answer,
    )
