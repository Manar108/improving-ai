"""Mentor Chat Route — mentor-specific chatbot endpoint.

Routes mentor messages through mentor intent classification and handlers.
Reuses the same ChatRequest/ChatResponse schema as mentee chatbot.
"""

import asyncio
import logging
import time

from fastapi import APIRouter

from config import settings
from schemas import ChatRequest, ChatResponse
from services.intent_service import intent_service
from services.mentor_intent_service import mentor_intent_service
from services.llm_service import llm_service
from services.mentor_response_service import mentor_response_service
from services.mentor_context_service import mentor_context_service
from services.user_context_service import user_context_service

logger = logging.getLogger(__name__)
router = APIRouter()


# ────────────────────────────────────────────────────────────────────
# Main mentor chat endpoint
# ────────────────────────────────────────────────────────────────────

@router.post("/mentor-chat", response_model=ChatResponse)
async def mentor_chat(payload: ChatRequest) -> ChatResponse:
    """Handle mentor chatbot messages.
    
    Reuses the same request/response format as mentee chatbot, but with
    mentor-specific intent classification and handlers.
    """
    t0 = time.perf_counter()
    
    # ── Step 1: Detect language ──
    language = mentor_intent_service.detect_language(payload.message)

    # ── Step 2: Classify intent (MENTOR-SPECIFIC) ──
    if settings.GROQ_API_KEY:
        try:
            intent = await mentor_intent_service.detect_intent_async(payload.message)
        except Exception as e:
            logger.warning("Mentor intent detection failed: %s", e)
            intent = "general_question"
    else:
        intent = "general_question"

    # ── Step 3: Load mentor context ──
    mentor_ctx = {}
    if payload.user_id:
        user_ctx = user_context_service.get_user_context(payload.user_id)
        if user_ctx.get("user_id"):
            # Verify user is actually a mentor
            role = user_ctx.get("role", "")
            if role not in ("mentor", "both"):
                elapsed = time.perf_counter() - t0
                logger.warning("Non-mentor attempted mentor chat: role=%s (%.2fs)", role, elapsed)
                return ChatResponse(
                    language=language,
                    intent="off_topic",
                    response_type="text",
                    answer="هذا الـ chatbot للمرشدين فقط 📚" if language == "ar" else "This chatbot is for mentors only 📚",
                )
            mentor_ctx = {
                "user_id": user_ctx.get("user_id"),
                "first_name": user_ctx.get("first_name"),
                "last_name": user_ctx.get("last_name"),
                "domain_name": user_ctx.get("domain_name"),
                "role": role,
            }
            logger.info(
                "Mentor context loaded: %s %s | domain=%s",
                mentor_ctx.get("first_name"),
                mentor_ctx.get("last_name"),
                mentor_ctx.get("domain_name"),
            )

    # ── Step 4: Normalize history ──
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
            content = content[-MAX_MESSAGE_CHARS:]
        trimmed_history.append({"role": role, "content": content})
    if len(trimmed_history) != len(history):
        logger.debug("Trimmed history from %d to %d messages", len(history), len(trimmed_history))
    payload.history = trimmed_history

    # ── Step 5: Route to handler ──

    # ─── GREETING ───
    if intent == "greeting":
        elapsed = time.perf_counter() - t0
        logger.info("Mentor [greeting] (%.2fs)", elapsed)
        answer = await mentor_response_service.greet(language)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── OFF-TOPIC ───
    if intent == "off_topic":
        elapsed = time.perf_counter() - t0
        _profanity_markers = [
            "fuck", "shit", "bitch", "asshole", "joke", "weather", "news",
            "يلعن", "يا حمار", "كلب", "حيوان", "نكتة",
        ]
        msg_lower = payload.message.lower()
        is_abusive = any(w in msg_lower for w in _profanity_markers)
        if is_abusive:
            logger.warning("Mentor [off_topic/abuse] (%.2fs)", elapsed)
        answer = await mentor_response_service.off_topic(language)
        logger.info("Mentor [off_topic] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── FAQ ───
    if intent == "faq":
        answer = await mentor_response_service.faq(payload.message, language, mentor_ctx)
        elapsed = time.perf_counter() - t0
        logger.info("Mentor [faq] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── MATERIALS REQUEST ───
    if intent == "materials_request":
        answer, materials = await mentor_response_service.materials(payload.message, language)
        elapsed = time.perf_counter() - t0
        logger.info("Mentor [materials_request] %d results (%.2fs)", len(materials), elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="materials" if materials else "text",
            answer=answer,
            materials=materials,
        )

    # ─── MENTOR ANALYTICS ───
    if intent == "mentor_analytics":
        if not payload.user_id:
            answer = (
                "لا يمكن عرض الإحصائيات بدون تسجيل دخول 😊"
                if language == "ar"
                else "Please log in to view your statistics 😊"
            )
        else:
            answer = await mentor_response_service.analytics(
                payload.message, language, payload.user_id
            )
        elapsed = time.perf_counter() - t0
        logger.info("Mentor [mentor_analytics] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── MENTOR WORKFLOW HELP ───
    if intent == "mentor_workflow_help":
        if not payload.user_id:
            answer = (
                "لا يمكن عرض معلومات المنتيز بدون تسجيل دخول 😊"
                if language == "ar"
                else "Please log in to manage your mentees 😊"
            )
        else:
            answer = await mentor_response_service.workflow_help(
                payload.message, language, payload.user_id
            )
        elapsed = time.perf_counter() - t0
        logger.info("Mentor [mentor_workflow_help] (%.2fs)", elapsed)
        return ChatResponse(
            language=language,
            intent=intent,
            response_type="text",
            answer=answer,
        )

    # ─── GENERAL QUESTION ───
    answer = await mentor_response_service.general_question(
        payload.message, language, payload.history
    )
    elapsed = time.perf_counter() - t0
    logger.info("Mentor [general_question] (%.2fs)", elapsed)
    return ChatResponse(
        language=language,
        intent="general_question",
        response_type="text",
        answer=answer,
    )
