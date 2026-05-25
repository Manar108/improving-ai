"""Mentor Intent Classification Service — LLM-primary classifier for mentor chatbot.

Classifies mentor queries into mentor-specific intents (analytics, workflow, FAQs, materials).
Reuses the same LLM infrastructure and caching as the mentee chatbot.

Mentor Intents:
1. greeting — Social greeting
2. faq — Platform rules/features ("How to create a program?", "Pricing?")
3. materials_request — Teaching resources ("Exercises for beginners", "Interview questions")
4. mentor_analytics — Mentor statistics ("How many mentees enrolled?", "Top programs?")
5. mentor_workflow_help — Help managing mentorships ("How to review submissions?")
6. ask_document — Questions about uploaded documents
7. general_question — General educational questions
8. off_topic — Unrelated/profanity
"""

import json
import logging
import re
import time
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_INTENT_MODEL = "llama-3.1-8b-instant"

# ────────────────────────────────────────────────────────────────────
# Mentor-specific valid intents
# ────────────────────────────────────────────────────────────────────

MENTOR_VALID_INTENTS = frozenset([
    "greeting",
    "faq",
    "materials_request",
    "mentor_analytics",
    "mentor_workflow_help",
    "general_question",
    "off_topic",
])

_MENTOR_INTENT_ALIASES: dict[str, str] = {}


# ────────────────────────────────────────────────────────────────────
# Language detection (reused from mentee)
# ────────────────────────────────────────────────────────────────────

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: str) -> str:
    if not text:
        return "en"
    if ARABIC_CHAR_RE.search(text):
        return "ar"
    return "en"


# ────────────────────────────────────────────────────────────────────
# Intent cache (same TTL as mentee)
# ────────────────────────────────────────────────────────────────────

_intent_cache: dict[str, tuple[float, str]] = {}
_INTENT_CACHE_TTL = 300  # 5 minutes


def _get_cached_intent(text: str) -> Optional[str]:
    key = text.strip().lower()
    entry = _intent_cache.get(key)
    if entry is None:
        return None
    ts, intent = entry
    if time.monotonic() - ts > _INTENT_CACHE_TTL:
        del _intent_cache[key]
        return None
    return intent


def _set_cached_intent(text: str, intent: str) -> None:
    _intent_cache[text.strip().lower()] = (time.monotonic(), intent)


# ────────────────────────────────────────────────────────────────────
# Intent extraction and resolution
# ────────────────────────────────────────────────────────────────────

def _resolve_intent(label: str) -> Optional[str]:
    """Return a valid mentor intent, resolving aliases."""
    label = label.strip().lower()
    if label in MENTOR_VALID_INTENTS:
        return label
    mapped = _MENTOR_INTENT_ALIASES.get(label)
    if mapped and mapped in MENTOR_VALID_INTENTS:
        logger.debug("Resolved alias: %s → %s", label, mapped)
        return mapped
    return None


def _extract_intent(raw: str) -> Optional[str]:
    """Extract valid mentor intent from LLM output.
    
    Tries multiple parsing strategies:
      1. Direct JSON parse
      2. Regex-extracted JSON object
      3. Plain-text match
      4. Substring scan
    """
    raw = raw.strip()

    # Strategy 1: Direct JSON parse
    try:
        d = json.loads(raw)
        if isinstance(d, dict):
            i = _resolve_intent(d.get("intent", ""))
            if i:
                return i
    except Exception:
        pass

    # Strategy 2: Extract JSON from surrounding text
    m = re.search(r'\{[^}]+\}', raw)
    if m:
        try:
            d = json.loads(m.group())
            i = _resolve_intent(d.get("intent", ""))
            if i:
                return i
        except Exception:
            pass

    # Strategy 3: Plain text
    candidate = raw.strip().strip("\"'").lower()
    resolved = _resolve_intent(candidate)
    if resolved:
        return resolved

    # Strategy 4: Substring scan
    raw_lower = raw.lower()
    for intent in MENTOR_VALID_INTENTS:
        if intent in raw_lower:
            return intent

    return None


# ────────────────────────────────────────────────────────────────────
# System prompt for mentor intent classification
# ────────────────────────────────────────────────────────────────────

_MENTOR_SYSTEM_PROMPT = """\
You are an intent classifier for a mentorship platform chatbot designed for MENTORS (teachers).

Classify the mentor's message into EXACTLY ONE intent.
Return ONLY valid JSON: {"intent":"<label>"}
Do NOT add any explanation, markdown, or extra text.

═══ MENTOR INTENTS (7 total) ═══

1. greeting
   Mentor is saying hello or opening the conversation.
   Examples: "hi", "hello", "مرحبا", "ازيك", "صباح الخير"

2. faq
   Mentor asks about PLATFORM RULES, FEATURES, or HOW TO USE the platform.
   This includes: how to create a program, how to review submissions, pricing, how mentorship works.
   ⚠️ NOT asking for MENTEES ("find mentees for my program" → mentor_workflow_help)
   Examples:
     AR: "إزاي أنشئ برنامج", "الرسوم قد ايه", "ازاي أراجع الواجبات",
         "شنو الميزات اللي عندي", "ازاي ابقى مرشد"
     EN: "How to create a program?", "What are the features?", "How to review submissions?",
         "How much does it cost?", "How does mentorship work?"

3. materials_request
   Mentor asks for TEACHING MATERIALS or RESOURCES to give to mentees.
   This includes: interview questions, exercises, quizzes, project ideas, assignment ideas.
   ⚠️ NOT asking for his own learning (→ general_question).
   Examples:
     AR: "إعطني أسئلة مقابلات Python", "تمارين للمبتدئين", "أفكار مشاريع AI",
         "أسئلة امتحان JavaScript", "فكرة تاسك سهل"
     EN: "Interview questions for backend", "Python exercises for beginners",
         "Project ideas for machine learning", "Quiz questions for advanced React"

4. mentor_analytics
   Mentor wants STATISTICS about his programs and mentees.
   Key signals: "statistics", "analytics", "how many", "كام", "إحصائيات", "أداء"
   Examples:
     AR: "كام منتي في برنامجي", "ايه أفضل برنامج", "كام واحد خلص البرنامج",
         "أداء البرنامج إيه", "مين أفضل منتي"
     EN: "How many mentees in my program?", "What's my engagement?",
         "Top performing program?", "How many completed?"

5. mentor_workflow_help
   Mentor needs help with MANAGING his mentorships and mentees.
   This includes: how to find mentees, how to contact mentees, how to structure sessions,
   how to give feedback, how to manage applications, mentee communication.
   Examples:
     AR: "إزاي أتواصل مع المنتيز", "كيف أختار منتي", "إزاي أدي feedback",
         "ازاي أنظم الجلسات", "المنتيز التانية اللي تقدموا"
     EN: "How to contact mentees?", "How to give feedback?",
         "How to structure sessions?", "Who are my mentees?"

6. general_question
   Any other educational or learning-related question (mentor self-learning).
   Examples: "شرحلي OOP", "ايه الفرق بين الـ frameworks دي"

7. off_topic
   Unrelated topic, profanity, or abuse.
   Examples: "tell me a joke", "يا حمار"

═══ KEY RULES ═══

MATERIALS vs WORKFLOW:
  "أعطني تمارين" → materials_request (resources to GIVE mentees)
  "ازاي أسلم التمارين" → mentor_workflow_help (managing delivery)

ANALYTICS vs WORKFLOW:
  "كام منتي" → mentor_analytics (statistics)
  "إزاي أتواصل معهم" → mentor_workflow_help (management/communication)

ALWAYS:
  • Profanity/insults → off_topic
  • Unsure → general_question

Return ONLY {"intent":"..."} — no explanation."""


# ────────────────────────────────────────────────────────────────────
# LLM-based intent detection for mentors
# ────────────────────────────────────────────────────────────────────

async def detect_intent_async(text: str) -> str:
    """Classify mentor intent using LLM.
    
    Fallback chain:
      Cache → LLM → keyword fallback → general_question
    """
    cached = _get_cached_intent(text)
    if cached is not None:
        logger.debug("Mentor intent cache HIT: %s", text[:60])
        return cached

    if not settings.GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY — using keyword fallback")
        intent = _keyword_fallback(text)
        _set_cached_intent(text, intent)
        return intent

    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _INTENT_MODEL,
        "messages": [
            {"role": "system", "content": _MENTOR_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 30,
    }

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(_GROQ_URL, headers=headers, json=payload)

        if resp.status_code == 429:
            logger.warning("Mentor intent LLM rate-limited — falling back to keywords")
            intent = _keyword_fallback(text)
            _set_cached_intent(text, intent)
            return intent

        if resp.status_code != 200:
            logger.warning("Mentor intent LLM HTTP %d — falling back", resp.status_code)
            intent = _keyword_fallback(text)
            _set_cached_intent(text, intent)
            return intent

        content = resp.json()["choices"][0]["message"]["content"].strip()
        intent = _extract_intent(content)

        if intent:
            logger.info("Mentor LLM intent: '%s' → %s", text[:60], intent)
            _set_cached_intent(text, intent)
            return intent

        logger.warning("Mentor intent parse failed — falling back to keywords")
        intent = _keyword_fallback(text)
        _set_cached_intent(text, intent)
        return intent

    except Exception as exc:
        logger.warning("Mentor intent LLM error: %s — falling back", exc)
        intent = _keyword_fallback(text)
        _set_cached_intent(text, intent)
        return intent


# ────────────────────────────────────────────────────────────────────
# Keyword fallback (BACKUP ONLY)
# ────────────────────────────────────────────────────────────────────

def _keyword_fallback(text: str) -> str:
    """Classify mentor intent using keywords (fallback only)."""
    text_lower = text.lower()

    # Check for profanity/off-topic
    profanity_markers = [
        "fuck", "shit", "bitch", "joke", "weather", "news",
        "يلعن", "يا حمار", "كلب", "نكتة", "الجو",
    ]
    if any(p in text_lower for p in profanity_markers):
        return "off_topic"

    # Materials request keywords
    if any(k in text_lower for k in ["exercise", "تمرين", "interview", "مقابلة",
                                       "quiz", "امتحان", "question", "أسئلة",
                                       "assignment", "واجب", "project", "مشروع",
                                       "idea", "فكرة"]):
        return "materials_request"

    # Analytics keywords
    if any(k in text_lower for k in ["how many", "كام", "statistic", "إحصاء",
                                       "analytics", "تحليل", "performance", "أداء",
                                       "engagement", "top", "أفضل"]):
        return "mentor_analytics"

    # Workflow help keywords
    if any(k in text_lower for k in ["contact", "تواصل", "feedback", "تقييم",
                                       "session", "جلسة", "review", "مراجعة",
                                       "application", "تقديم", "mentee", "منتي",
                                       "manage", "إدارة"]):
        return "mentor_workflow_help"

    # FAQ keywords
    if any(k in text_lower for k in ["how to", "إزاي", "how", "كيف",
                                       "create", "أنشئ", "feature", "ميزة",
                                       "rule", "قاعدة", "cost", "رسوم"]):
        return "faq"

    # Greeting keywords
    if any(k in text_lower for k in ["hello", "hi", "hey", "مرحبا", "ازيك",
                                       "السلام", "صباح", "مساء"]):
        return "greeting"

    return "general_question"


# ────────────────────────────────────────────────────────────────────
# Public interface (same as mentee intent service)
# ────────────────────────────────────────────────────────────────────

class MentorIntentService:
    """Stateless intent classifier for mentor chatbot."""

    @staticmethod
    def detect_language(text: str) -> str:
        return detect_language(text)

    @staticmethod
    async def detect_intent_async(text: str) -> str:
        return await detect_intent_async(text)


# Singleton instance
mentor_intent_service = MentorIntentService()
