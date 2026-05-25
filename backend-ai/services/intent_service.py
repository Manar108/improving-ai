"""Intent classification service — LLM-primary with keyword fallback.

Uses a refined prompt + llama-3.1-8b-instant for speed (<1s typical).
Classifies into exactly one of thirteen intents.

Architecture:
  1. LLM (primary)   — always attempted first via detect_intent_async
  2. Keyword fallback — ONLY on API failure, parse failure, or rate limit
  3. Safety net       — invalid intent → keyword → general_question
"""

import json
import logging
import re
import time
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

# ✅ ADDED: Import error handler
from services.error_handling import ChatbotErrorHandler

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Use the FAST model for intent classification only
_INTENT_MODEL = "llama-3.1-8b-instant"

# ────────────────────────────────────────────────────────────────────
# Valid intents — the single source of truth
# ────────────────────────────────────────────────────────────────────

VALID_INTENTS = frozenset([
    "greeting",
    "find_mentor",
    "ask_mentor_recommendation",
    "ask_program_recommendation",
    "recommendation_explanation",
    "task_help",
    "submit_task",
    "roadmap_request",
    "materials_request",
    "faq",
    "complaint",
    "support_request",
    "general_question",
    "off_topic",
    "mentor_analytics",      # Mentor-specific: view analytics/stats
    "mentor_workflow_help",  # Mentor-specific: workflow/communication help
])

# Map legacy/removed intents to their replacements for backward compatibility
_INTENT_ALIASES: dict[str, str] = {
    "explanation_request": "general_question",  # merged: same handler
}


# ────────────────────────────────────────────────────────────────────
# Language detection
# ────────────────────────────────────────────────────────────────────


def detect_language(text: str) -> str:
    if not text:
        return "en"
    if ARABIC_CHAR_RE.search(text):
        return "ar"
    return "en"


# ────────────────────────────────────────────────────────────────────
# LLM response parser
# ────────────────────────────────────────────────────────────────────


def _resolve_intent(label: str) -> Optional[str]:
    """Return a valid intent, resolving aliases for removed/merged intents."""
    label = label.strip().lower()
    if label in VALID_INTENTS:
        return label
    # Check aliases (e.g. explanation_request → general_question)
    mapped = _INTENT_ALIASES.get(label)
    if mapped and mapped in VALID_INTENTS:
        logger.debug("Resolved alias: %s → %s", label, mapped)
        return mapped
    return None


def _extract_intent(raw: str) -> Optional[str]:
    """Extract a valid intent from LLM output.

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

    # Strategy 3: Plain text (e.g. LLM just returned the label)
    candidate = raw.strip().strip("\"'").lower()
    resolved = _resolve_intent(candidate)
    if resolved:
        return resolved

    # Strategy 4: Substring scan (check aliases + valid intents)
    raw_lower = raw.lower()
    for intent in VALID_INTENTS:
        if intent in raw_lower:
            return intent
    for alias, target in _INTENT_ALIASES.items():
        if alias in raw_lower:
            return target

    return None


# ────────────────────────────────────────────────────────────────────
# System prompt — clear, non-overlapping definitions with examples
# ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an intent classifier for a mentorship/learning platform chatbot.
Classify the user message into EXACTLY ONE intent.
Return ONLY valid JSON: {"intent":"<label>"}
Do NOT add any explanation, markdown, or extra text.

═══ INTENTS (13 total) ═══

1. greeting
   User is saying hello/hi or opening a conversation with a social greeting.
   Examples: "hi", "hello", "مرحبا", "السلام عليكم", "ازيك", "أهلاً", "صباح الخير"

2. find_mentor
   User wants to SEARCH, BROWSE, or GET INFO about mentors on the platform.
   Key signal: asking about WHO exists, ratings, stats, or filtering mentors.
   ⚠️ NOT "recommend FOR ME" (→ ask_mentor_recommendation). NOT "become a mentor" (→ faq).
   Examples: "مين أحسن mentor في AI", "best mentor in web dev", "كام مرشد في البرمجة", "top rated mentors", "show me mentors in data science"

3. ask_mentor_recommendation
   User wants the AI to RECOMMEND mentors FOR THEM PERSONALLY based on their profile.
   Key signal: "رشحلي", "recommend for me", "suggest for me", "مناسب ليا", "suitable for me".
   Examples: "رشحلي مرشدين", "recommend mentors for me", "اقترح عليا مرشدين مناسبين", "I need a mentor recommendation"

4. ask_program_recommendation
   User wants PROGRAM/COURSE recommendations from the platform's programs.
   Key signal: mentions "program"/"برنامج"/"برامج" combined with recommendation language.
   Examples: "رشحلي برامج", "recommend programs", "عايز برنامج إرشاد", "برامج مناسبة ليا"

5. recommendation_explanation
   User asks WHY a specific mentor/program was recommended — a FOLLOW-UP question.
   Key signal: "ليه", "why", "إيه السبب", "ايه اللي خلاه" — asking about a PAST recommendation.
   Examples: "ليه رشحتلي ده؟", "why this mentor?", "إيه اللي خلاه مناسب؟", "why not the other one?"

6. task_help
   User needs help with a TASK, ASSIGNMENT, PROJECT, or CODE — INCLUDING frustrated/struggling students.
   This is a BROAD intent covering ALL of:
     • "I don't understand the task" (understanding)
     • "Explain this code to me" (code explanation)
     • "My code isn't working" (debugging)
     • "The task is too hard, I'm lost" (frustration with ACADEMIC work)
     • "I'm stuck on the project" (being stuck)
     • "I'm frustrated/exhausted with this assignment" (emotional struggle with WORK)
   ⚠️ CRITICAL: A student expressing frustration, exhaustion, or feeling lost about their TASKS/CODE/ASSIGNMENTS
      is STILL task_help. They need encouragement + guidance, NOT a complaint form.
   ⚠️ If they mention code, function, error, assignment, task, project, or homework → task_help.
   Examples:
     AR: "مش فاهم التاسك", "شرحلي الكود ده", "التاسك صعبه أوي", "أنا تايه في المشروع",
         "الكود مش شغال عندي", "مش عارف أحل المشكلة دي", "أنا محبط من التاسك",
         "تعبت من الكود ده", "الفنكشن دي مش راضيه تشتغل", "مش فاهم حاجة",
         "الواجب ده صعب جداً ومش عارف أعمل ايه", "أنا ضايع خالص في البروجكت"
     EN: "help me with this code", "can you explain this function", "I can't solve this problem",
         "I'm so frustrated with this task", "this assignment is killing me", "I'm stuck",
         "my code keeps crashing", "I don't understand anything", "debugging help please"

7. submit_task
   User asks HOW TO SUBMIT work, upload assignments, or asks about DEADLINES.
   Key signal: asking about the submission PROCESS or when something is due, NOT about understanding the task.
   Examples: "إزاي أسلم التاسك", "how to submit?", "الديدلاين امتى", "فين أرفع الشغل"

8. roadmap_request
   User wants a LEARNING ROADMAP, study plan, or structured learning path to follow.
   Key signal: asking for a PLAN, PATH, or SEQUENCE to learn a topic.
   ⚠️ NOT asking for a specific video/article (→ materials_request).
   Examples: "عايز roadmap للـ AI", "learning path for Python", "كيف أبدأ أتعلم web dev", "خطة تعلم", "ازاي أبدأ"

9. materials_request
   User wants specific LEARNING MATERIALS: videos, articles, courses, tutorials.
   Key signal: asking for a specific CONTENT TYPE (video, article, course, resource).
   ⚠️ NOT a learning roadmap/plan (→ roadmap_request). NOT help with a task (→ task_help).
   Examples: "هات فيديو python", "عايز فيديو عن ML", "مقالات عن JavaScript", "كورسات AI"

10. faq
    User asks about PLATFORM RULES, features, registration, pricing, or how the platform works.
    Key signal: questions about the PLATFORM ITSELF or its processes.
    Examples: "مدة البرنامج قد ايه", "هل المنصة مجانية", "ازاي ابقى مرشد", "how to register?"

11. complaint
    User wants to REPORT or COMPLAIN about a specific PERSON's behavior on the platform.
    ⚠️ The complaint MUST be about a PERSON (mentor, another user) — not about code, tasks, or the system.
    ⚠️ Look for: a person being named/referenced + negative behavior (rude, ignored me, harassed me, bad attitude).
    ⚠️ NOT a frustrated student (→ task_help). NOT a system bug (→ support_request).
    Examples:
      AR: "عايز أشتكي من المرشد", "المرشد وحش", "المرشد مش بيرد عليا خالص",
          "المرشد بيتجاهلني", "سوء سلوك", "تحرش", "المنتور قليل الأدب"
      EN: "I want to report the mentor", "mentor is rude", "harassment", "mentor ignores me"

12. support_request
    User has a TECHNICAL/SYSTEM problem with the PLATFORM — something is broken, errors, can't access.
    ⚠️ ONLY for PLATFORM BUGS (website, app, login, upload broken).
    ⚠️ NOT for code not working in an assignment (→ task_help).
    Examples: "الموقع فيه error", "مش بيرفع الملف", "can't login", "الصفحة مش بتفتح"

13. general_question
    Any educational question, concept explanation, learning advice, or career question that doesn't fit above.
    This includes: "what is X?", "explain Y", "difference between A and B", "what language should I learn?",
    "tips for job interviews", etc.
    Examples: "ايه machine learning", "شرحلي OOP", "what is REST API", "ايه الفرق بين Python و Java",
             "what language should I learn first", "tips for job interviews", "كيف أحسن مستواي"

14. off_topic
    ONLY if truly UNRELATED to education, mentorship, learning, or the platform. Also profanity/insults.
    Examples: "الجو عامل ايه", "tell me a joke", "what's the news today"

═══ EMOTIONAL INTELLIGENCE RULES (READ CAREFULLY) ═══

The BIGGEST mistake is confusing a FRUSTRATED STUDENT with a COMPLAINT.

ASK: "Is the user upset about a PERSON's behavior, or upset about their OWN struggle with work?"
• Upset about THEIR OWN struggle → task_help (they need help and encouragement)
• Upset about a PERSON's behavior → complaint (they need to file a report)

FRUSTRATION SIGNALS → task_help:
  "أنا محبط" / "أنا تعبت" / "مش قادر أكمل" / "أنا زهقت" / "خلاص مش قادر"
  "مش فاهم حاجة" / "ضايع" / "تايه" / "يائس" / "مش عارف أعمل ايه"
  "I'm frustrated" / "I'm exhausted" / "I'm lost" / "I can't do this" / "I give up"
  → ALL of these are task_help. The student needs support, not a complaint form.

COMPLAINT SIGNALS → complaint:
  "المرشد..." + negative behavior verb (مش بيرد / بيتجاهلني / وحش / سيء)
  "عايز أشتكي" / "عايز أقدم شكوى" / "أبلغ عن"
  "mentor is..." + negative (rude / abusive / ignoring me)
  → A specific PERSON is being accused of bad behavior.

═══ OTHER DISAMBIGUATION RULES ═══

CODE HELP vs PLATFORM BUG:
  "الكود مش شغال" → task_help (their assignment code)
  "الموقع مش شغال" / "الصفحة مش بتفتح" → support_request (platform broken)

EXPLAIN CONCEPT vs EXPLAIN CODE:
  "شرحلي OOP" / "what is REST API" → general_question (abstract concept)
  "شرحلي الكود ده" / "الفنكشن دي بتعمل ايه" → task_help (specific code)

RECOMMENDATIONS:
  "رشحلي مرشد" → ask_mentor_recommendation
  "رشحلي برامج" → ask_program_recommendation
  "ليه رشحتلي ده" → recommendation_explanation
  "مين أحسن mentor" → find_mentor (browsing, not personal recommendation)

BECOME A MENTOR:
  "ازاي ابقى مرشد" / "how to become a mentor" → faq (NOT find_mentor)

ALWAYS:
  • Profanity/insults → off_topic
  • شكوى/أشتكي/complain/harassment/rude/سوء سلوك → complaint
  • If unsure between task_help and complaint → check if a PERSON is blamed → if no person, it's task_help
  • If unsure generally → general_question

Return ONLY {"intent":"..."} — no explanation."""


# ────────────────────────────────────────────────────────────────────
# LLM-based intent detection (PRIMARY)
# ────────────────────────────────────────────────────────────────────


# ── Simple TTL cache for intent classification (300s) ──
_intent_cache: dict[str, tuple[float, str]] = {}
_INTENT_CACHE_TTL = 300  # 5 minutes


def _get_cached_intent(text: str) -> str | None:
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


async def detect_intent_async(text: str) -> str:
    """Classify intent using the fast 8B model (primary classifier).

    Fallback chain:
      Cache → LLM → keyword fallback → general_question
    
    ✅ ADDED: Error handling with ChatbotErrorHandler
    """
    cached = _get_cached_intent(text)
    if cached is not None:
        logger.debug("Intent cache HIT: %s", text[:60])
        return cached

    if not settings.GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY configured — using keyword fallback")
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
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 30,  # intent JSON is tiny — cap output
    }

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(_GROQ_URL, headers=headers, json=payload)

        # ── Rate limited ──
        if resp.status_code == 429:
            logger.warning("Intent LLM rate-limited — falling back to keywords")
            # ✅ ADDED: Handle rate limit with error handler
            intent = ChatbotErrorHandler.handle_rate_limit()
            # Still use keyword fallback for intent
            intent = _keyword_fallback(text)
            _set_cached_intent(text, intent)
            return intent

        # ── Other API errors ──
        if resp.status_code != 200:
            logger.warning(
                "Intent LLM HTTP %d: %s — falling back to keywords",
                resp.status_code, resp.text[:120],
            )
            # ✅ ADDED: Log intent service error
            ChatbotErrorHandler.handle_intent_service_error()
            intent = _keyword_fallback(text)
            _set_cached_intent(text, intent)
            return intent

        # ── Parse LLM response ──
        content = resp.json()["choices"][0]["message"]["content"].strip()
        intent = _extract_intent(content)

        if intent:
            logger.info("LLM intent: '%s' → %s", text[:60], intent)
            _set_cached_intent(text, intent)
            return intent

        # ── Parse failure: LLM returned something but we can't extract ──
        logger.warning(
            "Intent parse failed (LLM returned: '%s') — unclear intent, using general_question",
            content[:80],
        )
        # ✅ ADDED: Handle unclear intent with error handler
        intent = ChatbotErrorHandler.handle_unclear_intent()
        _set_cached_intent(text, intent)
        return intent if intent else _keyword_fallback(text)

    except Exception as exc:
        logger.warning("Intent LLM error: %s — falling back to keywords", exc)
        # ✅ ADDED: Use error handler for exceptions
        fallback_intent = ChatbotErrorHandler.handle_intent_service_error()
        if isinstance(fallback_intent, str):
            # If error handler returns a string (intent), use it
            _set_cached_intent(text, fallback_intent)
            return fallback_intent
        # Otherwise use keyword fallback
        intent = _keyword_fallback(text)
        _set_cached_intent(text, intent)
        return intent


# ────────────────────────────────────────────────────────────────────
# Keyword fallback (BACKUP ONLY — used when LLM is unavailable)
# ────────────────────────────────────────────────────────────────────


def _keyword_fallback(text: str) -> str:
    """Priority-ordered keyword fallback — BACKUP classifier only.

    Used when:
      - API fails / rate limited
      - Response parsing fails
      - No API key configured

    Priority order (most specific → least specific):
      1. Greeting (exact match)
      2. Complaint (check before generic "mentor" words)
      3. Support request (technical issues)
      4. Submit task (deadlines, submission)
      5. Task help (learning support)
      6. Materials request (videos, articles, courses)
      7. Explanation request (what is X, explain X)
      8. Roadmap request (learning path/plan)
      9. Ask mentor recommendation (personalized)
     10. Find mentor (search/browse)
     11. Off-topic
     12. FAQ (platform how-to)
     13. General question (final fallback)
    """
    t = text.lower().strip()
    logger.info("Keyword fallback triggered for: '%s'", t[:60])

    # ─── 0. PROFANITY / ABUSE (check first — always off_topic) ───
    profanity_keywords = [
        # English
        "fuck", "shit", "bitch", "asshole", "bastard", "dick", "pussy",
        "stfu", "wtf", "idiot", "stupid bot", "useless",
        "dumb", "trash", "garbage", "suck",
        # Arabic
        "يلعن", "كسمك", "كسم", "ابن المتناكة", "متناك", "عرص", "شرموط",
        "يا حمار", "يا غبي", "كلب", "حيوان", "ولاد الوسخة",
        "احا", "اخرس", "يا واطي", "يا قذر", "ابن الكلب",
        "زبالة", "تافه", "يا عبيط",
    ]
    if any(k in t for k in profanity_keywords):
        return "off_topic"

    # ─── 1. GREETING (exact prefix match) ───
    greeting_words = [
        "hi", "hello", "hey", "yo", "sup",
        "اهلا", "اهلاً", "سلام", "مرحبا", "مرحباً", "السلام عليكم",
        "صباح الخير", "صباح النور", "مساء الخير", "تصبح على خير",
        "كيفك", "كيف حالك", "كيف أخبارك", "ازيك", "أنت كويس",
    ]
    if any(
        t == g or t.startswith(g + " ") or t.startswith(g + "،") or t.startswith(g + ".")
        for g in greeting_words
    ):
        return "greeting"

    # ─── 2. COMPLAINT (check early — before "mentor" triggers find_mentor) ───
    complaint_keywords = [
        # Arabic: complaint actions
        "شكوى", "اشتكي", "بشتكي", "اشتك", "أشتكي",
        "بلاغ", "أبلغ عن", "ابلغ عن",
        "اقدم شكوى", "أقدم شكوى",
        # Arabic: behavior descriptors (specific)
        "سوء سلوك", "مش محترم", "غير محترم", "قلة احترام",
        "شتم", "شتمني", "تحرش", "إساءة", "اساءة",
        "المرشد وحش", "المرشد سيء", "مينتوري وحش",
        # English
        "complaint", "complain", "report harassment", "report abuse",
        "rude", "abusive", "harassment", "misconduct", "disrespectful",
    ]
    if any(k in t for k in complaint_keywords):
        return "complaint"

    # ─── 3. SUPPORT REQUEST (technical problems) ───
    support_keywords = [
        # Arabic: technical issues
        "مش شغال", "مش بيشتغل", "مش بيرفع", "مش عارف أرفع",
        "مش بيفتح", "الصفحة مش", "مش بتفتح",
        "فيه مشكلة", "فيه error", "فيه خطأ", "فيه بق",
        "مش بيحمل", "مش بيظهر",
        # English
        "not working", "doesn't work", "can't upload", "cannot upload",
        "error", "bug", "broken", "crashed", "system error",
        "can't login", "cannot login", "page not loading",
    ]
    if any(k in t for k in support_keywords):
        return "support_request"

    # ─── 4. SUBMIT TASK (deadlines, submission methods) ───
    submit_task_keywords = [
        # Arabic
        "أسلم", "اسلم", "تسليم", "سلم", "ارفع الشغل",
        "ديدلاين", "الديدلاين", "الموعد النهائي",
        "فين أسلم", "فين اسلم", "أين أسلم", "فين أرفع",
        "إزاي أسلم", "ازاي اسلم", "كيف أسلم", "كيفية التسليم",
        "الديدلاين امتى", "امتى الموعد", "آخر موعد للتسليم",
        # English
        "submit", "submission", "turn in", "hand in",
        "deadline", "how to submit", "where to submit",
    ]
    if any(k in t for k in submit_task_keywords):
        return "submit_task"

    # ─── 5. TASK HELP (understanding / solving tasks / frustrated students) ───
    task_help_keywords = [
        # Arabic: task/assignment words
        "مش فاهم التاسك", "ما فاهمتش", "التاسك صعب", "التاسك صعبة",
        "ساعدني في التاسك", "ساعدني في الواجب",
        "تاسك", "واجب", "assignment",
        # Arabic: code help
        "شرحلي الكود", "الكود ده", "الكود مش شغال", "الفنكشن",
        "مش عارف أحل", "مش عارف احل", "ساعدني في الكود",
        # Arabic: frustration with WORK (NOT a person)
        "أنا محبط", "انا محبط", "أنا تعبت", "انا تعبت",
        "مش قادر أكمل", "مش قادر اكمل", "خلاص مش قادر",
        "أنا ضايع", "انا ضايع", "أنا تايه", "انا تايه",
        "مش فاهم حاجة", "مش فاهم اي حاجة",
        "يائس", "محبط", "تعبت من الكود", "تعبت من التاسك",
        "صعب أوي", "صعبة أوي", "مش عارف أعمل ايه",
        # English: code + frustration
        "help with task", "help with assignment", "task is hard",
        "don't understand the task", "can't solve",
        "help me with this code", "explain this function",
        "i'm frustrated", "i'm stuck", "i'm lost", "i can't do this",
        "i give up", "this is too hard", "debugging help",
        "my code", "code keeps crashing", "code not working",
    ]
    if any(k in t for k in task_help_keywords):
        return "task_help"

    # ─── 6. MATERIALS REQUEST (videos, articles, courses, resources) ───
    materials_keywords = [
        # Arabic
        "فيديو", "فيديوهات", "فديو",
        "مقال", "مقالات", "مقالة",
        "كورس", "كورسات", "دورة", "دورات",
        "موارد", "مواد تعليمية", "مادة تعليمية",
        # English
        "video", "videos", "article", "articles",
        "course", "courses", "tutorial", "tutorials",
        "resource", "resources", "material", "materials",
    ]
    if any(k in t for k in materials_keywords):
        return "materials_request"

    # ─── 7. RECOMMENDATION EXPLANATION (why was this mentor recommended?) ───
    # Must be checked BEFORE generic explanation_request to avoid swallowing
    # phrases like "ايه اللي خلاه مناسب" or "explain the recommendation".
    recommendation_explanation_keywords = [
        # Arabic: follow-up "why" questions about recommendations
        "ليه رشحتلي", "ليه رشحت", "ليه اخترت",
        "ليه ده أحسن", "ليه ده احسن", "ليه ده الأفضل",
        "إيه اللي خلاه مناسب", "ايه اللي خلاه مناسب",
        "ليه المنتور ده", "ليه المرشد ده",
        "سبب الترشيح", "أسباب الترشيح",
        "ليه ده مش",  # "ليه ده مش التاني" — comparison
        # English
        "why did you recommend", "why this mentor",
        "why is this the best", "why was this recommended",
        "explain the recommendation", "why not the other",
        "what made this mentor", "why is he recommended",
        "why is she recommended", "why choose this",
    ]
    if any(k in t for k in recommendation_explanation_keywords):
        return "recommendation_explanation"

    # ─── 8. CONCEPT EXPLANATION → general_question (merged intent) ───
    explanation_keywords = [
        # Arabic — "ايه" with space catches "ايه machine learning", "ايه ده", etc.
        "ايه ", "ايه هو", "ايه هي", "ايه ال", "ايه ده",
        "يعني ايه", "معنى",
        "شرحلي", "شرح لي", "إشرح",
        "ايه الفرق", "الفرق بين",
        "ليه بنستخدم", "ليه ال",
        # English
        "what is", "what are", "what does",
        "explain", "definition of", "meaning of",
        "difference between", "why do we use",
        "how does", "what's the difference",
    ]
    if any(k in t for k in explanation_keywords):
        return "general_question"

    # ─── 9. ROADMAP REQUEST (learning paths, study plans) ───
    roadmap_keywords = [
        # Arabic
        "roadmap", "خريطة طريق", "خريطة", "خطة تعلم",
        "مسار تعلم", "مسار", "learning path", "study plan",
        "أبدأ أتعلم", "أبدأ اتعلم", "كيف أبدأ",
        "ازاي أبدأ", "إزاي أبدأ", "كيف أتعلم", "ازاي أتعلم",
        "المسار الأفضل", "خطة الدراسة", "ترتيب الدراسة",
        # English
        "learning path", "study path", "how to learn",
        "how to start learning", "roadmap for", "path to learn",
        "guide to learn", "steps to learn",
    ]
    if any(k in t for k in roadmap_keywords):
        return "roadmap_request"

    # ─── 10a. ASK PROGRAM RECOMMENDATION (program suggestions) ───
    program_rec_keywords = [
        # Arabic
        "رشحلي برامج", "رشحلي برنامج", "برامج مناسبة", "برنامج إرشاد",
        "برنامج ارشاد", "اقترح برامج", "برامج ليا",
        # English
        "recommend program", "recommend programs", "suggest program",
        "suggest programs", "program for me", "programs for me",
        "mentorship program", "find program",
    ]
    if any(k in t for k in program_rec_keywords):
        return "ask_program_recommendation"

    # ─── 10b. ASK MENTOR RECOMMENDATION (personalized suggestions) ───
    recommendation_keywords = [
        # Arabic
        "رشحلي", "رشح", "اقترح", "توصية", "مناسب",
        "توصيات", "اقتراحات", "ارشحلي",
        # English
        "recommend me", "recommend a mentor", "recommend mentors",
        "suggest mentors", "suggest a mentor", "best mentor for me",
    ]
    if any(k in t for k in recommendation_keywords):
        return "ask_mentor_recommendation"

    # ─── 10. FIND MENTOR (search / browse) ───
    # First check: if user is asking HOW TO BECOME a mentor → faq, not find_mentor
    become_mentor_keywords = [
        "become a mentor", "become mentor", "how to be a mentor",
        "be a mentor", "join as mentor", "register as mentor",
        "sign up as mentor", "want to mentor", "i want to be a mentor",
        "ابقى مرشد", "أبقى مرشد", "ابقى mentor", "أبقى mentor",
        "اكون مرشد", "أكون مرشد", "اكون mentor", "أكون mentor",
        "اسجل كمرشد", "أسجل كمرشد", "اسجل ك mentor",
        "عايز أبقى mentor", "عايز ابقى mentor",
        "عايز أبقى مرشد", "عايز ابقى مرشد",
        "ازاي ابقى مرشد", "إزاي أبقى مرشد",
        "كيف اصير مرشد", "كيف أصبح مرشد",
    ]
    if any(k in t for k in become_mentor_keywords):
        return "faq"

    find_mentor_keywords = [
        # Arabic
        "عايز مرشد", "ابغى مرشد", "ابغى حد", "عايز حد",
        "دور على", "ابحث عن", "ابحث",
        "احسن", "افضل", "مين أحسن", "مين افضل",
        "مين أحسن مرشد", "مين افضل مرشد",
        "كام مرشد", "عدد المرشدين", "عايز mentor", "ممكن mentor",
        # English
        "need a mentor", "i need a mentor", "find mentor", "search mentor",
        "best mentor", "top mentor", "mentor in", "mentor for", "looking for",
    ]
    if any(k in t for k in find_mentor_keywords):
        return "find_mentor"

    # ─── 11. OFF-TOPIC ───
    off_topic_keywords = [
        "الجو", "weather", "مين رئيس", "president", "tell me a joke", "joke",
        "سياسة", "politics", "رياضة", "sports", "football", "match",
        "أغنية", "song", "music", "movies", "film",
        "اعملي شعر", "شعر", "poem", "poetry", "قصيدة",
        "اخبار", "news", "ترفيه", "entertainment",
    ]
    if any(k in t for k in off_topic_keywords):
        return "off_topic"

    # ─── 12. FAQ (platform how-to, rules, registration, pricing) ───
    faq_keywords = [
        # Arabic: duration, registration, pricing
        "مدة البرنامج", "كم مدة", "كام ساعة",
        "تسجيل", "اسجل", "أسجل", "تسجيل الدخول",
        "سعر", "مجاني", "مجانا",
        "إلغاء", "الغي", "الغاء",
        "تقييم", "أعطي رأي", "اعطي رأي",
        "متابعة", "حفظ برنامج",
        "حالات الطلب", "حالة الطلب",
        "هل المنصة", "هل يمكن", "هل ممكن",
        "القواعد", "الشروط", "المنصة",
        "اتواصل", "الدعم", "التواصل",
        # English
        "register", "sign up", "login", "create account",
        "free", "price", "cost",
        "cancel", "cancellation",
        "feedback", "rate", "review process",
        "how it works", "platform rules",
        "application status", "duration", "how long",
        "contact support", "reach support", "support team",
    ]
    if any(k in t for k in faq_keywords):
        return "faq"

    # ─── 13. GENERAL QUESTION (final fallback) ───
    logger.info("Keyword fallback: no match — returning general_question")
    return "general_question"


# ────────────────────────────────────────────────────────────────────
# Synchronous fallback (for test scripts)
# ────────────────────────────────────────────────────────────────────


def detect_intent(text: str) -> str:
    """Synchronous intent detection — uses keyword fallback only."""
    return _keyword_fallback(text)


# ────────────────────────────────────────────────────────────────────
# Service class
# ────────────────────────────────────────────────────────────────────


class IntentService:
    def detect_language(self, text: str) -> str:
        return detect_language(text)

    def detect_intent(self, text: str) -> str:
        return detect_intent(text)

    async def detect_intent_async(self, text: str) -> str:
        return await detect_intent_async(text)


intent_service = IntentService()
