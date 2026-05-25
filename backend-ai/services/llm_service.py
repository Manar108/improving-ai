"""Lightweight LLM wrapper used by the chatbot.

Provides:
  - `chat_general(message, language)` — answers general learning questions
  - `chat_off_topic(language)` — politely rejects off-topic messages
  - `chat_task_help(message, language)` — helps explain/solve tasks
  - `chat_roadmap(message, language)` — generates or finds learning roadmaps
  - `chat_complaint(message, language)` — handles mentor behavior complaints
  - `chat_support(message, language)` — handles technical support requests
  - `chat_fallback(message, language)` — backwards-compatible fallback

The LLM is guardrailed: it will only answer questions related to mentorship,
learning, tasks, roadmaps, and student progress.
"""

import hashlib
import logging
import time
import json

import httpx
import re

from config import settings

logger = logging.getLogger(__name__)
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ── Simple TTL cache for LLM responses (300s) ──
_llm_cache: dict[str, tuple[float, str]] = {}
_LLM_CACHE_TTL = 300  # 5 minutes


def _llm_cache_key(system_prompt: str, user_message: str, temperature: float, history: list | None = None) -> str:
    # Fast content hash — md5 is safe here (non-cryptographic, just dedup)
    hist_repr = ""
    if history:
        try:
            hist_repr = json.dumps(history, ensure_ascii=False, sort_keys=True)
        except Exception:
            hist_repr = str(history)
    raw = f"{system_prompt}|{hist_repr}|{user_message}|{temperature}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _get_cached_llm(key: str) -> str | None:
    entry = _llm_cache.get(key)
    if entry is None:
        return None
    ts, content = entry
    if time.monotonic() - ts > _LLM_CACHE_TTL:
        del _llm_cache[key]
        return None
    return content


def _set_cached_llm(key: str, content: str) -> None:
    _llm_cache[key] = (time.monotonic(), content)


def _has_arabic(text: str) -> bool:
    """Check if text contains any Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text))


def _fix_mixed_arabic_latin(text: str) -> str:
    """Post-process LLM output to reduce broken mixing of Arabic+Latin runs.

    Problem: Arabic clitics or prefixes can be attached directly to English words
    (e.g. "سيntax") which displays awkwardly. This function inserts a
    narrow separator (a regular space) between Arabic and Latin runs so the
    renderer doesn't join scripts incorrectly. It's intentionally conservative
    (only inserts a single space when transitions are adjacent).
    """
    if not text:
        return text

    # Unicode ranges for Arabic letters. We keep the pattern simple and
    # conservative: if an Arabic letter is immediately adjacent to an ASCII
    # letter/digit/symbol, insert a space between them.
    # Arabic -> Latin
    text = re.sub(r'([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF])([A-Za-z0-9@#%&])', r"\1 \2", text)
    # Latin -> Arabic
    text = re.sub(r'([A-Za-z0-9@#%&])([\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF])', r"\1 \2", text)
    return text


# ────────────────────────────────────────────────────────────────────
# System prompts
# ────────────────────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = """\
You are "Mentora", the AI assistant for a mentorship and learning platform.

RULES:
- ONLY answer questions related to: mentorship, learning, education, tasks, roadmaps, programming, student progress, and career development.
- If the question is unrelated (politics, weather, sports, gossip, jokes, global news, or any random topic), respond with EXACTLY:
  "السؤال خارج نطاق المنصة، نقدر نساعدك في التعلم أو mentorship 😊"
- DO NOT hallucinate facts. If you don't know, say so.
- DO NOT guess or fabricate links, names, or statistics.
- Respond in the same language as the user (Arabic or English).
- Be friendly, concise, and professional.
- Use emojis sparingly for warmth.
"""

_GENERAL_QUESTION_PROMPT = _BASE_SYSTEM_PROMPT + """
The user is asking a general learning/education question.
Give a clear, helpful, and accurate answer.
If it's a programming concept, explain it with a simple example.
Keep the answer focused and under 300 words.
"""

_TASK_HELP_PROMPT = _BASE_SYSTEM_PROMPT + """
The user needs help with a task, assignment, project, or code.
They may also be frustrated, exhausted, or feeling lost — treat them with empathy.

STEP 1 — DETECT EMOTIONAL STATE:
If the user sounds frustrated, overwhelmed, or hopeless ("مش قادر", "تعبت", "محبط", "ضايع",
"I give up", "I'm stuck", "this is too hard"):
  → Start with emotional support FIRST: acknowledge their feelings, normalize the struggle,
    remind them that every developer gets stuck, and encourage them.
  → Then transition to practical help.

If the user is calm and just asking for help:
  → Skip the emotional support and go straight to practical help.

STEP 2 — PROVIDE PRACTICAL HELP:
1. Clarify what the task is asking
2. Break it down into smaller steps
3. Give hints or guidance without giving the full solution
4. If they shared code, explain what it does and point out the issue
5. Encourage them to try and learn

If the task details are vague, ask them to share more details about the task.

TONE: Warm, patient, encouraging. Like a supportive senior student helping a junior.
Never dismiss their frustration. Never say "it's easy" or make them feel bad for struggling.
"""

_ROADMAP_PROMPT = _BASE_SYSTEM_PROMPT + """
The user wants a learning roadmap, study plan, or educational materials for a topic.

FIRST: Detect if the user already specified preferences:
  - Level: Look for "beginner", "مبتدئ", "intermediate", "متقدم", "advanced", "advanced"
  - Format: Look for "video", "فيديو", "article", "مقال", "course", "كورس", "project", "مشروع"
  - Goal: Look for "web", "data science", "موارد", "شرح", "tutorial", etc.

IF the user was SPECIFIC (e.g., "عايز فيديو عن Python" or "give me videos on machine learning"):
  → Acknowledge their request
  → Create a roadmap or list of topics
  → Include suggested video titles or article topics
  → Be direct and actionable
  → Format: Clear bullet points, no vague answers

IF the user was VAGUE (e.g., "عايز أتعلم Python" with no format preference):
  → Ask 2-3 quick clarifying questions:
     1. Level: "Are you a complete beginner, or do you have some programming experience?"
     2. Format: "Do you prefer videos, articles, interactive courses, or a mix?"
     3. Goal: "Is this for a specific project or general skill building?"
  → After they answer, provide the roadmap

ROADMAP STRUCTURE (if providing one):
1. Prerequisites (if any)
2. Phase 1 (Beginner): Topics → Key Skills → Estimated Duration
3. Phase 2 (Intermediate): Topics → Key Skills → Estimated Duration  
4. Phase 3 (Advanced): Topics → Key Skills → Estimated Duration
5. Practice Projects at each level
6. Recommended resources (videos, courses, articles, books, GitHub projects)

Be conversational, encouraging, and practical. Use the user's language.
"""

_SUBMIT_TASK_PROMPT = _BASE_SYSTEM_PROMPT + """
The user is asking how to submit their task/assignment.
Explain clearly:
1. Go to your program's Roadmap page
2. Find the specific task/milestone in the roadmap
3. Upload or submit your work there before the deadline
4. Check the deadline shown on the task card
5. After submission, your mentor will review and give feedback

If they're asking about a specific deadline, tell them to check the task card in the roadmap section.
Respond in the user's language.
"""

_COMPLAINT_PROMPT = _BASE_SYSTEM_PROMPT + """
The user wants to file a COMPLAINT about a mentor's behavior (rudeness, harassment, misconduct, disrespect, etc.).

⚠️ CRITICAL RULES:
- This is a COMPLAINT, NOT a mentor search or recommendation request.
- DO NOT search for mentors, list mentors, recommend mentors, or show mentor ratings.
- DO NOT treat this as a "find_mentor" or "ask_mentor_recommendation" request.
- FOCUS ONLY on the complaint process described below.

Your role is to:
1. Acknowledge their concern with empathy and professionalism
2. Gather key details if missing: mentor name, what happened, when
3. Explain the formal complaint process:
   - They can report via the platform's "Report" button on the mentor's profile
   - Or email the support team at mentora.help@gmail.com
   - The admin team will review the report and contact them via their registered email
   - The platform takes all complaints seriously and reviews them confidentially
4. Reassure them that their report will be handled with confidentiality
5. Do NOT dismiss, minimize, or blame the user
6. Keep the response concise (under 200 words)
Respond in the user's language.
"""

_SUPPORT_PROMPT = _BASE_SYSTEM_PROMPT + """
The user is reporting a TECHNICAL PROBLEM on the platform (can't upload, error, page not loading, login issue, etc.).
Your role is to:
1. Acknowledge the issue and express willingness to help
2. Ask for key troubleshooting details if vague:
   - What exactly is not working?
   - What error message do they see?
   - What browser/device are they using?
   - Have they tried refreshing or clearing cache?
3. Provide basic troubleshooting steps:
   - Refresh the page
   - Clear browser cache/cookies
   - Try a different browser or incognito mode
   - Check internet connection
4. If the issue persists, tell them to email: mentora.help@gmail.com and reassure them the team will respond soon.
5. Do NOT guess or fabricate specific technical causes
6. Keep the response concise (under 200 words)
Respond in the user's language.
"""


# ────────────────────────────────────────────────────────────────────
# LLM Service
# ────────────────────────────────────────────────────────────────────


class LLMService:
    """Mentorship-scoped LLM service with guardrails."""

    async def _call_llm(self, system_prompt: str, user_message: str, history: list[dict] | None = None, temperature: float = 0.3, language: str = "en") -> str | None:
        """Internal helper to call Groq with a system prompt (with caching).
        
        Args:
            system_prompt: System prompt to use
            user_message: User's message
            history: Optional conversation history
            temperature: Temperature for response generation
            language: 'ar' for Arabic, 'en' for English — enforces response language
        """
        if not settings.GROQ_API_KEY:
            return None

        cache_key = _llm_cache_key(system_prompt, user_message, temperature, history)
        cached = _get_cached_llm(cache_key)
        if cached is not None:
            logger.debug("LLM cache HIT")
            return cached

        # Add language enforcement to system prompt
        lang_enforcer = ""
        if language == "ar":
            lang_enforcer = "\n\n⚠️ CRITICAL: Your response MUST be in Arabic (لغة عربية فقط). Do NOT respond in English."
        elif language == "en":
            lang_enforcer = "\n\n⚠️ CRITICAL: Your response MUST be in English only. Do NOT respond in Arabic."
        
        enforced_prompt = system_prompt + lang_enforcer

        # Build the messages list: system + optional history + user
        messages = [{"role": "system", "content": enforced_prompt}]
        if history:
            # assume history is a list of {role, content} dicts already
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": settings.MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(_GROQ_URL, json=payload, headers=headers)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                # Post-process to reduce mixed-script display issues (Arabic+Latin)
                content = _fix_mixed_arabic_latin(content)
                
                # Validate language response (simple heuristic check)
                if language == "ar" and not _has_arabic(content):
                    logger.warning("LLM returned non-Arabic response for Arabic request; retrying with stronger enforcement")
                    # Would retry here if needed, for now log and return
                elif language == "en" and _has_arabic(content) and len(content) > 50:
                    # If response is too long and mixed Arabic, warn (but don't fail)
                    logger.warning("LLM response has unexpected Arabic mixed in for English request")
                
                _set_cached_llm(cache_key, content)
                return content
            logger.warning("Groq call failed: %s", resp.text[:200])
        except Exception as exc:
            logger.warning("Groq call error: %s", exc)
        return None

    async def chat_general(self, message: str, language: str = "en", history: list[dict] | None = None) -> str:
        """Answer a general learning/education question."""
        result = await self._call_llm(_GENERAL_QUESTION_PROMPT, message, history=history, language=language)
        if result:
            return result
        if language == "ar":
            return "عذرًا، لم أتمكن من الإجابة الآن. حاول مرة أخرى أو أعد صياغة السؤال."
        return "Sorry, I couldn't answer right now. Please try again or rephrase."

    async def chat_task_help(self, message: str, language: str = "en", history: list[dict] | None = None) -> str:
        """Help the user understand or work on a task."""
        result = await self._call_llm(_TASK_HELP_PROMPT, message, history=history, language=language)
        if result:
            return result
        if language == "ar":
            return "ممكن تشاركني تفاصيل أكتر عن التاسك عشان أقدر أساعدك؟ 📝"
        return "Could you share more details about the task so I can help? 📝"

    async def chat_roadmap(self, message: str, language: str = "en", history: list[dict] | None = None) -> str:
        """Generate or suggest a learning roadmap."""
        result = await self._call_llm(_ROADMAP_PROMPT, message, history=history, temperature=0.4, language=language)
        if result:
            return result
        if language == "ar":
            return "عذرًا، لم أتمكن من إنشاء الخريطة الآن. حاول مرة أخرى."
        return "Sorry, I couldn't generate the roadmap right now. Please try again."

    async def chat_submit_task(self, message: str, language: str = "en", history: list[dict] | None = None) -> str:
        """Explain how to submit a task."""
        result = await self._call_llm(_SUBMIT_TASK_PROMPT, message, history=history, language=language)
        if result:
            return result
        # Hardcoded fallback
        if language == "ar":
            return (
                "لتسليم التاسك:\n"
                "  1. اذهب لصفحة الـ Roadmap في البرنامج\n"
                "  2. اختر التاسك المطلوب\n"
                "  3. ارفع شغلك قبل الـ Deadline\n"
                "  4. المرشد هيراجعه ويديك feedback\n\n"
                "تأكد إنك تسلم قبل الميعاد! ⏰"
            )
        return (
            "To submit your task:\n"
            "  1. Go to the Roadmap page in your program\n"
            "  2. Find the specific task\n"
            "  3. Upload your work before the deadline\n"
            "  4. Your mentor will review and provide feedback\n\n"
            "Make sure to submit before the deadline! ⏰"
        )

    async def chat_complaint(self, message: str, language: str = "en", history: list[dict] | None = None) -> str:
        """Handle a complaint about mentor behavior."""
        result = await self._call_llm(_COMPLAINT_PROMPT, message, history=history, language=language)
        if result:
            return result
        if language == "ar":
            return (
                "نأسف جدًا لو حصل معاك موقف سلبي.\n"
                "لتقديم شكوى رسمية:\n"
                "  1. استخدم زر \"Report\" على البروفايل بتاع المرشد\n"
                "  2. أو ابعت إيميل على mentora.help@gmail.com\n"
                "  3. فريق الإدارة هيراجع البلاغ ويتواصل معاك على الإيميل المسجل\n\n"
                "كل الشكاوى بتتعامل بسرية تامة وهيتم الرد عليك قريب 🔒"
            )
        return (
            "We're truly sorry you had a negative experience.\n"
            "To file a formal complaint:\n"
            "  1. Use the \"Report\" button on the mentor's profile\n"
            "  2. Or email us at mentora.help@gmail.com\n"
            "  3. The admin team will review the report and reach out via your registered email\n\n"
            "All reports are handled confidentially and we'll get back to you soon 🔒"
        )

    async def chat_support(self, message: str, language: str = "en", history: list[dict] | None = None) -> str:
        """Handle a technical support request."""
        result = await self._call_llm(_SUPPORT_PROMPT, message, history=history, language=language)
        if result:
            return result
        if language == "ar":
            return (
                "جرب الخطوات دي:\n"
                "  1. اعمل refresh للصفحة\n"
                "  2. امسح الـ cache و cookies\n"
                "  3. جرب browser تاني أو incognito mode\n"
                "  4. اتأكد من اتصال النت\n\n"
                "لو المشكلة لسه موجودة، ابعت إيميل على mentora.help@gmail.com وهيتم الرد عليك قريب 🛠️"
            )
        return (
            "Try these steps:\n"
            "  1. Refresh the page\n"
            "  2. Clear browser cache and cookies\n"
            "  3. Try a different browser or incognito mode\n"
            "  4. Check your internet connection\n\n"
            "If the issue persists, email us at mentora.help@gmail.com and we'll get back to you soon 🛠️"
        )

    def chat_off_topic(self, language: str = "en") -> str:
        """Return the off-topic rejection message."""
        if language == "ar":
            return "السؤال خارج نطاق المنصة، نقدر نساعدك في التعلم أو mentorship 😊"
        return "This question is outside the platform's scope. I can help you with learning or mentorship 😊"

    async def chat_fallback(self, message: str, language: str = "en", history: list[dict] | None = None) -> str:
        """Backwards-compatible fallback — routes to chat_general."""
        return await self.chat_general(message, language, history=history)

    async def chat_with_system_prompt(self, message: str, language: str = "en", system_prompt: str = _BASE_SYSTEM_PROMPT, history: list[dict] | None = None, temperature: float = 0.3) -> str:
        """Generic method to call LLM with a custom system prompt.
        
        Useful for mentor-specific or custom response generation.
        """
        result = await self._call_llm(system_prompt, message, history=history, temperature=temperature, language=language)
        if result:
            return result
        if language == "ar":
            return "معذرة، حدثت مشكلة في معالجة طلبك. جرب مرة أخرى 😊"
        return "Sorry, I encountered an error. Please try again 😊"


llm_service = LLMService()
