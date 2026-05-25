import json
import logging
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import text

from config import settings
from database.db import DatabaseAccessError, MissingTableError, database


logger = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ------------------------------------------------------------------
# In-file safe SQL query helpers (parameterized, SQLAlchemy)
# ------------------------------------------------------------------

def get_top_mentors_by_rating(domain_id: Optional[str] = None, limit: int = 5):
    if domain_id:
        sql = """
        SELECT
            CONCAT(u.first_name, ' ', u.last_name) AS name,
            COALESCE(d.name, 'General') AS domain,
            COALESCE(CAST(mp.average_rating AS float), 0) AS rating,
            COALESCE(mp.total_reviews, 0) AS reviews
        FROM mentor_profile mp
        INNER JOIN users u ON u.user_id = mp.user_id
        LEFT JOIN domains d ON d.domain_id = mp.domain_id
        WHERE mp.domain_id = :did
        ORDER BY mp.average_rating DESC, mp.total_reviews DESC
        OFFSET 0 ROWS FETCH NEXT :limit ROWS ONLY
        """
        params = {"did": domain_id, "limit": limit}
    else:
        sql = """
        SELECT
            CONCAT(u.first_name, ' ', u.last_name) AS name,
            COALESCE(d.name, 'General') AS domain,
            COALESCE(CAST(mp.average_rating AS float), 0) AS rating,
            COALESCE(mp.total_reviews, 0) AS reviews
        FROM mentor_profile mp
        INNER JOIN users u ON u.user_id = mp.user_id
        LEFT JOIN domains d ON d.domain_id = mp.domain_id
        WHERE mp.average_rating IS NOT NULL AND mp.total_reviews > 0
        ORDER BY mp.average_rating DESC, mp.total_reviews DESC
        OFFSET 0 ROWS FETCH NEXT :limit ROWS ONLY
        """
        params = {"limit": limit}

    return database.run_query_df(sql, params)


def get_top_mentors_by_feedback(domain_id: Optional[str] = None, limit: int = 5):
    if domain_id:
        sql = """
        SELECT
            CONCAT(u.first_name, ' ', u.last_name) AS name,
            COALESCE(d.name, 'General') AS domain,
            COUNT(f.FeedbackId) AS feedback_count,
            COALESCE(AVG(CAST(f.Rating AS float)), 0) AS avg_rating
        FROM mentor_profile mp
        INNER JOIN users u ON u.user_id = mp.user_id
        LEFT JOIN domains d ON d.domain_id = mp.domain_id
        LEFT JOIN feedbacks f ON f.MentorProfileId = mp.user_id
        WHERE mp.domain_id = :did
        GROUP BY u.first_name, u.last_name, d.name
        ORDER BY feedback_count DESC, avg_rating DESC
        OFFSET 0 ROWS FETCH NEXT :limit ROWS ONLY
        """
        params = {"did": domain_id, "limit": limit}
    else:
        sql = """
        SELECT
            CONCAT(u.first_name, ' ', u.last_name) AS name,
            COALESCE(d.name, 'General') AS domain,
            COUNT(f.FeedbackId) AS feedback_count,
            COALESCE(AVG(CAST(f.Rating AS float)), 0) AS avg_rating
        FROM mentor_profile mp
        INNER JOIN users u ON u.user_id = mp.user_id
        LEFT JOIN domains d ON d.domain_id = mp.domain_id
        LEFT JOIN feedbacks f ON f.MentorProfileId = mp.user_id
        GROUP BY u.first_name, u.last_name, d.name
        ORDER BY feedback_count DESC, avg_rating DESC
        OFFSET 0 ROWS FETCH NEXT :limit ROWS ONLY
        """
        params = {"limit": limit}
    return database.run_query_df(sql, params)


def get_mentors_by_domain(domain_name: str):
    sql = """
    SELECT
        mp.user_id AS mentor_id,
        CONCAT(u.first_name, ' ', u.last_name) AS name,
        COALESCE(d.name, 'General') AS domain,
        COALESCE(CAST(mp.average_rating AS float), 0) AS rating,
        COALESCE(mp.total_reviews, 0) AS reviews
    FROM mentor_profile mp
    INNER JOIN users u ON u.user_id = mp.user_id
    LEFT JOIN domains d ON d.domain_id = mp.domain_id
    WHERE LOWER(d.name) LIKE LOWER(:dname)
    ORDER BY mp.average_rating DESC, mp.total_reviews DESC
    """
    return database.run_query_df(sql, {"dname": f"%{domain_name}%"})


def get_open_programs(domain_id: Optional[str] = None, limit: int = 10):
    """Return published programs that are currently accepting applications.

    Requires:
      - ProgramPostStatus = 'Published' (publication state)
      - Availability NOT IN ('Closed', 'Archived') (mentor hasn't closed the program)

    Programs in draft or closed by mentor are never returned.
    """
    base_sql = """
        SELECT p.ProgramId, p.Title, COALESCE(d.name, 'General') AS domain,
               CONCAT(u.first_name, ' ', u.last_name) AS mentor, p.CreatedAt
        FROM programs p
        INNER JOIN mentor_profile mp ON mp.user_id = p.MentorProfileId
        INNER JOIN users u ON u.user_id = mp.user_id
        LEFT JOIN domains d ON d.domain_id = p.DomainId
        WHERE p.ProgramPostStatus = 'Published'
          AND p.Availability NOT IN ('Closed', 'Archived', 'Cancelled')
    """
    if domain_id:
        sql = base_sql + " AND p.DomainId = :did\nORDER BY p.CreatedAt DESC\nOFFSET 0 ROWS FETCH NEXT :limit ROWS ONLY"
        params = {"did": domain_id, "limit": limit}
    else:
        sql = base_sql + "\nORDER BY p.CreatedAt DESC\nOFFSET 0 ROWS FETCH NEXT :limit ROWS ONLY"
        params = {"limit": limit}
    return database.run_query_df(sql, params)


def get_feedback_summary(mentor_profile_id: str):
    total = database.run_scalar(
        "SELECT COUNT(1) FROM feedbacks WHERE MentorProfileId = :mid",
        {"mid": mentor_profile_id},
    )
    avg = database.run_scalar_float(
        "SELECT ISNULL(AVG(CAST(Rating AS float)), 0) FROM feedbacks WHERE MentorProfileId = :mid",
        {"mid": mentor_profile_id},
    )
    recent = database.run_query_df(
        """
        SELECT TOP 10 f.FeedbackId, f.Rating, f.Comment, f.CreatedAt,
               CONCAT(u.first_name, ' ', u.last_name) AS from_user
        FROM feedbacks f
        LEFT JOIN users u ON u.user_id = f.MenteeProfileId
        WHERE f.MentorProfileId = :mid
        ORDER BY f.CreatedAt DESC
        """,
        {"mid": mentor_profile_id},
    )
    return {"total_feedbacks": total, "avg_rating": avg, "recent_feedbacks": recent}


def get_mentor_counts_by_domain():
    sql = """
    SELECT COALESCE(d.name, 'General') AS domain, COUNT(mp.user_id) AS mentor_count
    FROM mentor_profile mp
    LEFT JOIN domains d ON d.domain_id = mp.domain_id
    GROUP BY d.name
    ORDER BY mentor_count DESC
    """
    return database.run_query_df(sql)


def get_programs_by_mentor(mentor_profile_id: str):
    sql = """
    SELECT p.ProgramId, p.Title, COALESCE(d.name, 'General') AS domain, p.CreatedAt
    FROM programs p
    LEFT JOIN domains d ON d.domain_id = p.DomainId
    WHERE p.MentorProfileId = :mid
    ORDER BY p.CreatedAt DESC
    """
    return database.run_query_df(sql, {"mid": mentor_profile_id})


def get_mentors_with_min_reviews(min_reviews: int = 5, limit: int = 50):
    sql = """
    SELECT
        CONCAT(u.first_name, ' ', u.last_name) AS name,
        COALESCE(d.name, 'General') AS domain,
        COALESCE(CAST(mp.average_rating AS float), 0) AS rating,
        COALESCE(mp.total_reviews, 0) AS reviews
    FROM mentor_profile mp
    INNER JOIN users u ON u.user_id = mp.user_id
    LEFT JOIN domains d ON d.domain_id = mp.domain_id
    WHERE COALESCE(mp.total_reviews, 0) >= :min_reviews
    ORDER BY mp.average_rating DESC, mp.total_reviews DESC
    OFFSET 0 ROWS FETCH NEXT :limit ROWS ONLY
    """
    return database.run_query_df(sql, {"min_reviews": min_reviews, "limit": limit})


class RagService:
    """Database-driven RAG service for the AI chatbot.

    Answers platform questions using live SQL Server data, FAQ knowledge,
    and step-by-step guidance for common user workflows.
    """

    def __init__(self) -> None:
        self._faq: list[dict] = self._load_faq()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def answer_platform_question(
        self, message: str, language: str, user_context: dict | None = None,
        intent: str = "",
    ) -> str:
        """Route platform questions to the best handler.

        Uses a fast 8b LLM to semantically understand the sub-intent,
        then dispatches to the correct database handler.
        """
        query = message.lower().strip()
        ctx = user_context or {}

        # --- FAQ match first (highest priority, instant) ---
        # Skip FAQ for find_mentor intent — those should go to DB sub-router
        if intent not in ("find_mentor",):
            faq_answer = self._match_faq(query, language)
            if faq_answer:
                return faq_answer

        # --- Personal queries (require user_context) ---
        try:
            if ctx.get("user_id"):
                personal = self._try_personal_answer(query, language, ctx)
                if personal:
                    return personal

            # --- LLM-powered smart sub-routing (fast 8b model) ---
            sub_intent = await self._smart_sub_route(query)
            return self._dispatch_sub_intent(sub_intent, query, language, ctx)

        except MissingTableError as exc:
            logger.error("RAG platform query failed: missing table: %s", exc)
            return self._unavailable(language)
        except DatabaseAccessError as exc:
            logger.error("RAG platform query failed: database error: %s", exc)
            return self._unavailable(language)

    # ------------------------------------------------------------------
    # Smart LLM sub-router (8b-instant, ~0.3s)
    # ------------------------------------------------------------------

    _SUB_ROUTE_MODEL = "llama-3.1-8b-instant"

    _SUB_ROUTE_PROMPT = """\
Classify this mentorship platform question into ONE sub-intent. Return ONLY JSON: {"sub":"<label>","domain":"<if mentioned>"}

Sub-intents:
- top_mentors: asking about best/top/highest-rated mentors or mentors in a field ("عايز mentor في AI", "best mentor in web", "مين أحسن مرشد", "find me a mentor")
- mentor_feedback: asking about mentors with best feedback/reviews/ratings ("احسن مينتور فيدباك", "mentors with highest ratings", "best reviewed mentor")
- mentor_stats: asking about mentor counts/statistics ("كام مرشد", "how many mentors")
- domains: asking what fields/domains are available on the platform ("ايه المجالات المتاحة")
- subdomains: asking about subdomains/specializations within a domain ("ايه التخصصات في AI", "what subdomains")
- technologies: asking ONLY about technologies/tech stack list ("ايه التقنيات", "what technologies")
- programs: asking about mentorship programs or program details
- applications: asking about application status/stats
- mentorships: asking about active mentorships
- cancellations: asking about cancellations/cancelled mentorships or applications
- feedback: general feedback/rating stats (NOT about finding mentors)
- follows: follower stats
- saved_posts: asking about saved/bookmarked or shared programs
- countries: available countries
- registration: how to register/sign up
- apply: how to apply for a program
- become_mentor: how to become a mentor
- mentorship_process: how mentorship works
- verification: mentor verification process
- unknown: doesn't match any above

KEY: if the user mentions "mentor" + any field/domain → top_mentors (NOT technologies). If they mention "feedback/rating" + "mentor" → mentor_feedback.
If a domain is mentioned (AI, web, Python, design, etc), include "domain":"<name>".
Return ONLY the JSON."""

    async def _smart_sub_route(self, query: str) -> dict:
        """Use fast 8b model to classify the sub-intent. Returns dict with 'sub' and optional 'domain'."""
        if not settings.GROQ_API_KEY:
            return {"sub": "unknown"}

        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._SUB_ROUTE_MODEL,
            "messages": [
                {"role": "system", "content": self._SUB_ROUTE_PROMPT},
                {"role": "user", "content": query},
            ],
            "temperature": 0.0,
            "max_tokens": 60,
        }

        try:
            async with httpx.AsyncClient(timeout=6) as client:
                resp = await client.post(_GROQ_URL, headers=headers, json=payload)

            if resp.status_code != 200:
                logger.warning("RAG sub-route LLM %d", resp.status_code)
                return {"sub": "unknown"}

            content = resp.json()["choices"][0]["message"]["content"].strip()
            parsed = self._parse_json(content)
            if parsed and "sub" in parsed:
                logger.info("RAG sub-route: '%s' → %s", query[:50], parsed.get("sub"))
                return parsed
        except Exception as exc:
            logger.warning("RAG sub-route LLM failed: %s", exc)

        return {"sub": "unknown"}

    def _dispatch_sub_intent(self, result: dict, query: str, language: str, ctx: dict) -> str:
        """Dispatch to the correct handler based on LLM sub-intent."""
        sub = result.get("sub", "unknown")
        domain_name = result.get("domain")

        if sub == "domains":
            return self._answer_domains(language)

        if sub == "technologies":
            return self._answer_technologies(language)

        if sub == "top_mentors":
            if domain_name:
                top_df = get_mentors_by_domain(domain_name)
                if not top_df.empty:
                    header = f"المرشدون في {domain_name}:" if language == "ar" else f"Mentors in {domain_name}:"
                    lines = [header]
                    for _, row in top_df.head(5).iterrows():
                        lines.append(f"  • {row['name']} — {row['domain']} (⭐ {row['rating']:.1f}, {row['reviews']} reviews)")
                    return "\n".join(lines)
            domain_id = ctx.get("domain_id") if self._is_personal(query) else None
            return self._answer_top_mentors(language, domain_id=domain_id, domain_name=domain_name or ctx.get("domain_name"))

        if sub == "mentor_stats":
            if domain_name:
                top_df = get_mentors_by_domain(domain_name)
                if not top_df.empty:
                    header = f"المرشدون في {domain_name}:" if language == "ar" else f"Mentors in {domain_name}:"
                    lines = [header]
                    for _, row in top_df.head(5).iterrows():
                        lines.append(f"  • {row['name']} — {row['domain']} (⭐ {row['rating']:.1f}, {row['reviews']} reviews)")
                    return "\n".join(lines)
            return self._answer_mentor_stats(language)

        if sub == "mentor_feedback":
            domain_id = None
            if domain_name:
                try:
                    did_df = database.run_query_df(
                        "SELECT domain_id FROM domains WHERE LOWER(name) LIKE LOWER(:n)",
                        {"n": f"%{domain_name}%"},
                    )
                    if not did_df.empty:
                        domain_id = str(did_df.iloc[0]["domain_id"])
                except Exception:
                    pass
            top_df = get_top_mentors_by_feedback(domain_id, limit=5)
            if not top_df.empty:
                header = "أفضل المرشدين حسب التقييمات" if language == "ar" else "Top mentors by feedback"
                if domain_name:
                    header += f" ({domain_name})"
                header += ":"
                lines = [header]
                for _, row in top_df.head(5).iterrows():
                    lines.append(f"  • {row['name']} — {row['domain']} ({row['feedback_count']} feedbacks, ⭐ {row['avg_rating']:.1f})")
                return "\n".join(lines)
            return self._answer_feedback(language)

        if sub == "programs":
            return self._answer_programs(language)
        if sub == "applications":
            return self._answer_applications(language)
        if sub == "mentorships":
            return self._answer_mentorships(language)
        if sub == "feedback":
            return self._answer_feedback(language)
        if sub == "follows":
            return self._answer_follows(language)
        if sub == "countries":
            return self._answer_countries(language)
        if sub == "subdomains":
            return self._answer_subdomains(language, domain_name)
        if sub == "cancellations":
            return self._answer_cancellations(language)
        if sub == "saved_posts":
            return self._answer_saved_shared(language)

        if sub == "registration":
            return self._guide_registration(language)
        if sub == "apply":
            return self._guide_apply(language)
        if sub == "become_mentor":
            return self._guide_become_mentor(language)
        if sub == "mentorship_process":
            return self._guide_mentorship_process(language)
        if sub == "verification":
            return self._guide_verification(language)

        # Personal fallback
        if ctx.get("user_id"):
            personal = self._try_personal_answer(query, language, ctx)
            if personal:
                return personal

        return self._unavailable(language)

    # ------------------------------------------------------------------
    # FAQ matching
    # ------------------------------------------------------------------

    def _load_faq(self) -> list[dict]:
        faq_path = settings.FAQ_PATH
        if not faq_path.exists():
            logger.warning("FAQ file not found at %s", faq_path)
            return []
        try:
            return json.loads(faq_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load FAQ: %s", exc)
            return []

    def _match_faq(self, query: str, language: str) -> str | None:
        key_field = f"keywords_{language}" if language in ("ar", "en") else "keywords_en"
        answer_field = f"answer_{language}" if language in ("ar", "en") else "answer_en"
        best_match = None
        best_score = 0
        for entry in self._faq:
            keywords = entry.get(key_field, [])
            score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                # Skip very short keywords (< 4 chars) to avoid false positives
                if len(kw_lower) < 4:
                    continue
                if kw_lower in query:
                    # Multi-word keywords get higher score
                    score += len(kw_lower)
            if score > best_score:
                best_score = score
                best_match = entry.get(answer_field, "")
        # Require minimum match quality
        if best_score >= 4:
            return best_match
        return None

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Database-driven answers
    # ------------------------------------------------------------------

    def _answer_domains(self, language: str) -> str:
        count = database.run_scalar("SELECT COUNT(1) FROM domains")
        names_df = database.run_query_df("SELECT name FROM domains ORDER BY name")
        names = ", ".join(names_df["name"].tolist()) if not names_df.empty else "N/A"
        if language == "ar":
            return f"يوجد حاليًا {count} مجال في المنصة:\n{names}"
        return f"There are currently {count} domains on the platform:\n{names}"

    def _answer_technologies(self, language: str) -> str:
        count = database.run_scalar("SELECT COUNT(1) FROM technologies")
        top_df = database.run_query_df("""
            SELECT TOP 15 t.name AS tech, s.name AS subdomain
            FROM technologies t
            LEFT JOIN subdomain s ON s.subdomain_id = t.subdomain_id
            ORDER BY t.name
        """)
        if top_df.empty:
            listing = "N/A"
        else:
            listing = "\n".join(
                f"  • {row['tech']} ({row['subdomain']})" for _, row in top_df.iterrows()
            )
        if language == "ar":
            return f"يوجد {count} تقنية في المنصة. أبرزها:\n{listing}"
        return f"There are {count} technologies on the platform. Here are some:\n{listing}"

    def _answer_top_mentors(self, language: str, *, domain_id=None, domain_name=None) -> str:
        """Top mentors — optionally filtered by the user's domain."""
        if domain_id is not None:
            top_df = get_top_mentors_by_rating(domain_id, limit=5)
            label_ar = f"أفضل المرشدين في مجالك ({domain_name or 'General'}):"
            label_en = f"Top mentors in your domain ({domain_name or 'General'}):"
        else:
            top_df = get_top_mentors_by_rating(limit=5)
            label_ar = "أفضل المرشدين تقييمًا:"
            label_en = "Top rated mentors:"

        if top_df.empty:
            return self._no_data(language)
        lines = []
        for i, row in top_df.iterrows():
            lines.append(f"  {len(lines)+1}. {row['name']} — {row['domain']} (⭐ {row['rating']:.1f}, {row['reviews']} reviews)")
        listing = "\n".join(lines)
        if language == "ar":
            return f"{label_ar}\n{listing}"
        return f"{label_en}\n{listing}"

    def _answer_mentor_stats(self, language: str) -> str:
        total = database.run_scalar("SELECT COUNT(1) FROM mentor_profile")
        verified = database.run_scalar("SELECT COUNT(1) FROM mentor_profile WHERE is_verified = 1")
        avg_rating = database.run_scalar_float(
            "SELECT ISNULL(AVG(CAST(average_rating AS float)), 0) FROM mentor_profile WHERE average_rating > 0"
        )
        if language == "ar":
            return (
                f"إحصائيات المرشدين:\n"
                f"  • إجمالي المرشدين: {total}\n"
                f"  • المرشدون الموثقون: {verified}\n"
                f"  • متوسط التقييم: {avg_rating}"
            )
        return (
            f"Mentor statistics:\n"
            f"  • Total mentors: {total}\n"
            f"  • Verified mentors: {verified}\n"
            f"  • Average rating: {avg_rating}"
        )

    def _answer_programs(self, language: str) -> str:
        total = database.run_scalar("SELECT COUNT(1) FROM programs")
        top_df = get_open_programs(limit=5)
        if top_df.empty:
            listing = "  No programs available yet."
        else:
            listing = "\n".join(
                f"  • {row['Title']} — {row['domain']} (by {row['mentor']})"
                for _, row in top_df.iterrows()
            )
        if language == "ar":
            return f"يوجد {total} برنامج إرشادي. أحدثها:\n{listing}"
        return f"There are {total} mentorship programs. Latest:\n{listing}"

    def _answer_applications(self, language: str) -> str:
        if language == "ar":
            return (
                "يمكنك متابعة حالات طلباتك من صفحة 'طلباتي'.\n"
                "الحالات الممكنة: مقبول ✅، معلق 🟡، أو مرفوض ❌."
            )
        return (
            "You can track your application statuses from the 'My Applications' page.\n"
            "Possible statuses: Accepted ✅, Pending 🟡, or Rejected ❌."
        )

    def _answer_mentorships(self, language: str) -> str:
        if language == "ar":
            return (
                "يمكنك متابعة إرشاداتك من صفحة 'إرشاداتي'.\n"
                "الحالات: نشط 🟢، مكتمل ✅، أو ملغي ❌."
            )
        return (
            "You can track your mentorships from the 'My Mentorships' page.\n"
            "Statuses: Active 🟢, Completed ✅, or Cancelled ❌."
        )

    def _answer_feedback(self, language: str) -> str:
        if language == "ar":
            return (
                "يمكنك تقييم المرشد بعد إتمام الإرشاد.\n"
                "التقييمات بتساعد المتدربين الآخرين في اختيار المرشد المناسب ⭐"
            )
        return (
            "You can rate your mentor after completing a mentorship.\n"
            "Ratings help other mentees find the right mentor ⭐"
        )

    def _answer_follows(self, language: str) -> str:
        if language == "ar":
            return (
                "يمكنك متابعة مرشدين للحصول على تحديثاتهم وبرامجهم الجديدة.\n"
                "اضغط زر 'متابعة' على صفحة أي مرشد 🔖"
            )
        return (
            "You can follow mentors to get updates on their new programs.\n"
            "Click the 'Follow' button on any mentor's profile 🔖"
        )

    def _answer_countries(self, language: str) -> str:
        df = database.run_query_df("SELECT country_code, country_name FROM countries ORDER BY country_name")
        if df.empty:
            return self._no_data(language)
        listing = ", ".join(df["country_name"].tolist())
        if language == "ar":
            return f"الدول المتاحة ({len(df)}):\n{listing}"
        return f"Available countries ({len(df)}):\n{listing}"

    # ------------------------------------------------------------------
    # New extended handlers
    # ------------------------------------------------------------------

    def _answer_subdomains(self, language: str, domain_name: str | None = None) -> str:
        if domain_name:
            df = database.run_query_df(
                """
                SELECT s.name AS subdomain, d.name AS domain
                FROM subdomain s
                INNER JOIN domains d ON d.domain_id = s.domain_id
                WHERE LOWER(d.name) LIKE LOWER(:dn)
                ORDER BY s.name
                """,
                {"dn": f"%{domain_name}%"},
            )
            if not df.empty:
                listing = "\n".join(f"  • {row['subdomain']}" for _, row in df.iterrows())
                if language == "ar":
                    return f"التخصصات الفرعية في {domain_name} ({len(df)}):\n{listing}"
                return f"Subdomains in {domain_name} ({len(df)}):\n{listing}"

        df = database.run_query_df(
            """
            SELECT s.name AS subdomain, d.name AS domain
            FROM subdomain s
            INNER JOIN domains d ON d.domain_id = s.domain_id
            ORDER BY d.name, s.name
            """
        )
        if df.empty:
            return self._no_data(language)
        lines = []
        current_domain = ""
        for _, row in df.iterrows():
            if row["domain"] != current_domain:
                current_domain = row["domain"]
                lines.append(f"\n📂 {current_domain}:")
            lines.append(f"  • {row['subdomain']}")
        listing = "\n".join(lines)
        if language == "ar":
            return f"التخصصات الفرعية ({len(df)} تخصص):{listing}"
        return f"All subdomains ({len(df)} total):{listing}"

    def _answer_cancellations(self, language: str) -> str:
        if language == "ar":
            return (
                "للاطلاع على تفاصيل الإلغاء، راجع صفحة 'إرشاداتي' أو 'طلباتي'.\n"
                "⚠️ الإلغاء نهائي ولا يمكن التراجع عنه."
            )
        return (
            "For cancellation details, check your 'My Mentorships' or 'My Applications' page.\n"
            "⚠️ Cancellation is permanent and cannot be undone.\n"
            "For more information, please visit the support page."
        )

    def _answer_saved_shared(self, language: str) -> str:
        if language == "ar":
            return (
                "يمكنك حفظ البرامج من صفحة أي برنامج بالضغط على أيقونة الحفظ 🔖\n"
                "تجد محفوظاتك في صفحة 'المحفوظات'.\n"
                "لمزيد من المعلومات، يرجى زيارة صفحة الدعم."
            )
        return (
            "You can save programs from any program page by clicking the save icon 🔖\n"
            "Find your saved items on the 'Saved' page.\n"
            "For more information, please visit the support page."
        )

    # ------------------------------------------------------------------
    # Step-by-step guides (platform how-to)
    # ------------------------------------------------------------------

    def _guide_registration(self, language: str) -> str:
        if language == "ar":
            return (
                "خطوات التسجيل في المنصة:\n"
                "  1. اذهب إلى صفحة التسجيل\n"
                "  2. أدخل بريدك الإلكتروني وكلمة المرور\n"
                "  3. اختر دورك: مرشد أو متدرب\n"
                "  4. أكمل ملفك الشخصي (المجال، الدولة، المهارات)\n"
                "  5. تحقق من بريدك الإلكتروني لتأكيد الحساب\n"
                "  6. ابدأ في استكشاف البرامج والمرشدين!"
            )
        return (
            "How to register on the platform:\n"
            "  1. Go to the registration page\n"
            "  2. Enter your email and create a password\n"
            "  3. Choose your role: Mentor or Mentee\n"
            "  4. Complete your profile (domain, country, skills)\n"
            "  5. Verify your email to activate your account\n"
            "  6. Start exploring programs and mentors!"
        )

    def _guide_apply(self, language: str) -> str:
        programs_count = database.run_scalar("SELECT COUNT(1) FROM programs")
        if language == "ar":
            return (
                f"كيفية التقديم على برنامج إرشادي ({programs_count} برنامج متاح):\n"
                "  1. تصفح البرامج المتاحة في صفحة البحث\n"
                "  2. اختر البرنامج المناسب لمجالك ومستواك\n"
                "  3. راجع متطلبات البرنامج بعناية\n"
                "  4. اضغط على 'تقديم طلب'\n"
                "  5. أجب على أسئلة المرشد (إن وجدت)\n"
                "  6. انتظر قرار المرشد (قبول / رفض)"
            )
        return (
            f"How to apply for a mentorship program ({programs_count} programs available):\n"
            "  1. Browse available programs on the search page\n"
            "  2. Find a program matching your domain and level\n"
            "  3. Review the program requirements carefully\n"
            "  4. Click 'Apply'\n"
            "  5. Answer the mentor's screening questions (if any)\n"
            "  6. Wait for the mentor's decision (accept / reject)"
        )

    def _guide_become_mentor(self, language: str) -> str:
        mentors = database.run_scalar("SELECT COUNT(1) FROM mentor_profile")
        if language == "ar":
            return (
                f"كيف تصبح مرشدًا (حاليًا {mentors} مرشد في المنصة):\n"
                "  1. سجل حساب جديد واختر 'مرشد' كدور\n"
                "  2. أكمل ملفك الشخصي بالكامل\n"
                "  3. أضف خبراتك ومهاراتك التقنية\n"
                "  4. حدد مجالك والمجالات الفرعية\n"
                "  5. أنشئ برنامج إرشادي وحدد المتطلبات\n"
                "  6. انتظر طلبات المتدربين وابدأ الإرشاد!"
            )
        return (
            f"How to become a mentor (currently {mentors} mentors on the platform):\n"
            "  1. Register a new account and choose 'Mentor' as your role\n"
            "  2. Complete your profile in full\n"
            "  3. Add your experience and technical expertise\n"
            "  4. Select your domain and subdomains\n"
            "  5. Create a mentorship program and define requirements\n"
            "  6. Wait for mentee applications and start mentoring!"
        )

    def _guide_mentorship_process(self, language: str) -> str:
        if language == "ar":
            return (
                "كيف تعمل عملية الإرشاد:\n"
                "  1. يبحث المتدرب عن برامج إرشادية مناسبة\n"
                "  2. يقدم المتدرب طلبًا للبرنامج المختار\n"
                "  3. يراجع المرشد الطلب والمتطلبات\n"
                "  4. يقبل أو يرفض المرشد الطلب\n"
                "  5. في حالة القبول، يبدأ الإرشاد رسميًا\n"
                "  6. يتم التواصل وجلسات الإرشاد وفق الخطة\n"
                "  7. عند الانتهاء، يقدم كلا الطرفين تقييمًا"
            )
        return (
            "How the mentorship process works:\n"
            "  1. Mentee browses available mentorship programs\n"
            "  2. Mentee submits an application to a chosen program\n"
            "  3. Mentor reviews the application and requirements\n"
            "  4. Mentor accepts or rejects the application\n"
            "  5. If accepted, the mentorship officially begins\n"
            "  6. Communication and sessions follow the agreed plan\n"
            "  7. Upon completion, both parties submit feedback"
        )

    def _guide_verification(self, language: str) -> str:
        verified = database.run_scalar("SELECT COUNT(1) FROM mentor_profile WHERE is_verified = 1")
        total = database.run_scalar("SELECT COUNT(1) FROM mentor_profile")
        if language == "ar":
            return (
                f"التحقق من المرشد ({verified} من {total} موثقون):\n"
                "  • يتم مراجعة الملف الشخصي والخبرة\n"
                "  • يتم التحقق من الروابط المهنية (LinkedIn)\n"
                "  • يتم مراجعة التوصيات والتقييمات\n"
                "  • بعد التحقق يظهر شارة التوثيق على الملف"
            )
        return (
            f"Mentor verification ({verified} out of {total} are verified):\n"
            "  • Profile and experience are reviewed\n"
            "  • Professional links (LinkedIn) are verified\n"
            "  • Recommendations and ratings are reviewed\n"
            "  • After verification, a badge appears on the profile"
        )

    # ------------------------------------------------------------------
    # Personal query handlers
    # ------------------------------------------------------------------

    def _try_personal_answer(self, query: str, language: str, ctx: dict) -> str | None:
        """Handle queries about the user's own data. Returns None if not matched."""
        from services.user_context_service import user_context_service
        uid = ctx.get("user_id", "")

        # My profile
        if any(t in query for t in ["my profile", "ملفي", "my status", "my info"]):
            name = f"{ctx.get('first_name', '')} {ctx.get('last_name', '')}".strip() or "User"
            role = ctx.get('role', 'user')
            domain = ctx.get('domain_name', 'General')
            country = ctx.get('country_name', '') or ctx.get('country_code', '')
            if language == "ar":
                return (f"ملفك الشخصي:\n  • الاسم: {name}\n  • الدور: {role}"
                        f"\n  • المجال: {domain}\n  • البلد: {country}")
            return (f"Your profile:\n  • Name: {name}\n  • Role: {role}"
                    f"\n  • Domain: {domain}\n  • Country: {country}")

        # My mentorships
        if any(t in query for t in ["my mentor", "my mentorship", "مرشدي", "إرشاداتي"]):
            rows = user_context_service.get_user_mentorships(uid)
            if not rows:
                return "ليس لديك إرشادات حالياً." if language == "ar" else "You have no mentorships yet."
            lines = []
            for r in rows[:5]:
                status = r.get('Status', '?')
                mentor = r.get('mentor_name', '?')
                mentee = r.get('mentee_name', '?')
                domain = r.get('domain', 'General')
                lines.append(f"  • {mentor} ↔ {mentee} — {domain} [{status}]")
            listing = "\n".join(lines)
            if language == "ar":
                return f"إرشاداتك ({len(rows)}):\n{listing}"
            return f"Your mentorships ({len(rows)}):\n{listing}"

        # My applications
        if any(t in query for t in ["my application", "طلباتي", "حالة طلبي"]):
            rows = user_context_service.get_user_applications(uid)
            if not rows:
                return "ليس لديك طلبات حالياً." if language == "ar" else "You have no applications yet."
            lines = []
            for r in rows[:5]:
                title = r.get('program_title', '?')
                status = r.get('Status', '?')
                mentor = r.get('mentor_name', '?')
                lines.append(f"  • {title} — {mentor} [{status}]")
            listing = "\n".join(lines)
            if language == "ar":
                return f"طلباتك ({len(rows)}):\n{listing}"
            return f"Your applications ({len(rows)}):\n{listing}"

        # My programs (for mentors)
        if any(t in query for t in ["my program", "برامجي"]):
            rows = user_context_service.get_user_programs(uid)
            if not rows:
                return "ليس لديك برامج حالياً." if language == "ar" else "You have no programs yet."
            lines = []
            for r in rows[:5]:
                title = r.get('Title', '?')
                domain = r.get('domain', 'General')
                lines.append(f"  • {title} — {domain}")
            listing = "\n".join(lines)
            if language == "ar":
                return f"برامجك ({len(rows)}):\n{listing}"
            return f"Your programs ({len(rows)}):\n{listing}"

        return None

    @staticmethod
    def _is_personal(query: str) -> bool:
        """Check if the query contains possessive language (my/for me)."""
        return any(t in query for t in [
            "my ", "for me", "in my domain", "لي", "مجالي", "الخاص بي",
        ])

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _unavailable(language: str) -> str:
        if language == "ar":
            return "المعلومة غير متوفرة حاليًا. حاول السؤال بطريقة مختلفة أو تواصل مع الدعم."
        return "Information is not available right now. Try rephrasing your question or contact support."

    @staticmethod
    def _no_data(language: str) -> str:
        if language == "ar":
            return "لا توجد بيانات متاحة حاليًا."
        return "No data available at this time."


rag_service = RagService()
