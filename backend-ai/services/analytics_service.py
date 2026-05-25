"""Analytics service — lightweight metrics for the chatbot/stats intent.

Provides a single public method `get_stats(language)` which returns a list of
small stat cards suitable for the chat response. The implementation prefers
live DB queries but falls back to safe static defaults on errors.
"""

import logging

from database.db import DatabaseAccessError, MissingTableError, database

logger = logging.getLogger(__name__)


class AnalyticsService:
    def get_stats(self, language: str = "en") -> list[dict]:
        """Return list of stat cards: {label, value}.

        Attempts to query the DB for simple aggregates; on failure returns a
        small set of placeholder cards.
        """
        try:
            # Attempt simple counts if tables exist
            counts = {}
            if database.table_exists("users"):
                counts["users"] = database.run_scalar("SELECT COUNT(1) FROM users")
            if database.table_exists("mentor_profile"):
                counts["mentors"] = database.run_scalar("SELECT COUNT(1) FROM mentor_profile")
            if database.table_exists("mentee_profile"):
                counts["mentees"] = database.run_scalar("SELECT COUNT(1) FROM mentee_profile")
            if database.table_exists("mentorships"):
                counts["mentorships"] = database.run_scalar("SELECT COUNT(1) FROM mentorships")

            cards = []
            if "mentors" in counts:
                cards.append({"label": "Mentors" if language == "en" else "المرشدون", "value": str(int(counts["mentors"]))})
            if "mentees" in counts:
                cards.append({"label": "Mentees" if language == "en" else "المسترشدون", "value": str(int(counts["mentees"]))})
            if "mentorships" in counts:
                cards.append({"label": "Active Mentorships" if language == "en" else "جلسات الإرشاد", "value": str(int(counts["mentorships"]))})
            if "users" in counts:
                cards.append({"label": "Total Users" if language == "en" else "المستخدمون", "value": str(int(counts["users"]))})

            if not cards:
                raise RuntimeError("No DB counts available")

            return cards
        except (MissingTableError, DatabaseAccessError, RuntimeError, Exception) as exc:
            logger.warning("Analytics DB query failed: %s — returning defaults", exc)
            # Safe defaults
            if language == "ar":
                return [
                    {"label": "المرشدون", "value": "—"},
                    {"label": "المسترشدون", "value": "—"},
                    {"label": "جلسات الإرشاد", "value": "—"},
                ]
            return [
                {"label": "Mentors", "value": "—"},
                {"label": "Mentees", "value": "—"},
                {"label": "Mentorships", "value": "—"},
            ]


analytics_service = AnalyticsService()
