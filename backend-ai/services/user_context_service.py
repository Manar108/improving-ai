"""User Context Service — loads user profile and personal data from DB.

This is a TRANSITIONAL layer. All DB access is wrapped in clean methods
so they can be swapped to .NET API calls later without touching callers.

Future migration:
    Replace the body of each method with an HTTP call to the .NET backend:
        get_user_context()  →  GET /api/users/{id}/profile
        get_user_mentors()  →  GET /api/users/{id}/mentorships
        get_user_applications()  →  GET /api/users/{id}/applications
        get_user_programs()  →  GET /api/users/{id}/programs
"""

import logging
import uuid
from typing import Any

from database.db import DatabaseAccessError, MissingTableError, database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data type returned by get_user_context
# ---------------------------------------------------------------------------

_EMPTY_CONTEXT: dict[str, Any] = {
    "user_id": None,
    "first_name": "",
    "last_name": "",
    "role": "",
    "domain_id": None,
    "domain_name": "",
    "country_code": "",
    "country_name": "",
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class UserContextService:
    """Fetch user profile and personal data from SQL Server.

    Every public method returns a plain dict / list[dict] and never raises —
    callers always get a safe default on failure.
    """

    # ------------------------------------------------------------------ #
    # User profile                                                        #
    # ------------------------------------------------------------------ #

    def get_user_context(self, user_id: str) -> dict[str, Any]:
        """Load core profile: name, role, domain, country.

        Future: replace body with  GET /api/users/{user_id}/profile
        """
        if not self._is_valid_uuid(user_id):
            return {**_EMPTY_CONTEXT}

        try:
            df = database.run_query_df(
                """
                SELECT TOP 1
                    u.user_id,
                    u.first_name,
                    u.last_name,
                    CASE
                        WHEN mp.user_id IS NOT NULL AND me.user_id IS NOT NULL THEN 'both'
                        WHEN mp.user_id IS NOT NULL THEN 'mentor'
                        WHEN me.user_id IS NOT NULL THEN 'mentee'
                        ELSE 'user'
                    END AS role,
                    COALESCE(mp.domain_id, me.domain_id)   AS domain_id,
                    COALESCE(d.name, 'General')             AS domain_name,
                    COALESCE(mp.country_code, me.country_code, '')  AS country_code,
                    COALESCE(c.country_name, '')                     AS country_name
                FROM users u
                LEFT JOIN mentor_profile  mp ON mp.user_id = u.user_id
                LEFT JOIN mentee_profile  me ON me.user_id = u.user_id
                LEFT JOIN domains d
                    ON d.domain_id = COALESCE(mp.domain_id, me.domain_id)
                LEFT JOIN countries c
                    ON c.country_code = COALESCE(mp.country_code, me.country_code)
                WHERE u.user_id = :uid
                """,
                {"uid": str(uuid.UUID(user_id))},
            )
            if df.empty:
                logger.info("No user found for id=%s", user_id)
                return {**_EMPTY_CONTEXT}

            row = df.iloc[0]
            return {
                "user_id": str(row.get("user_id", "")),
                "first_name": str(row.get("first_name", "")),
                "last_name": str(row.get("last_name", "")),
                "role": str(row.get("role", "user")),
                "domain_id": row.get("domain_id"),
                "domain_name": str(row.get("domain_name", "General")),
                "country_code": str(row.get("country_code", "")),
                "country_name": str(row.get("country_name", "")),
            }
        except (MissingTableError, DatabaseAccessError) as exc:
            logger.error("get_user_context failed: %s", exc)
            return {**_EMPTY_CONTEXT}

    # ------------------------------------------------------------------ #
    # Personal mentorships                                                #
    # ------------------------------------------------------------------ #

    def get_user_mentorships(self, user_id: str) -> list[dict]:
        """Return mentorships where user is mentor OR mentee.

        Future: replace body with  GET /api/users/{user_id}/mentorships
        """
        if not self._is_valid_uuid(user_id):
            return []
        try:
            uid = str(uuid.UUID(user_id))
            df = database.run_query_df(
                """
                SELECT
                    ms.MentorshipId,
                    ms.Status,
                    CONCAT(um.first_name, ' ', um.last_name)  AS mentor_name,
                    CONCAT(ue.first_name, ' ', ue.last_name)  AS mentee_name,
                    COALESCE(d.name, 'General')                AS domain,
                    ms.StartDate,
                    ms.EndDate
                FROM mentorships ms
                INNER JOIN mentor_profile mp ON mp.user_id = ms.MentorProfileId
                INNER JOIN users um ON um.user_id = mp.user_id
                INNER JOIN mentee_profile mep ON mep.user_id = ms.MenteeProfileId
                INNER JOIN users ue ON ue.user_id = mep.user_id
                LEFT JOIN domains d ON d.domain_id = mp.domain_id
                WHERE mp.user_id = :uid OR mep.user_id = :uid
                ORDER BY ms.StartDate DESC
                """,
                {"uid": uid},
            )
            return df.to_dict(orient="records") if not df.empty else []
        except (MissingTableError, DatabaseAccessError) as exc:
            logger.error("get_user_mentorships failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    # Personal applications                                               #
    # ------------------------------------------------------------------ #

    def get_user_applications(self, user_id: str) -> list[dict]:
        """Return applications submitted by the user.

        Future: replace body with  GET /api/users/{user_id}/applications
        """
        if not self._is_valid_uuid(user_id):
            return []
        try:
            uid = str(uuid.UUID(user_id))
            df = database.run_query_df(
                """
                SELECT
                    a.ApplicationId,
                    a.Status,
                    p.Title           AS program_title,
                    COALESCE(d.name, 'General') AS domain,
                    CONCAT(um.first_name, ' ', um.last_name) AS mentor_name,
                    a.AppliedAt
                FROM applications a
                INNER JOIN programs p ON p.ProgramId = a.ProgramId
                INNER JOIN mentor_profile mp ON mp.user_id = p.MentorProfileId
                INNER JOIN users um ON um.user_id = mp.user_id
                LEFT JOIN domains d ON d.domain_id = p.DomainId
                WHERE a.MenteeProfileId = :uid
                ORDER BY a.AppliedAt DESC
                """,
                {"uid": uid},
            )
            return df.to_dict(orient="records") if not df.empty else []
        except (MissingTableError, DatabaseAccessError) as exc:
            logger.error("get_user_applications failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    # Personal programs (mentor's own programs)                           #
    # ------------------------------------------------------------------ #

    def get_user_programs(self, user_id: str) -> list[dict]:
        """Return programs created by the user (if mentor).

        Future: replace body with  GET /api/users/{user_id}/programs
        """
        if not self._is_valid_uuid(user_id):
            return []
        try:
            uid = str(uuid.UUID(user_id))
            df = database.run_query_df(
                """
                SELECT
                    p.ProgramId,
                    p.Title,
                    COALESCE(d.name, 'General') AS domain,
                    p.CreatedAt
                FROM programs p
                LEFT JOIN domains d ON d.domain_id = p.DomainId
                WHERE p.MentorProfileId = :uid
                ORDER BY p.CreatedAt DESC
                """,
                {"uid": uid},
            )
            return df.to_dict(orient="records") if not df.empty else []
        except (MissingTableError, DatabaseAccessError) as exc:
            logger.error("get_user_programs failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_valid_uuid(value: str) -> bool:
        if not value:
            return False
        try:
            uuid.UUID(str(value))
            return True
        except (ValueError, TypeError):
            return False


user_context_service = UserContextService()
