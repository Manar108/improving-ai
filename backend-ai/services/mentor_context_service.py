"""Mentor Context Service — loads mentor profile and program data from database.

Reuses the same DB infrastructure as mentee context service.
Provides mentor-specific data for analytics, workflow help, and FAQs.
"""

import logging
import uuid
from typing import Any, Optional

from database.db import DatabaseAccessError, MissingTableError, database

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Default/empty context
# ─────────────────────────────────────────────────────────────────────

_EMPTY_MENTOR_CONTEXT: dict[str, Any] = {
    "user_id": None,
    "first_name": "",
    "last_name": "",
    "domain_id": None,
    "domain_name": "",
    "years_of_experience": 0,
    "bio": "",
    "is_verified": False,
    "average_rating": 0.0,
    "total_reviews": 0,
    "program_count": 0,
}


# ─────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────

class MentorContextService:
    """Fetch mentor profile and program data from SQL Server.
    
    Every public method returns a plain dict and never raises —
    callers always get a safe default on failure.
    """

    @staticmethod
    def _is_valid_uuid(s: str) -> bool:
        try:
            uuid.UUID(s)
            return True
        except (ValueError, TypeError):
            return False

    def get_mentor_context(self, mentor_id: str) -> dict[str, Any]:
        """Load mentor profile: name, domain, experience, rating, program count."""
        if not self._is_valid_uuid(mentor_id):
            return {**_EMPTY_MENTOR_CONTEXT}

        try:
            df = database.run_query_df(
                """
                SELECT TOP 1
                    mp.user_id,
                    u.first_name,
                    u.last_name,
                    mp.domain_id,
                    d.name AS domain_name,
                    mp.years_of_experience,
                    mp.bio,
                    mp.is_verified,
                    COALESCE(mp.average_rating, 0.0) AS average_rating,
                    COALESCE(mp.total_reviews, 0) AS total_reviews,
                    (SELECT COUNT(*) FROM programs WHERE MentorProfileId = mp.user_id) AS program_count
                FROM mentor_profile mp
                LEFT JOIN users u ON u.user_id = mp.user_id
                LEFT JOIN domains d ON d.domain_id = mp.domain_id
                WHERE mp.user_id = :mentor_id
                """,
                {"mentor_id": str(uuid.UUID(mentor_id))},
            )

            if df.empty:
                logger.info("No mentor found for id=%s", mentor_id)
                return {**_EMPTY_MENTOR_CONTEXT}

            row = df.iloc[0]
            return {
                "user_id": str(row.get("user_id", "")),
                "first_name": str(row.get("first_name", "")),
                "last_name": str(row.get("last_name", "")),
                "domain_id": row.get("domain_id"),
                "domain_name": str(row.get("domain_name", "")),
                "years_of_experience": int(row.get("years_of_experience", 0) or 0),
                "bio": str(row.get("bio", "")),
                "is_verified": bool(row.get("is_verified", False)),
                "average_rating": float(row.get("average_rating", 0.0) or 0.0),
                "total_reviews": int(row.get("total_reviews", 0) or 0),
                "program_count": int(row.get("program_count", 0) or 0),
            }

        except (DatabaseAccessError, MissingTableError) as e:
            logger.warning("DB error loading mentor context: %s", e)
            return {**_EMPTY_MENTOR_CONTEXT}
        except Exception as e:
            logger.error("Unexpected error loading mentor context: %s", e)
            return {**_EMPTY_MENTOR_CONTEXT}

    def get_mentor_programs(self, mentor_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Load mentor's programs with basic stats."""
        if not self._is_valid_uuid(mentor_id):
            return []

        try:
            df = database.run_query_df(
                """
                SELECT TOP (:limit)
                    p.ProgramId AS program_id,
                    p.Title AS title,
                    p.Description AS description,
                    p.DomainId AS domain_id,
                    d.name AS domain_name,
                    p.Capacity AS capacity,
                    p.CreatedAt AS created_at,
                    (SELECT COUNT(*) FROM mentorships m WHERE m.ProgramId = p.ProgramId 
                     AND m.Status = 'Active') AS active_mentees,
                    (SELECT COUNT(*) FROM applications ma 
                     WHERE ma.ProgramId = p.ProgramId) AS applications_count
                FROM programs p
                LEFT JOIN domains d ON d.domain_id = p.DomainId
                WHERE p.MentorProfileId = :mentor_id
                ORDER BY p.CreatedAt DESC
                """,
                {"mentor_id": str(uuid.UUID(mentor_id)), "limit": limit},
            )

            if df.empty:
                return []

            return [
                {
                    "program_id": int(row.get("program_id", 0)),
                    "title": str(row.get("title", "")),
                    "description": str(row.get("description", "")),
                    "domain_id": int(row.get("domain_id", 0) or 0),
                    "domain_name": str(row.get("domain_name", "")),
                    "capacity": int(row.get("capacity", 0) or 0),
                    "created_at": str(row.get("created_at", "")),
                    "active_mentees": int(row.get("active_mentees", 0) or 0),
                    "applications_count": int(row.get("applications_count", 0) or 0),
                }
                for _, row in df.iterrows()
            ]

        except (DatabaseAccessError, MissingTableError) as e:
            logger.warning("DB error loading mentor programs: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error loading mentor programs: %s", e)
            return []

    def get_mentor_active_mentees(self, mentor_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Load mentor's active mentees (current mentorships)."""
        if not self._is_valid_uuid(mentor_id):
            return []

        try:
            df = database.run_query_df(
                """
                SELECT TOP (:limit)
                    m.MenteeProfileId AS mentee_profile_id,
                    u.first_name,
                    u.last_name,
                    mp.domain_id,
                    d.name AS domain_name,
                    m.StartDate AS start_date,
                    m.Status AS status
                FROM mentorships m
                LEFT JOIN mentee_profile mp ON mp.user_id = m.MenteeProfileId
                LEFT JOIN users u ON u.user_id = m.MenteeProfileId
                LEFT JOIN domains d ON d.domain_id = mp.domain_id
                WHERE m.MentorProfileId = :mentor_id AND m.Status = 'Active'
                ORDER BY m.StartDate DESC
                """,
                {"mentor_id": str(uuid.UUID(mentor_id)), "limit": limit},
            )

            if df.empty:
                return []

            return [
                {
                    "mentee_id": str(row.get("mentee_profile_id", "")),
                    "first_name": str(row.get("first_name", "")),
                    "last_name": str(row.get("last_name", "")),
                    "domain_name": str(row.get("domain_name", "")),
                    "start_date": str(row.get("start_date", "")),
                    "status": str(row.get("status", "")),
                }
                for _, row in df.iterrows()
            ]

        except (DatabaseAccessError, MissingTableError) as e:
            logger.warning("DB error loading mentor mentees: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error loading mentor mentees: %s", e)
            return []

    def get_mentor_pending_applications(self, mentor_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Load pending applications for mentor's programs."""
        if not self._is_valid_uuid(mentor_id):
            return []

        try:
            df = database.run_query_df(
                """
                SELECT TOP (:limit)
                    ma.ApplicationId AS application_id,
                    ma.ProgramId AS program_id,
                    p.Title AS program_title,
                    ma.MenteeProfileId AS mentee_profile_id,
                    u.first_name,
                    u.last_name,
                    ma.Status AS status,
                    ma.AppliedAt AS applied_at
                FROM applications ma
                LEFT JOIN programs p ON p.ProgramId = ma.ProgramId
                LEFT JOIN mentee_profile mp ON mp.user_id = ma.MenteeProfileId
                LEFT JOIN users u ON u.user_id = ma.MenteeProfileId
                WHERE p.MentorProfileId = :mentor_id AND ma.Status IN ('Pending', 'Under Review')
                ORDER BY ma.AppliedAt DESC
                """,
                {"mentor_id": str(uuid.UUID(mentor_id)), "limit": limit},
            )

            if df.empty:
                return []

            return [
                {
                    "application_id": int(row.get("application_id", 0)),
                    "program_id": int(row.get("program_id", 0)),
                    "program_title": str(row.get("program_title", "")),
                    "mentee_id": str(row.get("mentee_profile_id", "")),
                    "mentee_name": f"{row.get('first_name', '')} {row.get('last_name', '')}".strip(),
                    "status": str(row.get("status", "")),
                    "applied_at": str(row.get("applied_at", "")),
                }
                for _, row in df.iterrows()
            ]

        except (DatabaseAccessError, MissingTableError) as e:
            logger.warning("DB error loading pending applications: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error loading pending applications: %s", e)
            return []


# Singleton instance
mentor_context_service = MentorContextService()
