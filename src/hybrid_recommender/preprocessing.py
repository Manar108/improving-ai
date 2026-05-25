from __future__ import annotations

import hashlib
import json
import logging
import re as _re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeSplitConfig:
    train_end: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp


# ── Shared DB schema mapping (file → pipeline table name + column renaming) ──
# Used by both load_db_datasets (CSV) and load_db_datasets_from_db (SQL Server).
DB_TABLE_MAP = {
    "users": {
        "file": "users.csv",
        "query": "SELECT * FROM users",
        "columns": {
            "UserId": "user_id",
            "Email": "email",
            "PasswordHash": "password_hash",
            "FirstName": "first_name",
            "LastName": "last_name",
            "Role": "role",
            "CreatedAt": "created_at",
            "UpdatedAt": "updated_at",
            "LastLogin": "last_login",
            "IsActive": "is_active",
            "IsEmailVerified": "is_email_verified",
        },
    },
    "mentee_profile": {
        "file": "mentee_profile.csv",
        "query": "SELECT * FROM mentee_profile",
        "columns": {
            "UserId": "user_id",
            "DomainId": "domain_id",
            "EducationStatus": "education_status",
            "CurrentLevel": "current_level",
            "CountryCode": "country_code",
            "CareerGoalId": "career_goal_id",
            "LearningStyleId": "learning_style_id",
            "ProfilePictureUrl": "profile_picture_url",
            "Bio": "bio",
            "IsEmailVerified": "is_email_verified",
        },
    },
    "mentor_profile": {
        "file": "mentor_profile.csv",
        "query": "SELECT * FROM mentor_profile",
        "columns": {
            "UserId": "user_id",
            "DomainId": "domain_id",
            "YearsOfExperience": "years_of_experience",
            "CountryCode": "country_code",
            "LinkedInUrl": "linkedin_url",
            "ProfilePictureUrl": "profile_picture_url",
            "CreatedAt": "created_at",
            "AverageRating": "average_rating",
            "TotalReviews": "total_reviews",
            "IsVerified": "is_verified",
            "Bio": "bio",
            "CvUrl": "cv_url",
            "PastExperience": "past_experience",
            "IsEmailVerified": "is_email_verified",
        },
    },
    "mentee_subdomains": {
        "file": "MenteeSubDomains.csv",
        "query": "SELECT * FROM MenteeSubDomains",
        "columns": {
            "UserId": "mentee_id",
            "SubDomainId": "subdomain_id",
        },
    },
    "mentor_subdomains": {
        "file": "MentorSubDomains.csv",
        "query": "SELECT * FROM MentorSubDomains",
        "columns": {
            "MentorId": "mentor_id",
            "SubDomainId": "subdomain_id",
        },
    },
    "mentor_expertise": {
        "file": "mentor_expertise.csv",
        "query": "SELECT * FROM mentor_expertise",
        "columns": {
            "MentorId": "mentor_id",
            "TechnologyId": "technology_id",
        },
    },
    "mentee_interests": {
        "file": "mentee_interests.csv",
        "query": "SELECT * FROM mentee_interests",
        "columns": {
            "UserId": "mentee_id",
            "TechnologyId": "technology_id",
            "ExperienceLevel": "experience_level",
        },
    },
    "mentorship_posts": {
        "file": "programs.csv",
        "query": "SELECT * FROM programs",
        "columns": {
            "ProgramId": "post_id",
            "MentorProfileId": "mentor_id",
            "DomainId": "domain_id",
            "SubDomainId": "subdomain_id",
            "Title": "title",
            "Description": "description",
            "TargetLevel": "target_level",
            "EducationLevel": "education_level",
            "Capacity": "capacity",
            "ProgramPostStatus": "is_open",
            "CreatedAt": "created_at",
            "Availability": "availability",
            "Duration": "duration",
            "RoadmapId": "roadmap_id",
            "Deadline": "deadline",
        },
    },
    "mentorship_requirements": {
        "file": "mentorship_requirements.csv",
        "query": "SELECT * FROM mentorship_requirements",
        "columns": {
            "MentorshipRequirementId": "requirement_id",
            "ProgramId": "post_id",
            "TechnologyId": "technology_id",
            "RequiredExperienceLevel": "required_experience_level",
        },
    },
    "mentorships": {
        "file": "mentorships.csv",
        "query": "SELECT * FROM mentorships",
        "columns": {
            "MentorshipId": "mentorship_id",
            "MentorProfileId": "mentor_id",
            "MenteeProfileId": "mentee_id",
            "ProgramId": "post_id",
            "Status": "status",
            "StartDate": "start_date",
            "EndDate": "end_date",
        },
    },
    "mentorship_applications": {
        "file": "applications.csv",
        "query": "SELECT * FROM applications",
        "columns": {
            "ApplicationId": "app_id",
            "ProgramId": "post_id",
            "MenteeProfileId": "mentee_id",
            "Status": "status",
            "AppliedAt": "applied_at",
            "DecisionAt": "decisioned_at",
            "MeetRequirements": "meet_requirements",
        },
    },
    "mentors_feedback": {
        "file": "feedbacks.csv",
        "query": "SELECT * FROM feedbacks",
        "columns": {
            "FeedbackId": "feedback_id",
            "MentorshipId": "mentorship_id",
            "MentorProfileId": "mentor_id",
            "MenteeProfileId": "mentee_id",
            "Rating": "rating",
            "Comment": "text",
            "CreatedAt": "created_at",
        },
    },
    "follows": {
        "file": "follows.csv",
        "query": "SELECT * FROM follows",
        "columns": {
            "Id": "follow_id",
            "FollowerId": "follower_id",
            "FollowingId": "following_id",
            "FollowedAt": "created_at",
        },
    },
    "mentorship_cancellation": {
        "file": "mentorships_cancellation.csv",
        "query": "SELECT * FROM mentorships_cancellation",
        "columns": {
            "Id": "cancellation_id",
            "MentorshipId": "mentorship_id",
            "ProgramId": "post_id",
            "MenteeProfileId": "mentee_id",
            "MentorProfileId": "mentor_id",
            "CancellationActor": "cancellation_actor",
            "CancellationDate": "cancellation_date",
            "CancellationReasonValue": "cancellation_reason",
        },
    },
    "posts_likes_dataset": {
        "file": ["community_post_likes.csv", "mentorship_post_likes.csv"],
        "query": "SELECT * FROM post_likes",
        "columns": {
            "LikeId": "like_id",
            "ProgramId": "post_id",
            "UserId": "user_id",
            "CreatedAt": "created_at",
        },
    },
    "posts_comments": {
        "file": ["community_post_comments.csv", "mentorship_post_comments.csv"],
        "query": "SELECT * FROM [Post-Comment]",
        "columns": {
            "CommentId": "comment_id",
            "ProgramId": "post_id",
            "UserId": "user_id",
            "CommentText": "comment",
            "CreatedAt": "created_at",
            "IsDeleted": "is_deleted",
        },
    },
    "saved_posts_dataset": {
        "file": "saved_posts.csv",
        "query": "SELECT * FROM saved_posts",
        "columns": {
            "SaveId": "save_id",
            "ProgramId": "post_id",
            "UserId": "user_id",
            "CreatedAt": "created_at",
        },
    },
    "shared_posts_dataset": {
        "file": "shared_posts.csv",
        "query": "SELECT * FROM shared_posts",
        "columns": {
            "ShareId": "share_id",
            "ProgramId": "post_id",
            "UserId": "sender_id",
            "SharedAt": "shared_at",
        },
    },
    "countries": {
        "file": "countries.csv",
        "query": "SELECT * FROM countries",
        "columns": {
            "CountryCode": "country_code",
            "CountryName": "country_name",
        },
    },
    "domains": {
        "file": "domains.csv",
        "query": "SELECT * FROM domains",
        "columns": {
            "DomainId": "domain_id",
            "Name": "name",
            "Description": "description",
        },
    },
    "subdomains": {
        "file": "subdomain.csv",
        "query": "SELECT * FROM subdomain",
        "columns": {
            "SubDomainId": "subdomain_id",
            "DomainId": "domain_id",
            "Name": "name",
        },
    },
    "technologies": {
        "file": "technologies.csv",
        "query": "SELECT * FROM technologies",
        "columns": {
            "TechnologyId": "technology_id",
            "Name": "name",
            "ProgramId": "post_id",
            "SubDomainId": "subdomain_id",
        },
    },
}


# ── Canonical experience values (aligned with backend ExperienceLevel enum) ──
# Backend: None=1, Beginner=2, Intermediate=3, Advanced=4
_CANONICAL_EXPERIENCE_VALUES = frozenset({"none", "beginner", "intermediate", "advanced"})

# Legacy aliases from raw/CSV data.  Kept for backward compatibility;
# production DB should already use canonical integer enum values.
_EXPERIENCE_ALIASES: dict[str, str] = {
    "junior": "beginner",
    "mid": "intermediate",
    "senior": "advanced",
}

# Integer → canonical string mapping for DB integer enum values.
# EF Core stores CurrentLevel and ExperienceLevel as integers (no HasConversion).
# Backend enum: None=1, Beginner=2, Intermediate=3, Advanced=4
# Also handle legacy 0-based values from old seeded data.
_INT_TO_EXPERIENCE: dict[str, str] = {
    # 1-based (real .NET backend values)
    "1": "none",
    "2": "beginner",
    "3": "intermediate",
    "4": "advanced",
    # 0-based (legacy seeded data) — kept for backward compatibility
    "0": "none",
}

# Strings that represent intentional "no experience" (not missing data).
_NO_EXPERIENCE_SENTINELS = frozenset({"", "none", "null", "nan", "n/a", "na"})


def normalize_experience(value) -> str:
    """Normalize a single experience value to a canonical form.

    Canonical values (aligned with backend ``ExperienceLevel`` enum):
      none, beginner, intermediate, advanced

    Handles:
      - String canonical values ("beginner", "intermediate", etc.)
      - Legacy string aliases ("junior" → "beginner", etc.)
      - Integer enum values from DB (1→none, 2→beginner, etc.)
      - Legacy 0-based integers (0→none, 1→beginner, etc.)
      - Missing/null values → "none"
    """
    if pd.isna(value):
        return "none"
    raw = str(value).strip().lower()
    if raw in _NO_EXPERIENCE_SENTINELS:
        return "none"
    # Direct canonical match (fast path)
    if raw in _CANONICAL_EXPERIENCE_VALUES:
        return raw
    # Integer enum value from DB (e.g., "1", "2", "3", "4")
    if raw in _INT_TO_EXPERIENCE:
        canonical = _INT_TO_EXPERIENCE[raw]
        logger.debug(
            "normalize_experience: integer enum '%s' → '%s'",
            raw, canonical,
        )
        return canonical
    # Legacy alias
    if raw in _EXPERIENCE_ALIASES:
        canonical = _EXPERIENCE_ALIASES[raw]
        logger.debug(
            "normalize_experience: legacy alias '%s' → '%s' (consider updating source data)",
            raw, canonical,
        )
        return canonical
    # Unknown value — warn and fallback
    logger.warning(
        "normalize_experience: unknown value '%s' — falling back to 'none'. "
        "If this is a valid level, add it to _CANONICAL_EXPERIENCE_VALUES.",
        raw,
    )
    return "none"


def filter_by_time(df: pd.DataFrame, time_column: str, max_time: pd.Timestamp) -> pd.DataFrame:
    """Filter a DataFrame to rows where ``time_column <= max_time``.

    Returns the data unfiltered (with a warning) if the time column is missing.
    """
    if time_column not in df.columns:
        logger.warning(
            "filter_by_time: column '%s' not found (cols=%s) "
            "— returning unfiltered data. This may cause data leakage.",
            time_column, list(df.columns)[:5],
        )
        return df
    out = df.copy()
    out[time_column] = pd.to_datetime(out[time_column], errors="coerce")
    return out[out[time_column].notna() & (out[time_column] <= max_time)].copy()


EXPERIENCE_COLUMNS_BY_DATASET = {
    "mentee_interests": ["experience_level"],
    "mentorship_requirements": ["required_experience_level"],
}

IGNORE_NULL_COLUMNS = {
    "mentorship_applications": ["decisioned_at"],
    "mentors_feedback": ["cancellation_actor"],
}

TIME_TABLES = {
    "mentorship_posts": "created_at",
    # mentorship_applications is the PRIMARY LABEL SOURCE (label=1 if user applied).
    # It is NOT time-filtered here — instead it receives a time_split (train/valid/test)
    # so labels can be assigned per-split.  Train-only filtering for FEATURES happens
    # in pipeline.py before passing to build_mentor_features.
    "mentorship_applications": "applied_at",
    "mentorship_cancellation": "cancellation_date",
    "posts_likes_dataset": "created_at",
    "posts_comments": "created_at",
    "saved_posts_dataset": "created_at",
    "shared_posts_dataset": "shared_at",
    "follows": "created_at",
    "mentors_feedback": "created_at",
}

# --- SCD Type 1 Policy ---
# The following tables represent a user's *current identity* (skills, interests,
# subdomains) which change infrequently.  They are treated as Slowly-Changing
# Dimensions (Type 1) and are intentionally NOT time-filtered:
#   - mentee_subdomains, mentor_subdomains
#   - mentee_interests, mentor_expertise
#   - mentee_profile, mentor_profile
# If the platform begins tracking historical profile changes, these tables
# should be versioned and time-filtered accordingly.


def _strip_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all text columns (vectorized str.strip)."""
    out = df.copy()
    object_cols = out.select_dtypes(include=["object"]).columns
    for col in object_cols:
        out[col] = out[col].str.strip().where(out[col].notna(), other=out[col])
    return out


def _enforce_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all ID columns to numeric types."""
    out = df.copy()
    id_columns = [col for col in out.columns if "id" in col.lower()]
    for col in id_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _drop_invalid_ids(df: pd.DataFrame, table_name: str = "") -> pd.DataFrame:
    """Drop rows with invalid IDs using table-specific rules.

    Interaction tables (applications, feedback, mentorships, follows, likes,
    comments, saves, shares) require ALL their key ID columns to be present.
    Entity tables (profiles) only drop rows where ALL IDs are null.
    Nullable business fields like ``roadmap_id`` are never used for dropping.
    """
    id_cols = [c for c in df.columns if "id" in c.lower()]
    if not id_cols:
        return df

    # Nullable FK columns that should never trigger row-drops
    _NULLABLE_FK_COLS = {"roadmap_id", "career_goal_id", "learning_style_id"}

    # Interaction tables: require ALL key IDs (partially-broken rows are useless)
    _INTERACTION_REQUIRED_IDS: dict[str, list[str]] = {
        "mentorship_applications": ["post_id", "mentee_id"],
        "mentors_feedback":        ["mentorship_id", "mentor_id", "mentee_id"],
        "mentorships":             ["mentorship_id", "mentor_id", "mentee_id"],
        "follows":                 ["follower_id", "following_id"],
        "posts_likes_dataset":     ["user_id", "post_id"],
        "posts_comments":          ["user_id", "post_id"],
        "saved_posts_dataset":     ["user_id", "post_id"],
        "shared_posts_dataset":    ["user_id", "post_id"],
        "mentorship_cancellation": ["mentorship_id"],
        "mentee_subdomains":       ["mentee_id", "subdomain_id"],
        "mentor_subdomains":       ["mentor_id", "subdomain_id"],
        "mentee_interests":        ["mentee_id", "technology_id"],
        "mentor_expertise":        ["mentor_id", "technology_id"],
        "mentorship_requirements": ["post_id", "technology_id"],
    }

    required = _INTERACTION_REQUIRED_IDS.get(table_name)
    if required:
        # Only check columns that actually exist in this DataFrame
        check_cols = [c for c in required if c in df.columns]
        if check_cols:
            before = len(df)
            df = df.dropna(subset=check_cols, how="any")
            dropped = before - len(df)
            if dropped > 0:
                logger.info(
                    "_drop_invalid_ids [%s]: dropped %d rows with null required IDs %s",
                    table_name, dropped, check_cols,
                )
            return df

    # Entity / lookup tables: drop only when ALL IDs are null
    non_nullable_ids = [c for c in id_cols if c not in _NULLABLE_FK_COLS]
    if non_nullable_ids:
        return df.dropna(subset=non_nullable_ids, how="all")
    return df


def _replace_empty_strings_with_na(df: pd.DataFrame, ignored_cols: list[str]) -> pd.DataFrame:
    """Replace empty strings with NA in text columns (except ignored columns)."""
    out = df.copy()
    object_cols = [c for c in out.select_dtypes(include=["object"]).columns if c not in ignored_cols]
    if object_cols:
        out[object_cols] = out[object_cols].replace("", pd.NA)
    return out


# ── Field-aware imputation rules ──
# Maps column names (or patterns) to their appropriate fill strategy.
# This prevents the ML-harmful practice of filling all numerics with 0.
_MEDIAN_FILL_COLUMNS = {"years_of_experience", "average_rating", "rating"}
_ZERO_FILL_COLUMNS = {"total_reviews", "capacity"}  # Counters: 0 is semantically correct


def _handle_missing(df: pd.DataFrame, ignored_object_cols: list[str] | None = None) -> pd.DataFrame:
    """Fill missing values with field-aware imputation.

    Strategy per column type:
      - Ratings / years_of_experience → column median (0 ≠ missing for these)
      - Counters (total_reviews, capacity) → 0 (genuinely zero)
      - Other numerics → 0 (safe default for flags, encoded categoricals)
      - Text columns → 'unknown'

    ID columns are never filled (they indicate genuinely missing references).
    """
    out = df.copy()
    ignored_object_cols = set(ignored_object_cols or [])
    id_cols = {c for c in out.columns if "id" in c.lower()}
    numeric_cols = [c for c in out.select_dtypes(include=["number"]).columns if c not in id_cols]
    object_cols = [
        c for c in out.select_dtypes(include=["object", "string"]).columns
        if c not in ignored_object_cols
    ]

    for col in numeric_cols:
        n_missing = out[col].isna().sum()
        if n_missing == 0:
            continue
        if col in _MEDIAN_FILL_COLUMNS:
            median_val = out[col].median()
            # If entire column is NaN, median is NaN → fall back to 0
            fill_val = median_val if pd.notna(median_val) else 0
            out[col] = out[col].fillna(fill_val)
            logger.debug(
                "_handle_missing: filled %d nulls in '%s' with median=%.2f",
                n_missing, col, fill_val,
            )
        else:
            # Counters, flags, and other numerics: 0 is appropriate
            out[col] = out[col].fillna(0)

    if len(object_cols) > 0:
        out[object_cols] = out[object_cols].fillna("unknown")
    return out


# Numeric encoding for canonical experience levels.
# Aligned with backend ExperienceLevel enum: None=1, Beginner=2, Intermediate=3, Advanced=4.
_EXPERIENCE_TO_NUM: dict[str, int] = {
    "none": 1,
    "beginner": 2,
    "intermediate": 3,
    "advanced": 4,
}


def map_experience_to_num(val) -> int:
    """Map a *canonical* experience string to a numeric level (1–4).

    Input should already be normalized via ``normalize_experience()``.
    Unknown values return 1 (none) with a warning.
    """
    result = _EXPERIENCE_TO_NUM.get(val)
    if result is not None:
        return result
    logger.warning("map_experience_to_num: unexpected value '%s' — mapping to 1", val)
    return 1


def _normalize_experience_columns(df: pd.DataFrame, experience_cols: list[str]) -> pd.DataFrame:
    """Normalize and encode experience columns.

    For each experience column:
      1. Maps raw values to canonical strings via ``normalize_experience()``.
      2. Logs a summary of any non-canonical values encountered.
      3. Adds a ``{col}_num`` integer encoding column.
    """
    out = df.copy()
    for col in experience_cols:
        if col not in out.columns:
            continue
        raw_values = out[col].copy()
        normalized = out[col].map(normalize_experience)
        # Log summary of non-trivial normalizations (aliases + unknowns)
        changed_mask = (raw_values.astype(str).str.strip().str.lower() != normalized) & raw_values.notna()
        if changed_mask.any():
            changes = raw_values[changed_mask].value_counts().head(10)
            logger.info(
                "_normalize_experience_columns [%s]: normalized %d values — top changes: %s",
                col, changed_mask.sum(), changes.to_dict(),
            )
        out[col] = normalized.astype(object)
        out[f"{col}_num"] = normalized.map(map_experience_to_num).fillna(1).astype(int)
    return out


def _drop_duplicates(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply table-specific deduplication rules.

    Each table has a different business key (profile, bridge, interaction).
    """
    if name == "mentorships" and "mentorship_id" in df.columns:
        return df.drop_duplicates(subset=["mentorship_id"])

    if name in {"mentee_profile", "mentor_profile"} and "user_id" in df.columns:
        return df.drop_duplicates(subset=["user_id"])

    if name == "mentee_subdomains" and {"mentee_id", "subdomain_id"}.issubset(df.columns):
        return df.drop_duplicates(subset=["mentee_id", "subdomain_id"])

    if name == "mentor_subdomains" and {"mentor_id", "subdomain_id"}.issubset(df.columns):
        return df.drop_duplicates(subset=["mentor_id", "subdomain_id"])

    if name == "mentor_expertise" and {"mentor_id", "technology_id"}.issubset(df.columns):
        return df.drop_duplicates(subset=["mentor_id", "technology_id"])

    if name == "mentee_interests" and {"mentee_id", "technology_id"}.issubset(df.columns):
        return df.drop_duplicates(subset=["mentee_id", "technology_id"])

    if name == "mentorship_requirements" and {"post_id", "technology_id"}.issubset(df.columns):
        return df.drop_duplicates(subset=["post_id", "technology_id"])

    if name == "posts_comments" and {"user_id", "post_id", "comment"}.issubset(df.columns):
        return df.drop_duplicates(subset=["user_id", "post_id", "comment"])

    if name == "posts_likes_dataset" and {"user_id", "post_id"}.issubset(df.columns):
        return df.drop_duplicates(subset=["user_id", "post_id"])

    if name == "saved_posts_dataset" and {"user_id", "post_id"}.issubset(df.columns):
        return df.drop_duplicates(subset=["user_id", "post_id"])

    if name == "shared_posts_dataset" and {"user_id", "post_id"}.issubset(df.columns):
        return df.drop_duplicates(subset=["user_id", "post_id"])

    return df


def _validate_feedback_table(feedback: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean feedback table: normalize sentiment, enforce rating range.

    Invalid values are actively cleaned (not just warned about):
      - Ratings outside 1–5 are clamped to [1, 5].
      - Invalid sentiments are replaced with 'neutral' (safe default).
    Logs clear summaries of all corrections applied.
    """
    out = feedback.copy()

    # ── Sentiment validation ──
    _VALID_SENTIMENTS = {"positive", "neutral", "negative"}
    if "sentiment" in out.columns:
        normalized_sentiment = out["sentiment"].astype("string").str.strip().str.lower()
        invalid_mask = ~normalized_sentiment.isin(_VALID_SENTIMENTS)
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            invalid_examples = normalized_sentiment[invalid_mask].value_counts().head(5).to_dict()
            logger.warning(
                "_validate_feedback_table: %d rows with invalid sentiment %s — "
                "replacing with 'neutral'",
                n_invalid, invalid_examples,
            )
            normalized_sentiment = normalized_sentiment.where(~invalid_mask, "neutral")
        out["sentiment"] = normalized_sentiment.astype(object)

    # ── Rating validation ──
    if "rating" in out.columns:
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
        n_null_rating = out["rating"].isna().sum()
        if n_null_rating > 0:
            logger.warning(
                "_validate_feedback_table: %d rows with non-numeric rating — dropping",
                n_null_rating,
            )
            out = out[out["rating"].notna()].copy()

        below_min = (out["rating"] < 1).sum()
        above_max = (out["rating"] > 5).sum()
        if below_min > 0 or above_max > 0:
            logger.warning(
                "_validate_feedback_table: clamping %d ratings below 1 and %d above 5 to [1, 5]",
                below_min, above_max,
            )
            out["rating"] = out["rating"].clip(lower=1, upper=5)

    return out


def load_raw_datasets(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all raw CSV tables from the data directory.

    Raises ``FileNotFoundError`` if critical datasets are missing.
    """
    dataset_files = {
        "users": "users.csv",
        "mentee_profile": "mentee_profile.csv",
        "mentor_profile": "mentor_profile.csv",
        "mentee_subdomains": "mentee_subdomains.csv",
        "mentor_subdomains": "mentor_subdomains.csv",
        "mentor_expertise": "mentor_expertise.csv",
        "mentee_interests": "mentee_interests.csv",
        "mentorship_posts": "mentorship_posts.csv",
        "mentorship_requirements": "mentorship_requirements.csv",
        "mentorships": "mentorships.csv",
        "mentorship_applications": "mentorship_applications.csv",
        "mentors_feedback": "mentors_feedback.csv",
        "posts_likes_dataset": ["community_post_likes.csv", "mentorship_post_likes.csv"],
        "posts_comments": ["community_post_comments.csv", "mentorship_post_comments.csv"],
        "saved_posts_dataset": "saved_posts.csv",
        "shared_posts_dataset": "shared_posts.csv",
        "follows": "follows.csv",
        "mentorship_cancellation": "mentorship_cancellations.csv",
        "communities": "communities_updated.csv",
        "community_members": "community_members_v3.csv",
        "countries": "countries.csv",
        "domains": "domains.csv",
        "subdomains": "subdomains.csv",
        "technologies": "technologies.csv",
    }

    tables: Dict[str, pd.DataFrame] = {}
    missing = []
    for name, filename_or_list in dataset_files.items():
        filenames = [filename_or_list] if isinstance(filename_or_list, str) else filename_or_list
        dfs = []
        for filename in filenames:
            path = raw_dir / filename
            if path.exists():
                dfs.append(pd.read_csv(path, keep_default_na=False))
        if dfs:
            tables[name] = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        else:
            missing.append(name)

    if missing:
        logger.warning("load_raw_datasets: missing files in %s:\n  %s", raw_dir, "\n  ".join(missing))

    CRITICAL_DATASETS = {
        "mentee_profile", "mentor_profile",
        "mentee_subdomains", "mentor_subdomains",
        "mentee_interests", "mentor_expertise",
        "mentorship_posts", "mentorships",
        "mentorship_applications",
    }
    missing_critical = CRITICAL_DATASETS & set(missing)
    if missing_critical:
        raise FileNotFoundError(
            f"load_raw_datasets: critical datasets missing — pipeline cannot proceed: "
            f"{sorted(missing_critical)}"
        )

    return tables


def load_db_datasets(db_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load datasets from the database_import_ready directory.

    The DB uses PascalCase column names (e.g. ``UserId``, ``MentorProfileId``,
    ``ProgramId``) and different table names (``programs`` instead of
    ``mentorship_posts``, ``applications`` instead of ``mentorship_applications``).

    This function reads each DB-format CSV, renames columns to the snake_case
    format expected by the pipeline, and returns the same dict structure as
    ``load_raw_datasets()``.

    Missing optional tables (community posts, likes, comments) are silently
    skipped — the pipeline handles their absence gracefully.
    """

    tables: Dict[str, pd.DataFrame] = {}
    missing = []

    for table_name, spec in DB_TABLE_MAP.items():
        filenames = spec["file"] if isinstance(spec["file"], list) else [spec["file"]]
        dfs = []
        for filename in filenames:
            path = db_dir / filename
            if path.exists():
                dfs.append(pd.read_csv(path, keep_default_na=False))
        if not dfs:
            missing.append(f"{table_name} ({', '.join(filenames)})")
            continue

        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # Rename only columns that exist in the file
        rename_map = {k: v for k, v in spec["columns"].items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Special handling: ProgramPostStatus → is_open needs value mapping
        # DB uses "Draft"/"Published", pipeline expects "draft"/"published"
        if table_name == "mentorship_posts" and "is_open" in df.columns:
            df["is_open"] = df["is_open"].astype(str).str.strip().str.lower()

        # Normalize availability column (separate from publication state)
        # DB uses "Open"/"Closed", pipeline expects "open"/"closed"
        if table_name == "mentorship_posts" and "availability" in df.columns:
            df["availability"] = df["availability"].astype(str).str.strip().str.lower()

        tables[table_name] = df

    if missing:
        logger.warning(
            "load_db_datasets: missing files in %s:\n  %s",
            db_dir, "\n  ".join(missing),
        )

    CRITICAL_DATASETS = {
        "mentee_profile", "mentor_profile",
        "mentee_subdomains", "mentor_subdomains",
        "mentee_interests", "mentor_expertise",
        "mentorship_posts", "mentorships",
        "mentorship_applications",
    }
    missing_critical = CRITICAL_DATASETS - set(tables.keys())
    if missing_critical:
        raise FileNotFoundError(
            f"load_db_datasets: critical datasets missing — pipeline cannot proceed: "
            f"{sorted(missing_critical)}"
        )

    loaded_tables = sorted(tables.keys())
    logger.info(
        "load_db_datasets: loaded %d tables from %s:\n  %s",
        len(tables), db_dir, ", ".join(loaded_tables),
    )

    return tables


def _is_uuid(value) -> bool:
    """Return True if value looks like a UUID string."""
    if pd.isna(value):
        return False
    val_str = str(value).strip()
    if not val_str:
        return False
    try:
        int(val_str)
        return False
    except (ValueError, TypeError):
        return True


def _build_uuid_mapping(raw_tables: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """Build UUID → integer mapping for all user-related ID columns."""
    mapping: Dict[str, int] = {}
    next_id = 1

    user_related_cols = {"user_id", "mentee_id", "mentor_id", "follower_id", "following_id"}

    for df in raw_tables.values():
        for col in df.columns:
            if col.lower() not in user_related_cols:
                continue
            for val in df[col].dropna().unique():
                val_str = str(val).strip()
                if _is_uuid(val_str) and val_str not in mapping:
                    mapping[val_str] = next_id
                    next_id += 1

    return mapping


def _apply_uuid_mapping(raw_tables: Dict[str, pd.DataFrame], mapping: Dict[str, int]) -> Dict[str, pd.DataFrame]:
    """Apply UUID → integer mapping to all user-related ID columns."""
    user_related_cols = {"user_id", "mentee_id", "mentor_id", "follower_id", "following_id"}
    result: Dict[str, pd.DataFrame] = {}

    for table_name, df in raw_tables.items():
        df = df.copy()
        for col in df.columns:
            if col.lower() in user_related_cols:
                df[col] = df[col].map(lambda x: mapping.get(str(x).strip(), x) if pd.notna(x) else x)
                df[col] = pd.to_numeric(df[col], errors="coerce")
        result[table_name] = df

    return result


def load_db_datasets_from_db() -> Dict[str, pd.DataFrame]:
    """Load datasets directly from the production SQL Server database.

    Uses the DatabaseClient from ``backend-ai/database/db.py``.
    Renames columns from PascalCase to snake_case (same as ``load_db_datasets``),
    and handles UUID → integer mapping for user_id fields when the DB uses
    UUID strings instead of integers.

    The UUID mapping is saved to ``data/artifacts/uuid_mapping.json`` so
    downstream inference can reverse-map integer IDs back to UUIDs.

    Returns:
        Dict[str, pd.DataFrame]: Same structure as ``load_db_datasets``.
    """
    import sys
    from pathlib import Path

    backend_ai = Path(__file__).resolve().parents[2] / "backend-ai"
    if str(backend_ai) not in sys.path:
        sys.path.insert(0, str(backend_ai))

    # pyrefly: ignore [missing-import]
    from database.db import database    

    tables: Dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for table_name, spec in DB_TABLE_MAP.items():
        try:
            df = database.run_query_df(spec["query"])
        except Exception as e:
            logger.warning("load_db_datasets_from_db: %s query failed: %s", table_name, e)
            missing.append(table_name)
            continue

        # Rename only columns that exist in the result.
        # The production DB uses snake_case column names, while the legacy CSV
        # export used PascalCase.  We match both forms so ``DB_TABLE_MAP`` can
        # continue to serve both loaders without duplication.
        import re as _re

        rename_map: dict[str, str] = {}
        for k, v in spec["columns"].items():
            if k in df.columns:
                rename_map[k] = v
                continue
            # Try snake_case variant: UserId -> user_id, MentorProfileId -> mentor_profile_id
            snake = _re.sub(r"(?<!^)(?=[A-Z])", "_", k).lower()
            if snake in df.columns and snake not in rename_map.values():
                rename_map[snake] = v
                continue
            # Try plain lowercase (fallback for simple cases like "Id" -> "id")
            plain = k.lower()
            if plain in df.columns and plain not in rename_map.values():
                rename_map[plain] = v

        df = df.rename(columns=rename_map)

        # Special handling for mentorship_posts status values
        if table_name == "mentorship_posts" and "is_open" in df.columns:
            df["is_open"] = df["is_open"].astype(str).str.strip().str.lower()

        # Normalize availability column (separate from publication state)
        if table_name == "mentorship_posts" and "availability" in df.columns:
            df["availability"] = df["availability"].astype(str).str.strip().str.lower()

        # Parse Deadline if present in mentorship_posts
        if table_name == "mentorship_posts" and "deadline" in df.columns:
            df["deadline"] = pd.to_datetime(df["deadline"], errors="coerce")

        tables[table_name] = df

    CRITICAL_DATASETS = {
        "mentee_profile", "mentor_profile",
        "mentee_subdomains", "mentor_subdomains",
        "mentee_interests", "mentor_expertise",
        "mentorship_posts", "mentorships",
        "mentorship_applications",
    }
    missing_critical = CRITICAL_DATASETS - set(tables.keys())
    if missing_critical:
        raise RuntimeError(
            f"load_db_datasets_from_db: critical datasets missing — pipeline cannot proceed: "
            f"{sorted(missing_critical)}"
        )

    # ── UUID → integer mapping ──
    uuid_mapping = _build_uuid_mapping(tables)
    if uuid_mapping:
        logger.info(
            "load_db_datasets_from_db: detected %d UUID user IDs — building integer mapping",
            len(uuid_mapping),
        )
        tables = _apply_uuid_mapping(tables, uuid_mapping)

        # Save mapping artifacts for downstream inference
        artifacts_dir = Path(__file__).resolve().parents[2] / "data" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = artifacts_dir / "uuid_mapping.json"
        rev_mapping = {str(v): k for k, v in uuid_mapping.items()}
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({"uuid_to_int": uuid_mapping, "int_to_uuid": rev_mapping}, f, indent=2)
        logger.info("Saved UUID mapping to %s", mapping_path)
    else:
        logger.info("load_db_datasets_from_db: no UUIDs detected — using native integer IDs")

    loaded_tables = sorted(tables.keys())
    logger.info(
        "load_db_datasets_from_db: loaded %d tables from database:\n  %s",
        len(tables), ", ".join(loaded_tables),
    )

    return tables


def parse_datetime_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Parse a column to datetime, coercing invalid values to NaT."""
    out = df.copy()
    if column in out.columns:
        out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def build_time_split_config(
    mentorships: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    applications: pd.DataFrame | None = None,
) -> TimeSplitConfig:
    """Build a fixed train/valid/test time split.

    When ``applications`` is provided, computes split boundaries from
    ``applied_at`` dates.  This is critical because applications are the
    label source — splitting on mentorship ``start_date`` can produce
    severely unbalanced label distributions (e.g. 2% test positives).

    Falls back to mentorship ``start_date`` if applications are not provided.

    Uses quantile-based boundaries to ensure proportional temporal splits.
    """
    # Prefer application dates (label source) for split boundaries
    if applications is not None and "applied_at" in applications.columns:
        dates = pd.to_datetime(applications["applied_at"], errors="coerce").dropna().sort_values()
        if not dates.empty:
            logger.info(
                "build_time_split_config: using application dates (%d records) "
                "for split boundaries (label-aligned split).",
                len(dates),
            )
        else:
            dates = None
    else:
        dates = None

    # Fallback to mentorship start_date
    if dates is None or dates.empty:
        if "start_date" not in mentorships.columns:
            raise ValueError("mentorships must contain start_date")
        dates = pd.to_datetime(mentorships["start_date"], errors="coerce").dropna().sort_values()
        if dates.empty:
            raise ValueError("No valid date values available for time split")
        logger.info(
            "build_time_split_config: using mentorship start_date (%d records) "
            "for split boundaries (fallback).",
            len(dates),
        )

    train_end = dates.quantile(train_ratio).normalize()
    valid_end = dates.quantile(train_ratio + valid_ratio).normalize()
    if valid_end <= train_end:
        valid_end = train_end + pd.Timedelta(days=1)

    test_start = valid_end + pd.Timedelta(days=1)

    logger.info(
        "build_time_split_config: train_end=%s, valid_end=%s, test_start=%s",
        train_end.date(), valid_end.date(), test_start.date(),
    )

    return TimeSplitConfig(train_end=train_end, valid_end=valid_end, test_start=test_start)


def save_time_split_config(config: TimeSplitConfig, path: Path) -> pd.DataFrame:
    """Save time split configuration to CSV for reproducibility."""
    frame = pd.DataFrame([
        {
            "train_end": config.train_end,
            "valid_end": config.valid_end,
            "test_start": config.test_start,
        }
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def apply_time_split(
    df: pd.DataFrame,
    time_column: str,
    config: TimeSplitConfig,
) -> pd.DataFrame:
    """Assign ``time_split`` labels (train/valid/test) based on a time column."""
    out = df.copy()
    if time_column not in out.columns:
        raise ValueError(f"Expected {time_column} in DataFrame")

    out[time_column] = pd.to_datetime(out[time_column], errors="coerce")
    out = out[out[time_column].notna()].copy()

    out["time_split"] = "test"
    out.loc[out[time_column] <= config.valid_end, "time_split"] = "valid"
    out.loc[out[time_column] <= config.train_end, "time_split"] = "train"
    return out


def validate_processed_data(df: pd.DataFrame, table_name: str = "") -> None:
    """Raise ValueError if mentee_id or mentor_id contain nulls."""
    mentee_nulls = df["mentee_id"].isna().sum()
    mentor_nulls = df["mentor_id"].isna().sum()
    if mentee_nulls > 0 or mentor_nulls > 0:
        raise ValueError(
            f"validate_processed_data [{table_name}]: "
            f"mentee_id nulls={mentee_nulls}, mentor_id nulls={mentor_nulls}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback Processing Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _coerce_id_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce columns to numeric, drop rows with nulls in those cols, cast to int."""
    out = df.assign(**{c: pd.to_numeric(df[c], errors="coerce") for c in cols if c in df.columns})
    out = out.dropna(subset=[c for c in cols if c in out.columns])
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(int)
    return out


def _prepare_mentorship_links(mentorships: pd.DataFrame) -> pd.DataFrame:
    """Prepare deduplicated mentorship reference table for feedback joins."""
    required = ["mentorship_id", "mentee_id", "mentor_id", "start_date"]
    links = mentorships[required + ["status"]].dropna(subset=required)
    links = _coerce_id_columns(links, ["mentorship_id", "mentee_id", "mentor_id"])
    return links.sort_values("start_date").drop_duplicates(subset=["mentorship_id"], keep="first")


def _prepare_cancellation_lookup(cancel_df: pd.DataFrame) -> pd.DataFrame | None:
    """Prepare deduplicated cancellation lookup table."""
    if not {"mentorship_id", "cancellation_actor"}.issubset(cancel_df.columns):
        return None
    out = cancel_df.copy()
    if "cancellation_date" in out.columns:
        out["cancellation_date"] = pd.to_datetime(out["cancellation_date"], errors="coerce")
    out["mentorship_id"] = pd.to_numeric(out["mentorship_id"], errors="coerce")
    out = out.dropna(subset=["mentorship_id"]).assign(
        mentorship_id=lambda df: df["mentorship_id"].astype(int)
    )
    sort_col = "cancellation_date" if "cancellation_date" in out.columns else "mentorship_id"
    out = out.sort_values(sort_col).drop_duplicates(subset=["mentorship_id"], keep="last")
    cols = ["mentorship_id", "cancellation_actor"]
    if "cancellation_date" in out.columns:
        cols.append("cancellation_date")
    return out[cols]


def _apply_feedback_business_rules(
    feedback: pd.DataFrame, train_max_time: pd.Timestamp,
) -> pd.DataFrame:
    """Apply business rules to determine which feedback rows are valid.

    Business rules (PRESERVED — do not change):
      - Feedback is valid for accepted/ongoing/completed mentorships.
      - Feedback is also valid for cancelled mentorships IF the mentor was
        the cancellation actor (mentee-written feedback is still meaningful).
      - Feedback must reference a mentorship that started before train_end.
      - Cancellation date (if present) must also be before train_end.
    """
    status_series = (
        feedback.get("mentorship_status", feedback.get("status", ""))
        .astype(str).str.strip().str.lower()
    )
    actor_series = (
        feedback.get("cancellation_actor", "")
        .astype(str).str.strip().str.lower()
    )

    # Backend MentorshipStatus enum (HasConversion<string>): Active, Completed, Cancelled
    # After .str.lower(): "active", "completed", "cancelled"
    _ACCEPTED_STATUSES = {"active", "completed"}
    _CANCELLED_STATUSES = {"cancelled", "canceled"}

    # CancellationActor is stored as INT (no HasConversion): Mentor=1, Mentee=2
    # Map INT values to string names for uniform comparison
    actor_series = actor_series.replace({"1": "mentor", "2": "mentee"})

    valid_mask = status_series.isin(_ACCEPTED_STATUSES) | (
        status_series.isin(_CANCELLED_STATUSES) & (actor_series == "mentor")
    )

    # Time leakage prevention: only include feedback from mentorships
    # that started within the training window.
    feedback = feedback[
        valid_mask
        & feedback["start_date"].notna()
        & (feedback["start_date"] <= train_max_time)
    ].copy()

    if "cancellation_date" in feedback.columns:
        feedback = feedback[
            feedback["cancellation_date"].isna()
            | (feedback["cancellation_date"] <= train_max_time)
        ].copy()

    return feedback


def _process_feedback_with_mentorships(
    feedback: pd.DataFrame,
    processed: Dict[str, pd.DataFrame],
    train_max_time: pd.Timestamp,
) -> pd.DataFrame:
    """Join feedback with mentorships, apply business rules, and time-filter.

    This is the full feedback processing pipeline extracted from
    ``prepare_processed_tables`` for readability. All business logic is
    preserved exactly as before.
    """
    if "mentorships" not in processed or "mentorship_id" not in feedback.columns:
        return feedback

    mentorship_links = processed["mentorships"]
    if not {"mentorship_id", "mentee_id", "mentor_id", "start_date"}.issubset(mentorship_links.columns):
        return feedback

    # Step 1: Prepare reference tables
    mentorship_links = _prepare_mentorship_links(mentorship_links)
    feedback = _coerce_id_columns(feedback, ["mentorship_id", "mentee_id", "mentor_id"])

    # Step 2: Join feedback with mentorship data
    feedback = feedback.merge(
        mentorship_links, on="mentorship_id", how="inner", suffixes=("", "_mentorship"),
    )

    # Step 3: Verify ID consistency (feedback IDs must match mentorship IDs)
    id_match = (
        (feedback["mentee_id"] == feedback["mentee_id_mentorship"])
        & (feedback["mentor_id"] == feedback["mentor_id_mentorship"])
    )
    feedback = feedback[id_match].copy()

    # Step 4: Merge cancellation data (if available)
    if "mentorship_cancellation" in processed:
        cancel_lookup = _prepare_cancellation_lookup(processed["mentorship_cancellation"])
        if cancel_lookup is not None:
            feedback = feedback.merge(
                cancel_lookup, on="mentorship_id", how="left", suffixes=("", "_cancel"),
            )

    # Step 5: Apply business rules + time filtering
    feedback = _apply_feedback_business_rules(feedback, train_max_time)

    # Step 6: Use mentorship-verified IDs and clean up join artifacts
    feedback["mentee_id"] = feedback["mentee_id_mentorship"].astype(int)
    feedback["mentor_id"] = feedback["mentor_id_mentorship"].astype(int)
    feedback = feedback.drop(
        columns=["mentee_id_mentorship", "mentor_id_mentorship", "status", "cancellation_date"],
        errors="ignore",
    )

    return feedback


# ═══════════════════════════════════════════════════════════════════════════════
# Main Processing Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_processed_tables(
    raw_tables: Dict[str, pd.DataFrame],
    config: TimeSplitConfig,
) -> Dict[str, pd.DataFrame]:
    """Process raw tables: clean, normalize, deduplicate, and time-filter.

    Processing stages:
      1. **Cleaning** — strip whitespace, enforce ID types, handle missing values.
      2. **Normalization** — experience levels, column renames.
      3. **Deduplication** — table-specific business-key dedup.
      4. **Temporal filtering** — prevent data leakage by filtering to train_end.
      5. **Label tagging** — assign time_split to applications and mentorships.

    Special handling:
    - ``mentors_feedback``: custom filtering (joined with mentorships).
    - ``mentorship_applications``: NOT time-filtered — receives a time_split
      column instead, because it serves as the label source (label=1 if user
      applied). Train-only filtering for features happens in pipeline.py.

    SCD Type 1 tables (profiles, subdomains, interests, expertise) are NOT
    time-filtered — see SCD Type 1 Policy docs above.
    """
    processed = {name: df.copy() for name, df in raw_tables.items()}
    train_max_time = config.train_end

    # ── Stage 1: Per-table cleaning, normalization, deduplication ──
    for name, df in list(processed.items()):
        cleaned = _strip_text_columns(df)
        cleaned = _enforce_ids(cleaned)
        cleaned = _drop_invalid_ids(cleaned, table_name=name)
        cleaned = _replace_empty_strings_with_na(cleaned, IGNORE_NULL_COLUMNS.get(name, []))
        cleaned = _normalize_experience_columns(cleaned, EXPERIENCE_COLUMNS_BY_DATASET.get(name, []))
        cleaned = _drop_duplicates(name, cleaned)
        cleaned = _handle_missing(cleaned, ignored_object_cols=IGNORE_NULL_COLUMNS.get(name, []))
        processed[name] = cleaned

    # Normalize column names: shared_posts uses sender_id instead of user_id
    if "shared_posts_dataset" in processed and "sender_id" in processed["shared_posts_dataset"].columns:
        processed["shared_posts_dataset"] = processed["shared_posts_dataset"].rename(columns={"sender_id": "user_id"})

    # ── Stage 2: Parse datetime columns ──
    for table_name, time_col in TIME_TABLES.items():
        if table_name in processed and time_col in processed[table_name].columns:
            processed[table_name] = parse_datetime_column(processed[table_name], time_col)

    if "mentorships" in processed and "start_date" in processed["mentorships"].columns:
        processed["mentorships"] = parse_datetime_column(processed["mentorships"], "start_date")

    # ── Stage 3: Temporal filtering (leakage prevention) ──
    # mentors_feedback: custom handling below.
    # mentorship_applications: label source — gets time_split, not time filter.
    for table_name, time_col in TIME_TABLES.items():
        if table_name in ("mentors_feedback", "mentorship_applications"):
            continue
        if table_name in processed:
            before = len(processed[table_name])
            processed[table_name] = filter_by_time(processed[table_name], time_col, train_max_time)
            after = len(processed[table_name])
            if before != after:
                logger.info(
                    "prepare_processed_tables: %s — removed %d rows after train_end",
                    table_name, before - after,
                )
            if after == 0:
                logger.warning(
                    "prepare_processed_tables: %s is EMPTY after time filtering "
                    "(had %d rows before) — check train_end=%s and timestamps.",
                    table_name, before, train_max_time,
                )

    # ── Stage 4: Feedback processing (business rules + temporal filtering) ──
    if "mentors_feedback" in processed:
        feedback = processed["mentors_feedback"]
        feedback = _process_feedback_with_mentorships(feedback, processed, train_max_time)
        if "start_date" in feedback.columns:
            feedback = feedback.drop(columns=["start_date"], errors="ignore")
        processed["mentors_feedback"] = _validate_feedback_table(feedback)

    # ── Stage 5: Assign temporal split labels ──
    if "mentorships" in processed:
        processed["mentorships"]["time_split"] = "train"
        if "start_date" in processed["mentorships"].columns:
            split_frame = apply_time_split(processed["mentorships"], "start_date", config)
            processed["mentorships"] = split_frame

    # Apply time_split to applications (label source: train/valid/test)
    # This preserves ALL applications but tags each one with its temporal split.
    # Pipeline.py will filter to train-only for feature building.
    if "mentorship_applications" in processed:
        if "applied_at" in processed["mentorship_applications"].columns:
            processed["mentorship_applications"] = apply_time_split(
                processed["mentorship_applications"], "applied_at", config
            )
            split_counts = processed["mentorship_applications"]["time_split"].value_counts().to_dict()
            logger.info(
                "prepare_processed_tables: mentorship_applications time_split = %s",
                split_counts,
            )

    # ── Final validation ──
    for table_name, df in processed.items():
        if {"mentee_id", "mentor_id"}.issubset(df.columns):
            validate_processed_data(df, table_name=table_name)

    return processed
