from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# Canonical values aligned with .NET ExperienceLevel enum
EXPERIENCE_LEVEL_MAP = {
    "none": 1,
    "beginner": 2,
    "intermediate": 3,
    "advanced": 4,
}
# Reverse map returns canonical string values
EXPERIENCE_LEVEL_MAP_REVERSE = {1: "none", 2: "beginner", 3: "intermediate", 4: "advanced"}
_VALID_EXPERIENCE_VALUES = set(EXPERIENCE_LEVEL_MAP.keys())

# Binary features that must NOT be scaled — used by pipeline.py to exclude from MinMaxScaler
BINARY_FEATURE_COLS = {"same_country", "mentor_more_experienced", "mentor_domain_match", "mentor_covers_all_skills"}

MENTEE_FEATURE_ARTIFACT_COLS = [
    "mentee_id",
    "experience_level",
    "experience_level_num",
    "education_status",
    "country_code",
    "domain_id",
    "subdomains_set",
    "interests_set",
]

MENTOR_FEATURE_ARTIFACT_COLS = [
    "mentor_id",
    "experience_level_num",
    "country_code",
    "domain_id",
    "subdomains_set",
    "expertise_set",
    "mentor_avg_rating",
    "mentor_review_count",
    "mentor_sentiment_score",
    "mentor_positive_feedback_ratio",
    "mentor_weighted_rating",
    "mentor_quality_score",
    "mentor_completion_rate",
    "mentor_cancel_rate",
    "mentor_reliability_score",
    "mentor_program_popularity",
    "mentor_follower_count_log",
    "mentor_open_post_count_log",
]

# Publication state: ProgramPostStatus stores whether a program is visible.
# EF-native values: "draft", "published". Legacy may also contain other values.
PUBLISHED_PROGRAM_STATUSES = {"published"}

# Availability state: Availability stores whether a program is open for applications.
# Values: "open", "opened", "closed" (string, case-insensitive).
OPEN_AVAILABILITY_STATUSES = {"open", "opened"}


def normalize_encode_experience_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Normalize experience values to canonical forms and encode as integers.

    Handles missing values and invalid strings by mapping them to ``none`` (1).
    Aligned with .NET ExperienceLevel enum: None=1, Beginner=2, Intermediate=3, Advanced=4

    Returns:
        Tuple of (normalized_labels, numeric_codes).
    """
    normalized = series.astype(str).str.strip().str.lower()
    normalized = normalized.mask(series.isna(), "none")
    normalized = normalized.replace(
        ["", "no_experience", "null", "nan", "n/a", "na"], "none"
    )

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_alias_map = {
        0: "none",
        1: "none",
        2: "beginner",
        3: "intermediate",
        4: "advanced",
    }
    numeric_labels = numeric.round().astype("Int64").map(numeric_alias_map)

    normalized = normalized.where(
        normalized.isin(_VALID_EXPERIENCE_VALUES),
        numeric_labels.fillna("none"),
    )
    encoded = normalized.map(EXPERIENCE_LEVEL_MAP).fillna(1).astype(int)
    return normalized, encoded


def as_safe_set(value) -> set:
    """Convert a value to a set, returning empty set for non-set inputs."""
    return value if isinstance(value, set) else set()


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets. Returns 0.0 if either is empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def skill_coverage(a: set, b: set) -> float:
    """Fraction of mentee skills (a) covered by mentor skills (b)."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


# ---------------------------------------------------------------------------
# Mentee features
# ---------------------------------------------------------------------------

def build_mentee_features(
    mentee_profile: pd.DataFrame,
    mentee_subdomains: pd.DataFrame,
    mentee_interests: pd.DataFrame,
) -> pd.DataFrame:
    """Build mentee feature table from profile, subdomains, and interests.

    Experience source: ``mentee_profile.current_level`` ONLY.

    Design decision: skill matching (interests/technologies) is decoupled from
    experience modeling.  ``interests.experience_level`` describes per-skill
    proficiency and is intentionally NOT aggregated into overall experience.
    Overall experience drives ``experience_gap`` and ``mentor_more_experienced``
    features in ``build_pair_features``.

    Performance note: groupby-agg lambdas run once per group (per user), not
    per row. Acceptable for current scale; revisit only if n_groups > 500k.
    """
    base = mentee_profile.rename(columns={"user_id": "mentee_id"}).copy()

    current_level_series = (
        base["current_level"]
        if "current_level" in base.columns
        else pd.Series([np.nan] * len(base), index=base.index)
    )
    profile_exp_norm, profile_exp_num = normalize_encode_experience_series(current_level_series)
    base["experience_level"] = profile_exp_norm
    base["experience_level_num"] = profile_exp_num

    education_series = (
        base["education_status"]
        if "education_status" in base.columns
        else pd.Series(["unknown"] * len(base), index=base.index)
    )
    base["education_status"] = education_series.astype(str).str.strip().str.lower()

    country_series = (
        base["country_code"]
        if "country_code" in base.columns
        else pd.Series([""] * len(base), index=base.index)
    )
    base["country_code"] = country_series.astype(str).str.strip().str.lower()

    sub_vec = (
        mentee_subdomains.groupby("mentee_id")["subdomain_id"]
        .agg(lambda x: set(x.dropna().astype(int)))
        .reset_index(name="subdomains_set")
    )
    # Skills come from interests (technology_id only — no experience mixing)
    int_vec = (
        mentee_interests.groupby("mentee_id")["technology_id"]
        .agg(lambda x: set(x.dropna().astype(int)))
        .reset_index(name="interests_set")
    )

    features = (
        base.merge(sub_vec, on="mentee_id", how="left")
        .merge(int_vec, on="mentee_id", how="left")
    )
    features["experience_level_num"] = (
        pd.to_numeric(features["experience_level_num"], errors="coerce").fillna(1).astype(int)
    )
    features["experience_level"] = (
        features["experience_level_num"].map(EXPERIENCE_LEVEL_MAP_REVERSE).fillna("none")
    )
    features["subdomains_set"] = features["subdomains_set"].apply(as_safe_set)
    features["interests_set"] = features["interests_set"].apply(as_safe_set)
    return features[[c for c in MENTEE_FEATURE_ARTIFACT_COLS if c in features.columns]]


# ---------------------------------------------------------------------------
# Mentor quality features (rating + sentiment + feedback)
# ---------------------------------------------------------------------------

def build_mentor_quality_features(feedback_hist: pd.DataFrame) -> pd.DataFrame:
    """Build mentor quality features from historical feedback.

    The input ``feedback_hist`` MUST be pre-filtered to ``<= train_end``.
    This function does NOT perform time filtering itself.

    Produces a unified ``mentor_quality_score`` that combines:
      - 60% Bayesian-smoothed weighted rating
      - 25% mean sentiment score
      - 15% positive feedback ratio

    Ratings outside 1-5 are treated as invalid and excluded.
    """
    feedback = feedback_hist.copy()
    feedback["rating"] = pd.to_numeric(feedback.get("rating"), errors="coerce")

    # Exclude ratings outside valid range (1-5) — bad data that skews averages
    valid_rating_mask = feedback["rating"].between(1, 5, inclusive="both")
    n_invalid = (~valid_rating_mask & feedback["rating"].notna()).sum()
    if n_invalid > 0:
        logger.warning("build_mentor_quality_features: %d ratings outside 1-5 range — ignored", n_invalid)
    feedback.loc[~valid_rating_mask, "rating"] = np.nan

    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    feedback["sentiment_score"] = (
        feedback.get("sentiment", pd.Series(dtype=str))
        .astype(str)
        .str.strip()
        .str.lower()
        .map(sentiment_map)
        .fillna(0)
    )

    mentor_quality = (
        feedback.groupby("mentor_id", as_index=False)
        .agg(
            mentor_avg_rating=("rating", "mean"),
            mentor_review_count=("mentor_id", "size"),
            mentor_sentiment_score=("sentiment_score", "mean"),
            mentor_positive_feedback_ratio=("sentiment_score", lambda s: (s > 0).mean()),
        )
        .fillna(
            {
                "mentor_avg_rating": 0,
                "mentor_sentiment_score": 0,
                "mentor_positive_feedback_ratio": 0,
            }
        )
    )

    # Bayesian smoothing: pull low-evidence mentors toward global average
    k = 5
    global_avg = mentor_quality["mentor_avg_rating"].mean() if len(mentor_quality) else 0
    mentor_quality["mentor_weighted_rating"] = (
        (mentor_quality["mentor_avg_rating"] * mentor_quality["mentor_review_count"] + global_avg * k)
        / (mentor_quality["mentor_review_count"] + k)
    )

    # Unified mentor quality score (business spec):
    # Combines rating, sentiment, and positive feedback into a single signal.
    # Sentiment is used ONLY here — never in candidate generation or interaction features.
    mentor_quality["mentor_quality_score"] = (
        0.60 * mentor_quality["mentor_weighted_rating"]
        + 0.25 * mentor_quality["mentor_sentiment_score"]
        + 0.15 * mentor_quality["mentor_positive_feedback_ratio"]
    )

    return mentor_quality


# ---------------------------------------------------------------------------
# Mentor reliability features (completion + cancellation rates)
# ---------------------------------------------------------------------------

def build_mentor_reliability_features(
    apps_hist: pd.DataFrame,
    posts_hist: pd.DataFrame,
    cancellations_hist: pd.DataFrame,
) -> pd.DataFrame:
    """Build mentor reliability features from applications and cancellations.

    All inputs MUST be pre-filtered to ``<= train_end``.
    Only cancellations by the *mentor* (not mentee) count against reliability.
    """
    posts = posts_hist[["post_id", "mentor_id"]].drop_duplicates()
    apps = apps_hist.merge(posts, on="post_id", how="left").dropna(subset=["mentor_id"]).copy()
    apps["mentor_id"] = apps["mentor_id"].astype(int)
    # Backend ApplicationStatus uses HasConversion<string>: "Pending", "Accepted", "Rejected"
    # Compare case-insensitively to handle both DB and CSV data sources
    apps["is_completed"] = (
        apps.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.lower() == "accepted"
    ).astype(int)

    mentor_completed = apps.groupby("mentor_id", as_index=False).agg(
        total_apps=("app_id", "nunique"),
        completed=("is_completed", "sum"),
    )

    # Normalize cancellation_actor before filtering to avoid case-sensitivity misses
    # Backend CancellationActor enum: Mentor=1, Mentee=2 (stored as INT, no HasConversion<string>)
    # Also handle legacy string values from CSV data
    cancellations = cancellations_hist.copy()
    if "cancellation_actor" in cancellations.columns:
        actor_raw = cancellations["cancellation_actor"].astype(str).str.strip().str.lower()
        # Map INT enum values to lowercase names for uniform comparison
        actor_raw = actor_raw.replace({"1": "mentor", "2": "mentee"})
        cancellations["cancellation_actor"] = actor_raw

    mentor_cancel_counts = (
        cancellations.query("cancellation_actor == 'mentor'")
        .groupby("mentor_id", as_index=False)
        .agg(mentor_real_cancels=("mentorship_id", "count"))
    )

    reliability = (
        mentor_completed.merge(mentor_cancel_counts, on="mentor_id", how="left")
        .fillna({"mentor_real_cancels": 0})
    )
    reliability["total_apps"] = reliability["total_apps"].replace(0, 1)
    reliability["mentor_completion_rate"] = reliability["completed"] / reliability["total_apps"]
    reliability["mentor_cancel_rate"] = reliability["mentor_real_cancels"] / reliability["total_apps"]
    reliability["mentor_reliability_score"] = (
        0.7 * reliability["mentor_completion_rate"] - 0.3 * reliability["mentor_cancel_rate"]
    )
    return reliability[
        ["mentor_id", "mentor_completion_rate", "mentor_cancel_rate", "mentor_reliability_score"]
    ]


# ---------------------------------------------------------------------------
# Mentor popularity features (programs + enrollments + engagement)
# ---------------------------------------------------------------------------

def build_mentor_popularity_features(
    posts_hist: pd.DataFrame,
    mentorships_hist: pd.DataFrame,
    apps_hist: pd.DataFrame,
    likes_hist: pd.DataFrame | None = None,
    comments_hist: pd.DataFrame | None = None,
    saves_hist: pd.DataFrame | None = None,
    shares_hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build mentor popularity features from programs, enrollments, and engagement.

    All inputs MUST be pre-filtered to ``<= train_end``.

    Engagement counts are computed from the time-filtered interaction tables
    (likes, comments, saves, shares) instead of using DB-level counters on
    ``mentorship_posts`` (which would leak future data).

    ⚠️ This feature EXCLUDES applications to prevent label leakage.
    Since label = 1 if user applied, including application counts in
    popularity would leak the target variable.
    """
    mentor_program_count = (
        posts_hist.groupby("mentor_id")["post_id"].nunique().reset_index(name="mentor_program_count")
    )
    mentor_enrollment_count = (
        mentorships_hist.groupby("mentor_id")["mentorship_id"]
        .nunique()
        .reset_index(name="mentor_enrollment_count")
    )
    popularity = (
        mentor_program_count.merge(mentor_enrollment_count, on="mentor_id", how="outer")
        .fillna(0)
    )

    # Compute engagement counts from time-filtered interaction tables
    post_to_mentor = posts_hist[["post_id", "mentor_id"]].drop_duplicates()

    def _count_interactions(table: pd.DataFrame | None, user_col: str = "user_id") -> pd.Series | None:
        if table is None or table.empty:
            return
        merged = table[[user_col, "post_id"]].drop_duplicates().merge(post_to_mentor, on="post_id", how="inner")
        return merged.groupby("mentor_id").size()

    likes_count = _count_interactions(likes_hist)
    comments_count = _count_interactions(comments_hist)
    saves_count = _count_interactions(saves_hist)
    shares_count = _count_interactions(shares_hist)

    engagement_parts = [s for s in [likes_count, comments_count, saves_count, shares_count] if s is not None]
    if engagement_parts:
        mentor_engagement = sum(engagement_parts).reset_index()
        mentor_engagement.columns = ["mentor_id", "mentor_engagement_count"]
        mentor_engagement["mentor_engagement_log"] = np.log1p(mentor_engagement["mentor_engagement_count"])
        popularity = popularity.merge(mentor_engagement, on="mentor_id", how="left").fillna(
            {"mentor_engagement_count": 0, "mentor_engagement_log": 0}
        )
        engagement_weight = 0.15
    else:
        logger.info(
            "build_mentor_popularity_features: interaction tables unavailable — "
            "mentor_program_popularity will use enrollments/applications only."
        )
        popularity["mentor_engagement_log"] = 0.0
        engagement_weight = 0.0

    for column in ["mentor_program_count", "mentor_enrollment_count"]:
        popularity[f"{column}_log"] = np.log1p(popularity[column])

    # Weights: enrollments (strongest clean signal), programs, engagement
    # ⚠️ application_count REMOVED to prevent label leakage
    enrollment_w = 0.55 if engagement_weight > 0 else 0.60
    program_w = 0.30 if engagement_weight > 0 else 0.40

    raw_popularity = (
        enrollment_w * popularity["mentor_enrollment_count_log"]
        + program_w * popularity["mentor_program_count_log"]
        + engagement_weight * popularity["mentor_engagement_log"]
    )

    # Cap at 95th percentile to prevent mega-popular mentors from dominating.
    # This preserves the ranking within the top tier while reducing the gap
    # between popular and less-popular-but-relevant mentors.
    cap = raw_popularity.quantile(0.95) if len(raw_popularity) > 10 else raw_popularity.max()
    if cap > 0:
        popularity["mentor_program_popularity"] = raw_popularity.clip(upper=cap)
    else:
        popularity["mentor_program_popularity"] = raw_popularity

    logger.info(
        "build_mentor_popularity_features: %d mentors, popularity range=[%.2f, %.2f], "
        "cap@95pct=%.2f",
        len(popularity), raw_popularity.min(), raw_popularity.max(), cap if cap > 0 else 0,
    )
    return popularity


# ---------------------------------------------------------------------------
# Mentor social features (follower count + open post availability)
# ---------------------------------------------------------------------------

def build_mentor_social_features(
    follows_hist: pd.DataFrame,
    posts_hist: pd.DataFrame,
    mentor_profile: pd.DataFrame,
) -> pd.DataFrame:
    """Build mentor-level social proof and availability features.

    All inputs MUST be pre-filtered to ``<= train_end``.

    -- Leakage audit --
    mentor_follower_count_log : derived from follows_hist (filtered <= train_end).
                                Counts how many users follow a mentor.
                                Completely independent of applications.
    mentor_open_post_count_log: posts_hist is filtered <= train_end.
                                Counts programs that are BOTH published
                                (ProgramPostStatus == 'Published')
                                AND open for applications
                                (Availability IN ('Open', 'Opened')).
                                For legacy data without Availability column,
                                falls back to is_open as the availability proxy.
                                No application count included.
    """
    # --- mentor_follower_count_log ---
    follows = follows_hist.copy()
    follows["following_id"] = pd.to_numeric(follows["following_id"], errors="coerce")
    mentor_ids = set(mentor_profile["user_id"].dropna().astype(int).unique())
    mentor_follows = follows[follows["following_id"].isin(mentor_ids)].copy()
    follower_counts = (
        mentor_follows.groupby("following_id").size()
        .reset_index(name="mentor_follower_count")
        .rename(columns={"following_id": "mentor_id"})
    )
    follower_counts["mentor_follower_count_log"] = np.log1p(
        follower_counts["mentor_follower_count"]
    )

    # --- mentor_open_post_count_log ---
    # Semantic separation (May 2026):
    #   Publication state  → is_open  (from ProgramPostStatus: "draft"/"published")
    #   Availability state → availability (from Availability: "open"/"opened"/"closed")
    # A program counts as "open" only if it is published AND available.
    posts = posts_hist.copy()
    has_availability = "availability" in posts.columns
    has_is_open = "is_open" in posts.columns

    if has_availability:
        # Modern EF-native data: separate publication and availability columns
        if has_is_open:
            published_posts = posts[
                posts["is_open"].astype(str).str.lower().str.strip().isin(PUBLISHED_PROGRAM_STATUSES)
            ]
        else:
            published_posts = posts
        open_posts = published_posts[
            published_posts["availability"].astype(str).str.lower().str.strip().isin(OPEN_AVAILABILITY_STATUSES)
        ]
    elif has_is_open:
        # Legacy fallback: Availability column absent.
        # In legacy data, is_open (from ProgramPostStatus) was used for both
        # publication and availability. We treat "open"/"opened" as available.
        open_posts = posts[
            posts["is_open"].astype(str).str.lower().str.strip().isin({"open", "opened"})
        ]
    else:
        open_posts = posts

    mentor_open = (
        open_posts.groupby("mentor_id", as_index=False)["post_id"]
        .nunique()
        .rename(columns={"post_id": "mentor_open_post_count"})
    )
    mentor_open["mentor_open_post_count_log"] = np.log1p(
        mentor_open["mentor_open_post_count"]
    )

    # Build mentor-level result
    mentor_social = (
        mentor_profile[["user_id"]].rename(columns={"user_id": "mentor_id"}).copy()
    )
    mentor_social["mentor_id"] = pd.to_numeric(mentor_social["mentor_id"], errors="coerce")
    mentor_social = (
        mentor_social
        .merge(follower_counts[["mentor_id", "mentor_follower_count_log"]], on="mentor_id", how="left")
        .merge(mentor_open[["mentor_id", "mentor_open_post_count_log"]], on="mentor_id", how="left")
        .fillna({"mentor_follower_count_log": 0.0, "mentor_open_post_count_log": 0.0})
    )
    return mentor_social


# ---------------------------------------------------------------------------
# Consolidated mentor features
# ---------------------------------------------------------------------------

def build_mentor_features(
    mentor_profile: pd.DataFrame,
    mentor_subdomains: pd.DataFrame,
    mentor_expertise: pd.DataFrame,
    feedback_hist: pd.DataFrame,
    apps_hist: pd.DataFrame,
    posts_hist: pd.DataFrame,
    cancellations_hist: pd.DataFrame,
    mentorships_hist: pd.DataFrame,
    follows_hist: pd.DataFrame,
    train_end: pd.Timestamp | None = None,
    likes_hist: pd.DataFrame | None = None,
    comments_hist: pd.DataFrame | None = None,
    saves_hist: pd.DataFrame | None = None,
    shares_hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build consolidated mentor feature table.

    Merges profile data with quality, reliability, and popularity features.
    All temporal inputs (feedback, apps, posts, cancellations, mentorships,
    follows, interaction tables) MUST be pre-filtered to ``<= train_end``
    by the caller.  This function validates but does not filter.

    ``mentor_expertise`` is treated as SCD Type 1 (latest snapshot) and is
    intentionally NOT time-filtered.
    """
    # Data leakage guard: warn if any temporal table contains future data
    if train_end is not None:
        for table_name, table, time_col in [
            ("feedback_hist", feedback_hist, "created_at"),
            ("apps_hist", apps_hist, "applied_at"),
            ("posts_hist", posts_hist, "created_at"),
        ]:
            if time_col in table.columns:
                future_rows = (
                    pd.to_datetime(table[time_col], errors="coerce") > train_end
                ).sum()
                if future_rows > 0:
                    logger.warning(
                        "build_mentor_features: %s has %d rows after train_end — potential data leakage!",
                        table_name,
                        future_rows,
                    )

    base = mentor_profile.rename(columns={"user_id": "mentor_id"}).copy()
    years = pd.to_numeric(
        base.get("years_of_experience", pd.Series([np.nan] * len(base), index=base.index)),
        errors="coerce",
    )
    years_exp_num = pd.cut(
        years.fillna(0),
        bins=[-1, 2, 5, 10, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype(int)
    base["experience_level_num"] = years_exp_num
    base["country_code"] = base.get("country_code", "").astype(str).str.strip().str.lower()

    sub_vec = (
        mentor_subdomains.groupby("mentor_id")["subdomain_id"]
        .agg(lambda x: set(x.dropna().astype(int)))
        .reset_index(name="subdomains_set")
    )
    exp_vec = (
        mentor_expertise.groupby("mentor_id")["technology_id"]
        .agg(lambda x: set(x.dropna().astype(int)))
        .reset_index(name="expertise_set")
    )

    features = (
        base.merge(sub_vec, on="mentor_id", how="left")
        .merge(exp_vec, on="mentor_id", how="left")
    )
    features["experience_level_num"] = pd.to_numeric(
        features["experience_level_num"], errors="coerce"
    ).fillna(0).astype(int)
    features["subdomains_set"] = features["subdomains_set"].apply(as_safe_set)
    features["expertise_set"] = features["expertise_set"].apply(as_safe_set)

    mentor_quality = build_mentor_quality_features(feedback_hist)
    mentor_reliability = build_mentor_reliability_features(apps_hist, posts_hist, cancellations_hist)
    mentor_popularity = build_mentor_popularity_features(
        posts_hist, mentorships_hist, apps_hist,
        likes_hist=likes_hist,
        comments_hist=comments_hist,
        saves_hist=saves_hist,
        shares_hist=shares_hist,
    )
    mentor_social = build_mentor_social_features(
        follows_hist, posts_hist, mentor_profile,
    )

    features = (
        features.merge(mentor_quality, on="mentor_id", how="left")
        .merge(mentor_reliability, on="mentor_id", how="left")
        .merge(mentor_popularity[["mentor_id", "mentor_program_popularity"]], on="mentor_id", how="left")
        .merge(mentor_social[["mentor_id", "mentor_follower_count_log", "mentor_open_post_count_log"]], on="mentor_id", how="left")
    )

    features = features.fillna(
        {
            "mentor_avg_rating": 0,
            "mentor_review_count": 0,
            "mentor_sentiment_score": 0,
            "mentor_positive_feedback_ratio": 0,
            "mentor_weighted_rating": 0,
            "mentor_quality_score": 0,
            "mentor_completion_rate": 0,
            "mentor_cancel_rate": 0,
            "mentor_reliability_score": 0,
            "mentor_program_popularity": 0,
            "mentor_follower_count_log": 0,
            "mentor_open_post_count_log": 0,
        }
    ).assign(
        experience_level_num=lambda df: pd.to_numeric(
            df["experience_level_num"], errors="coerce"
        ).fillna(0).astype(int)
    )
    return features[[c for c in MENTOR_FEATURE_ARTIFACT_COLS if c in features.columns]]


# ---------------------------------------------------------------------------
# Interaction features (likes, comments, saves, shares)
# ---------------------------------------------------------------------------

def build_interaction_features(
    likes_hist: pd.DataFrame,
    comments_hist: pd.DataFrame,
    saves_hist: pd.DataFrame,
    shares_hist: pd.DataFrame,
    follows_hist: pd.DataFrame,
    posts_hist: pd.DataFrame,
) -> pd.DataFrame:
    """Build user→mentor interaction features from engagement history.

    All inputs MUST be pre-filtered to ``<= train_end``.

    Follow signal is intentionally EXCLUDED from interaction_score to avoid
    double-counting — it is captured separately as the binary ``is_following``
    feature in ``build_pair_features``.
    """
    post_to_mentor = posts_hist[["post_id", "mentor_id"]].drop_duplicates()
    valid_mentor_ids = set(post_to_mentor["mentor_id"].dropna().astype(int).unique())

    def _map(table: pd.DataFrame, score: int, user_col: str = "user_id") -> pd.DataFrame:
        t = table.copy()
        # Filter to mentee actors only — mentor interactions on other posts
        # are not signals of mentorship intent.
        if "actor" in t.columns:
            t = t[t["actor"].astype(str).str.lower() == "mentee"]
        out = (
            t[[user_col, "post_id"]]
            .drop_duplicates()
            .merge(post_to_mentor, on="post_id", how="inner")
        )
        out["interaction_score"] = score
        return out[[user_col, "mentor_id", "interaction_score"]]

    likes_df    = _map(likes_hist, 1)
    comments_df = _map(comments_hist, 2)
    saves_df    = _map(saves_hist, 3)
    shares_df   = _map(shares_hist, 4)

    interaction_features = pd.concat(
        [likes_df, comments_df, saves_df, shares_df],
        ignore_index=True,
    ).dropna()

    interaction_features[["user_id", "mentor_id"]] = (
        interaction_features[["user_id", "mentor_id"]].astype(int)
    )
    interaction_features = interaction_features.groupby(
        ["user_id", "mentor_id"], as_index=False
    ).agg(
        interaction_score=("interaction_score", "sum"),
        interaction_count=("interaction_score", "count"),
    )
    interaction_features["interaction_score_log"] = np.log1p(interaction_features["interaction_score"])
    interaction_features["interaction_count_log"] = np.log1p(interaction_features["interaction_count"])
    return interaction_features


# ---------------------------------------------------------------------------
# Candidate pool generation (content-based)
# ---------------------------------------------------------------------------

def generate_candidate_pool(
    mentee_features: pd.DataFrame,
    mentor_features: pd.DataFrame,
    subdomains_map: pd.DataFrame | None = None,
    top_k: int = 30,
    min_candidates_per_mentee: int = 10,
    high_priority_cap: int = 30,
    low_priority_cap: int = 10,
    exploration_pct: float = 0.05,
) -> pd.DataFrame:
    """Generate candidate mentor pool using multi-tier matching.

    Tiers (in priority order):
      1. Skill match (technology_id intersection)
      2. Subdomain match (subdomain_id intersection)
      3. Domain match (via subdomains_map fallback, if available)
      4. Global top mentors (final fallback for cold start)

    All matching is vectorized — no apply() or Python loops on rows.

    Args:
        subdomains_map: DataFrame with [subdomain_id, domain_id] from subdomains.csv.
                        If None, domain fallback is disabled.
        high_priority_cap: Max candidates per mentee from top-priority tiers.
        low_priority_cap: Max candidates per mentee from lower-priority tiers.
    """
    mentee_exp = (
        mentee_features[["mentee_id", "experience_level_num"]]
        .rename(columns={"experience_level_num": "mentee_experience_level_num"})
        .copy()
    )
    mentor_exp = (
        mentor_features[["mentor_id", "experience_level_num"]]
        .rename(columns={"experience_level_num": "mentor_experience_level_num"})
        .copy()
    )

    def _add_experience_boost(df: pd.DataFrame) -> pd.DataFrame:
        df = (
            df.merge(mentee_exp, on="mentee_id", how="left")
            .merge(mentor_exp, on="mentor_id", how="left")
        )
        df["mentee_experience_level_num"] = (
            pd.to_numeric(df["mentee_experience_level_num"], errors="coerce").fillna(1)
        )
        df["mentor_experience_level_num"] = (
            pd.to_numeric(df["mentor_experience_level_num"], errors="coerce").fillna(1)
        )
        df["priority"] = df["priority"] + 2 * (
            df["mentor_experience_level_num"] >= df["mentee_experience_level_num"]
        ).astype(int)
        return df.drop(columns=["mentee_experience_level_num", "mentor_experience_level_num"])

    def _explode_and_merge(mentee_col: str, mentor_col: str, key_col: str, priority: int) -> pd.DataFrame:
        mentee_exp_col = (
            mentee_features[["mentee_id", mentee_col]]
            .explode(mentee_col)
            .rename(columns={mentee_col: key_col})
            .dropna(subset=[key_col])
        )
        mentor_exp_col = (
            mentor_features[["mentor_id", mentor_col]]
            .explode(mentor_col)
            .rename(columns={mentor_col: key_col})
            .dropna(subset=[key_col])
        )
        matches = (
            mentee_exp_col.merge(mentor_exp_col, on=key_col, how="inner")[["mentee_id", "mentor_id"]]
            .drop_duplicates()
        )
        matches["priority"] = priority
        return matches

    # 1. Skill matches (technology_id)
    skill_matches = _explode_and_merge("interests_set", "expertise_set", "skill", priority=5)

    # 2. Subdomain matches
    subdomain_matches = _explode_and_merge("subdomains_set", "subdomains_set", "subdomain", priority=4)

    # 3. Domain matches (fallback) — uses subdomains_map if available
    domain_matches = pd.DataFrame(columns=["mentee_id", "mentor_id", "priority"])
    if subdomains_map is not None and not subdomains_map.empty:
        sub_to_domain = subdomains_map[["subdomain_id", "domain_id"]].drop_duplicates()

        mentee_domains = (
            mentee_features[["mentee_id", "subdomains_set"]]
            .explode("subdomains_set")
            .rename(columns={"subdomains_set": "subdomain_id"})
            .dropna(subset=["subdomain_id"])
            .merge(sub_to_domain, on="subdomain_id", how="left")
            .dropna(subset=["domain_id"])[["mentee_id", "domain_id"]]
            .drop_duplicates()
        )
        mentor_domains = (
            mentor_features[["mentor_id", "subdomains_set"]]
            .explode("subdomains_set")
            .rename(columns={"subdomains_set": "subdomain_id"})
            .dropna(subset=["subdomain_id"])
            .merge(sub_to_domain, on="subdomain_id", how="left")
            .dropna(subset=["domain_id"])[["mentor_id", "domain_id"]]
            .drop_duplicates()
        )
        domain_matches = (
            mentee_domains.merge(mentor_domains, on="domain_id", how="inner")[["mentee_id", "mentor_id"]]
            .drop_duplicates()
        )
        domain_matches["priority"] = 3

    candidate_pool = pd.concat(
        [skill_matches, subdomain_matches, domain_matches], ignore_index=True
    ).drop_duplicates(["mentee_id", "mentor_id"])

    candidate_pool = _add_experience_boost(candidate_pool)

    # 4. Global top mentors (final fallback — small portion only)
    top_mentors = (
        mentor_features.nlargest(top_k, "mentor_program_popularity")["mentor_id"].tolist()
    )
    existing_df = candidate_pool[["mentee_id", "mentor_id"]].drop_duplicates()
    global_recs = (
        mentee_features[["mentee_id"]]
        .assign(mentor_id=[top_mentors] * len(mentee_features))
        .explode("mentor_id")
        .reset_index(drop=True)
    )
    global_recs["mentor_id"] = global_recs["mentor_id"].astype(
        mentee_features["mentee_id"].dtype
    )
    global_recs = global_recs.merge(
        existing_df, on=["mentee_id", "mentor_id"], how="left", indicator=True
    )
    global_recs = global_recs[global_recs["_merge"] == "left_only"].drop(columns="_merge").copy()
    global_recs["priority"] = 1
    global_recs = _add_experience_boost(global_recs)

    if candidate_pool.empty:
        # Cold start: all mentees fall back to global recommendations
        candidate_pool = global_recs[["mentee_id", "mentor_id", "priority"]].copy()
        logger.warning(
            "generate_candidate_pool: pool is empty — all users on global fallback"
        )
    else:
        candidate_pool = pd.concat(
            [candidate_pool[["mentee_id", "mentor_id", "priority"]],
             global_recs[["mentee_id", "mentor_id", "priority"]]],
            ignore_index=True,
        ).drop_duplicates(["mentee_id", "mentor_id"])

    # Diversity: sample from multiple priority tiers so the model sees varied
    # signal strengths during training
    high_priority = (
        candidate_pool
        .sort_values(["mentee_id", "priority"], ascending=[True, False])
        .groupby("mentee_id")
        .head(high_priority_cap)
    )
    low_priority = candidate_pool.drop(high_priority.index)
    low_sample = (
        low_priority
        .groupby("mentee_id")
        .head(low_priority_cap)
    )
    candidate_pool = pd.concat(
        [high_priority, low_sample], ignore_index=True
    ).drop_duplicates(["mentee_id", "mentor_id"]).reset_index(drop=True)

    # Exploration: add ~5% mentors per mentee for diversity.
    # Biased toward same-domain mentors (semantically relevant exploration)
    # with pure-random fallback for cold-start mentees.
    all_mentor_ids = mentor_features["mentor_id"].unique()
    mentee_existing = candidate_pool.groupby("mentee_id")["mentor_id"].apply(set).to_dict()
    explore_per_mentee = max(2, int(round(high_priority_cap * exploration_pct)))
    rng = np.random.RandomState(42)

    # Pre-build mentor domain lookup for biased exploration
    mentor_domain_lookup: dict[int, set] = {}
    if subdomains_map is not None and not subdomains_map.empty:
        sub_to_dom = subdomains_map[["subdomain_id", "domain_id"]].drop_duplicates()
        for _, row in mentor_features[["mentor_id", "subdomains_set"]].iterrows():
            subs = row["subdomains_set"] if isinstance(row["subdomains_set"], set) else set()
            domains = set()
            for s in subs:
                matched = sub_to_dom[sub_to_dom["subdomain_id"] == s]["domain_id"]
                domains.update(matched.values)
            if domains:
                mentor_domain_lookup[int(row["mentor_id"])] = domains

    explore_parts = []
    for mentee_id in mentee_features["mentee_id"].unique():
        current_mentors = mentee_existing.get(mentee_id, set())
        available = np.setdiff1d(all_mentor_ids, list(current_mentors))
        if len(available) == 0:
            continue
        n_sample = min(explore_per_mentee, len(available))

        # Bias: prefer same-domain mentors for exploration (70/30 split)
        mentee_row = mentee_features[mentee_features["mentee_id"] == mentee_id]
        mentee_subs = mentee_row["subdomains_set"].iloc[0] if len(mentee_row) > 0 else set()
        mentee_subs = mentee_subs if isinstance(mentee_subs, set) else set()

        if mentee_subs and mentor_domain_lookup:
            mentee_domains = set()
            if subdomains_map is not None:
                for s in mentee_subs:
                    matched = sub_to_dom[sub_to_dom["subdomain_id"] == s]["domain_id"]
                    mentee_domains.update(matched.values)
            same_domain_avail = [
                m for m in available
                if mentor_domain_lookup.get(int(m), set()) & mentee_domains
            ]
            if same_domain_avail:
                n_biased = min(int(n_sample * 0.7), len(same_domain_avail))
                n_random = n_sample - n_biased
                biased = rng.choice(same_domain_avail, size=n_biased, replace=False)
                remaining = np.setdiff1d(available, biased)
                random_part = rng.choice(remaining, size=min(n_random, len(remaining)), replace=False) if len(remaining) > 0 else np.array([])
                sampled = np.concatenate([biased, random_part])
            else:
                sampled = rng.choice(available, size=n_sample, replace=False)
        else:
            sampled = rng.choice(available, size=n_sample, replace=False)

        explore_parts.append(
            pd.DataFrame({"mentee_id": mentee_id, "mentor_id": sampled})
        )
    if explore_parts:
        explore_df = pd.concat(explore_parts, ignore_index=True)
        candidate_pool = pd.concat(
            [candidate_pool, explore_df], ignore_index=True
        ).drop_duplicates(["mentee_id", "mentor_id"]).reset_index(drop=True)
        logger.info(
            "generate_candidate_pool: added %d exploration pairs (~%d per mentee, domain-biased)",
            len(explore_df), explore_per_mentee,
        )

    # Ensure every mentee has at least some candidates (force global fallback)
    all_mentee_ids = set(mentee_features["mentee_id"].unique())
    mentees_with_candidates = set(candidate_pool["mentee_id"].unique())
    missing_mentees = all_mentee_ids - mentees_with_candidates
    if missing_mentees:
        logger.warning(
            "generate_candidate_pool: %d mentees had no candidates — forcing global fallback",
            len(missing_mentees),
        )
        missing_df = pd.DataFrame({"mentee_id": list(missing_mentees)})
        fallback_recs = (
            missing_df
            .assign(mentor_id=[top_mentors] * len(missing_df))
            .explode("mentor_id")
            .reset_index(drop=True)
        )
        candidate_pool = pd.concat(
            [candidate_pool[["mentee_id", "mentor_id"]], fallback_recs],
            ignore_index=True,
        ).drop_duplicates(["mentee_id", "mentor_id"])

    # Coverage sanity check
    final_coverage = candidate_pool["mentee_id"].nunique()
    total_mentees = mentee_features["mentee_id"].nunique()
    logger.info(
        "generate_candidate_pool: coverage = %d / %d mentees (%.1f%%)",
        final_coverage, total_mentees, 100 * final_coverage / max(total_mentees, 1),
    )
    if final_coverage < total_mentees:
        logger.error(
            "generate_candidate_pool: STILL %d mentees without candidates after forced fallback!",
            total_mentees - final_coverage,
        )

    return candidate_pool[["mentee_id", "mentor_id"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Collaborative filtering embeddings (SVD-based)
# ---------------------------------------------------------------------------

def build_cf_embeddings(
    interaction_features: pd.DataFrame,
    follows_hist: pd.DataFrame,
    mentorships_hist: pd.DataFrame,
    posts_hist: pd.DataFrame,
    mentor_ids_set: set | None = None,
    n_factors: int = 16,
) -> Dict:
    """Build collaborative filtering embeddings via TruncatedSVD.

    Constructs a user×mentor interaction matrix from multiple engagement
    signals and decomposes it into latent factors.  The resulting embeddings
    capture hidden preference patterns that content-based features miss.

    Signal sources (weighted by engagement strength):
      - Likes/comments/saves/shares (via interaction_features): 1-4
      - Mentorship enrollments: weight 5
      - Follow relationships: EXCLUDED (May 2026) — follows leak label bias
        into latent vectors, making CF a proxy for is_following.

    All inputs MUST be pre-filtered to ``<= train_end``.

    Returns:
        Dict with 'user_factors' (user_id -> ndarray) and
        'item_factors' (mentor_id -> ndarray).
    """
    signals = []

    # 1. Engagement interactions (user -> mentor interaction scores)
    if interaction_features is not None and not interaction_features.empty:
        score_col = (
            "interaction_score"
            if "interaction_score" in interaction_features.columns
            else "interaction_score_log"
        )
        int_df = interaction_features[["user_id", "mentor_id", score_col]].copy()
        int_df.columns = ["user_id", "mentor_id", "score"]
        signals.append(int_df)

    # 2. Follow signals — EXCLUDED (May 2026).
    # Reason: follow relationships are highly correlated with positive labels
    # (users tend to apply to mentors they follow). Including follows in CF
    # embeddings makes the latent vectors a proxy for "is_following", which
    # we explicitly removed from model features. CF now learns purely from
    # content engagement (likes, comments, saves, shares) and actual enrollments.
    #
    # if follows_hist is not None and not follows_hist.empty:
    #     follow_df = follows_hist[["follower_id", "following_id"]].drop_duplicates().copy()
    #     follow_df.columns = ["user_id", "mentor_id"]
    #     follow_df["score"] = 3.0
    #     signals.append(follow_df)

    # 3. Mentorship enrollments (strongest signal of preference)
    if mentorships_hist is not None and not mentorships_hist.empty:
        if "mentee_id" in mentorships_hist.columns and "mentor_id" in mentorships_hist.columns:
            enroll_df = mentorships_hist[["mentee_id", "mentor_id"]].drop_duplicates().copy()
            enroll_df.columns = ["user_id", "mentor_id"]
            enroll_df["score"] = 5.0
            signals.append(enroll_df)

    if not signals:
        logger.warning("build_cf_embeddings: no interaction signals available")
        return {"user_factors": {}, "item_factors": {}}

    combined = pd.concat(signals, ignore_index=True)
    combined["user_id"] = pd.to_numeric(combined["user_id"], errors="coerce")
    combined["mentor_id"] = pd.to_numeric(combined["mentor_id"], errors="coerce")
    combined = combined.dropna()
    combined[["user_id", "mentor_id"]] = combined[["user_id", "mentor_id"]].astype(int)

    # Aggregate: sum scores per (user, mentor) pair
    combined = combined.groupby(["user_id", "mentor_id"], as_index=False)["score"].sum()

    n_interactions = len(combined)
    if n_interactions < 20:
        logger.warning("build_cf_embeddings: only %d interactions — skipping SVD", n_interactions)
        return {"user_factors": {}, "item_factors": {}}

    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize as sk_normalize

    # Build ID mappings
    user_ids = sorted(combined["user_id"].unique())
    mentor_ids = sorted(combined["mentor_id"].unique())
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    mentor_map = {mid: i for i, mid in enumerate(mentor_ids)}

    rows = combined["user_id"].map(user_map).values
    cols = combined["mentor_id"].map(mentor_map).values
    vals = np.log1p(combined["score"].values)  # log-damp outlier interactions

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(mentor_ids)))

    n_components = min(n_factors, min(matrix.shape) - 1)
    if n_components < 2:
        logger.warning("build_cf_embeddings: matrix too small for SVD (shape=%s)", matrix.shape)
        return {"user_factors": {}, "item_factors": {}}

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix)       # (n_users, n_factors)
    item_factors = svd.components_.T                # (n_mentors, n_factors)

    # L2-normalize for cosine-like dot products
    user_factors_norm = sk_normalize(user_factors, axis=1)
    item_factors_norm = sk_normalize(item_factors, axis=1)

    user_factor_dict = {uid: user_factors_norm[i] for uid, i in user_map.items()}
    item_factor_dict = {mid: item_factors_norm[i] for mid, i in mentor_map.items()}

    explained_var = svd.explained_variance_ratio_.sum()
    logger.info(
        "build_cf_embeddings: %d users × %d mentors, %d factors, "
        "%.1f%% variance explained, %d total interactions",
        len(user_ids), len(mentor_ids), n_components,
        explained_var * 100, n_interactions,
    )

    return {"user_factors": user_factor_dict, "item_factors": item_factor_dict}


# ---------------------------------------------------------------------------
# Community co-membership features
# ---------------------------------------------------------------------------

def build_community_membership_sets(
    community_members: pd.DataFrame,
) -> Dict[int, set]:
    """Build community membership sets for each user.

    Returns dict mapping user_id -> set of community_ids.
    Used to compute community overlap (Jaccard) between mentee and mentor.
    """
    if community_members is None or community_members.empty:
        return {}

    user_col = None
    comm_col = None
    for candidate in ("user_id", "member_id"):
        if candidate in community_members.columns:
            user_col = candidate
            break
    for candidate in ("community_id", "group_id"):
        if candidate in community_members.columns:
            comm_col = candidate
            break

    if user_col is None or comm_col is None:
        logger.warning(
            "build_community_membership_sets: cannot find user/community columns "
            "(available: %s)", list(community_members.columns)
        )
        return {}

    cm = community_members[[user_col, comm_col]].drop_duplicates().copy()
    cm[user_col] = pd.to_numeric(cm[user_col], errors="coerce")
    cm[comm_col] = pd.to_numeric(cm[comm_col], errors="coerce")
    cm = cm.dropna()
    cm[[user_col, comm_col]] = cm[[user_col, comm_col]].astype(int)

    user_communities = cm.groupby(user_col)[comm_col].apply(set).to_dict()
    logger.info(
        "build_community_membership_sets: %d users with community memberships",
        len(user_communities),
    )
    return user_communities


# ---------------------------------------------------------------------------
# Post requirement matching features
# ---------------------------------------------------------------------------

def build_requirement_sets(
    requirements: pd.DataFrame,
    posts_hist: pd.DataFrame,
) -> Dict[int, set]:
    """Build mentor requirement sets from mentorship post requirements.

    For each mentor, collects all technology_ids required across their posts.
    Returns dict mapping mentor_id -> set of required technology_ids.

    Note: This function is deprecated and no longer used.
    requirement_coverage features have been removed as they measure
    program fit rather than mentor-mentee compatibility.
    """
    # Function kept for backwards compatibility but returns empty dict
    return {}

    req = requirements[["post_id", "technology_id"]].drop_duplicates().copy()
    req["post_id"] = pd.to_numeric(req["post_id"], errors="coerce")
    req["technology_id"] = pd.to_numeric(req["technology_id"], errors="coerce")
    req = req.dropna()
    req[["post_id", "technology_id"]] = req[["post_id", "technology_id"]].astype(int)

    # Map posts to mentors
    post_to_mentor = posts_hist[["post_id", "mentor_id"]].drop_duplicates().copy()
    post_to_mentor["post_id"] = pd.to_numeric(post_to_mentor["post_id"], errors="coerce")
    post_to_mentor["mentor_id"] = pd.to_numeric(post_to_mentor["mentor_id"], errors="coerce")
    post_to_mentor = post_to_mentor.dropna()
    post_to_mentor[["post_id", "mentor_id"]] = post_to_mentor[["post_id", "mentor_id"]].astype(int)

    req = req.merge(post_to_mentor, on="post_id", how="inner")
    mentor_reqs = req.groupby("mentor_id")["technology_id"].apply(set).to_dict()
    logger.info("build_requirement_sets: %d mentors with post requirements", len(mentor_reqs))
    return mentor_reqs


# ---------------------------------------------------------------------------
# Per-pair time-window helpers (causal feature computation)
# ---------------------------------------------------------------------------

def _compute_pair_interactions(
    pair_base: pd.DataFrame,
    event_time_by_mentee: pd.Series,
    likes_hist: pd.DataFrame | None,
    comments_hist: pd.DataFrame | None,
    saves_hist: pd.DataFrame | None,
    shares_hist: pd.DataFrame | None,
    posts_hist: pd.DataFrame,
) -> pd.DataFrame:
    """Compute interaction features per (mentee, mentor) pair up to the pair's event time.

    Uses post creation time as the causal anchor: a post must exist before
    the application for its interactions to be counted.  This avoids future
    data leakage for early-application pairs.
    """
    # Post → mentor mapping (posts created before they can receive interactions)
    if "post_id" not in posts_hist.columns or "mentor_id" not in posts_hist.columns:
        logger.warning("_compute_pair_interactions: posts_hist missing post_id/mentor_id — skipping")
        pair_base["interaction_score_log"] = 0.0
        pair_base["interaction_count_log"] = 0.0
        return pair_base

    post_to_mentor = (
        posts_hist[["post_id", "mentor_id"]]
        .drop_duplicates()
        .copy()
    )
    post_to_mentor["mentor_id"] = pd.to_numeric(post_to_mentor["mentor_id"], errors="coerce")
    post_to_mentor = post_to_mentor.dropna(subset=["post_id", "mentor_id"])

    all_events = []
    source_map = [
        ("likes", likes_hist, 1, "created_at"),
        ("comments", comments_hist, 2, "created_at"),
        ("saves", saves_hist, 3, "created_at"),
        ("shares", shares_hist, 4, "shared_at"),
    ]

    for _name, table, score, time_col in source_map:
        if table is None or table.empty:
            continue
        if "post_id" not in table.columns:
            continue
        user_col = "user_id"
        if user_col not in table.columns:
            continue
        t = table[[user_col, "post_id", time_col]].copy()
        t = t.rename(columns={time_col: "event_time"})
        t["score"] = score
        # Map post_id → mentor_id
        t = t.merge(post_to_mentor, on="post_id", how="inner")
        t = t[[user_col, "mentor_id", "event_time", "score"]]
        if not t.empty:
            all_events.append(t)

    if not all_events:
        pair_base["interaction_score_log"] = 0.0
        pair_base["interaction_count_log"] = 0.0
        return pair_base

    events = pd.concat(all_events, ignore_index=True)
    events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce")
    events = events.dropna(subset=["event_time"])
    events["user_id"] = pd.to_numeric(events["user_id"], errors="coerce").astype(int)
    events["mentor_id"] = pd.to_numeric(events["mentor_id"], errors="coerce").astype(int)
    # Rename to match pair_base column name for merge_asof
    events = events.rename(columns={"user_id": "mentee_id"})

    # Sort and compute cumulative stats per (mentee, mentor) over time
    # CRITICAL: merge_asof requires the RIGHT DataFrame to be sorted GLOBALLY
    # by the merge key (event_time), not just within groups.
    events = events.sort_values("event_time")
    events["cum_score"] = events.groupby(["mentee_id", "mentor_id"])["score"].cumsum()
    events["cum_count"] = events.groupby(["mentee_id", "mentor_id"]).cumcount() + 1

    # Build pairs with event_time
    pairs = pair_base[["mentee_id", "mentor_id"]].drop_duplicates().copy()
    pairs = pairs.merge(
        event_time_by_mentee.rename("event_time").reset_index(),
        left_on="mentee_id",
        right_on="mentee_id",
        how="left",
    )
    pairs["event_time"] = pd.to_datetime(pairs["event_time"], errors="coerce")
    # If a mentee has no event_time, fall back to the earliest event in the data
    # (conservative: counts nothing if truly unknown)
    if pairs["event_time"].isna().any():
        min_time = events["event_time"].min()
        pairs["event_time"] = pairs["event_time"].fillna(min_time)
    pairs = pairs.sort_values("event_time")

    # Ensure type consistency for merge_asof (must match exactly)
    pairs["mentee_id"] = pd.to_numeric(pairs["mentee_id"], errors="coerce").astype("int64")
    pairs["mentor_id"] = pd.to_numeric(pairs["mentor_id"], errors="coerce").astype("int64")
    events["mentee_id"] = pd.to_numeric(events["mentee_id"], errors="coerce").astype("int64")
    events["mentor_id"] = pd.to_numeric(events["mentor_id"], errors="coerce").astype("int64")

    # merge_asof: cumulative score at event_time for each (mentee, mentor) pair
    merged = pd.merge_asof(
        pairs,
        events,
        left_on="event_time",
        right_on="event_time",
        by=["mentee_id", "mentor_id"],
        direction="backward",
    )
    merged["interaction_score_log"] = np.log1p(merged["cum_score"].fillna(0))
    merged["interaction_count_log"] = np.log1p(merged["cum_count"].fillna(0))

    pair_base = pair_base.merge(
        merged[["mentee_id", "mentor_id", "interaction_score_log", "interaction_count_log"]],
        on=["mentee_id", "mentor_id"],
        how="left",
    )
    pair_base["interaction_score_log"] = pair_base["interaction_score_log"].fillna(0)
    pair_base["interaction_count_log"] = pair_base["interaction_count_log"].fillna(0)
    return pair_base


def _compute_pair_follower_count(
    pair_base: pd.DataFrame,
    event_time_by_mentee: pd.Series,
    follows_hist: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mentor follower count per pair up to the mentee's event time.

    A mentor's follower count at the time a mentee applies is the number of
    users who followed the mentor before that application date.
    """
    if follows_hist is None or follows_hist.empty:
        pair_base["mentor_follower_count_log"] = 0.0
        return pair_base

    if "following_id" not in follows_hist.columns:
        pair_base["mentor_follower_count_log"] = 0.0
        return pair_base

    follows = follows_hist.copy()
    follows["following_id"] = pd.to_numeric(follows["following_id"], errors="coerce")
    # Use the raw timestamp column; fall back to "created_at" (DB) then "followed_at" (CSV)
    time_col = None
    for c in ["created_at", "followed_at"]:
        if c in follows.columns:
            time_col = c
            break
    if time_col is None:
        logger.warning("_compute_pair_follower_count: no timestamp column in follows — skipping")
        pair_base["mentor_follower_count_log"] = 0.0
        return pair_base

    follows["follow_time"] = pd.to_datetime(follows[time_col], errors="coerce")
    follows = follows.dropna(subset=["following_id", "follow_time"])
    follows = follows.rename(columns={"following_id": "mentor_id"})
    follows["mentor_id"] = follows["mentor_id"].astype(int)

    # Cumulative followers per mentor over time
    # CRITICAL: merge_asof requires RIGHT df sorted GLOBALLY by merge key.
    follows = follows.sort_values("follow_time")
    follows["cum_followers"] = follows.groupby("mentor_id").cumcount() + 1

    # Pairs with event_time
    pairs = pair_base[["mentee_id", "mentor_id"]].drop_duplicates().copy()
    pairs = pairs.merge(
        event_time_by_mentee.rename("event_time").reset_index(),
        left_on="mentee_id",
        right_on="mentee_id",
        how="left",
    )
    pairs["event_time"] = pd.to_datetime(pairs["event_time"], errors="coerce")
    if pairs["event_time"].isna().any():
        min_time = follows["follow_time"].min()
        pairs["event_time"] = pairs["event_time"].fillna(min_time)
    pairs = pairs.sort_values("event_time")

    # Ensure type consistency for merge_asof
    pairs["mentor_id"] = pd.to_numeric(pairs["mentor_id"], errors="coerce").astype("int64")
    follows["mentor_id"] = pd.to_numeric(follows["mentor_id"], errors="coerce").astype("int64")

    merged = pd.merge_asof(
        pairs,
        follows,
        left_on="event_time",
        right_on="follow_time",
        by=["mentor_id"],
        direction="backward",
    )
    merged["mentor_follower_count_log"] = np.log1p(merged["cum_followers"].fillna(0))

    pair_base = pair_base.merge(
        merged[["mentee_id", "mentor_id", "mentor_follower_count_log"]],
        on=["mentee_id", "mentor_id"],
        how="left",
    )
    pair_base["mentor_follower_count_log"] = pair_base["mentor_follower_count_log"].fillna(0)
    return pair_base


def _compute_pair_popularity(
    pair_base: pd.DataFrame,
    event_time_by_mentee: pd.Series,
    posts_hist: pd.DataFrame,
    mentorships_hist: pd.DataFrame | None,
    likes_hist, comments_hist, saves_hist, shares_hist,
) -> pd.DataFrame:
    """Compute mentor popularity per pair up to the mentee's event time.

    Components (all causal — counted only if they occurred before the pair's
    event time):
      - program count: posts created by mentor before event_time
      - enrollment count: mentorships started before event_time
      - engagement count: likes/comments/saves/shares on mentor's posts
        where the post was created before event_time
    """
    # --- Program count ---
    programs = posts_hist[["mentor_id", "created_at"]].copy()
    programs["created_at"] = pd.to_datetime(programs["created_at"], errors="coerce")
    programs["mentor_id"] = pd.to_numeric(programs["mentor_id"], errors="coerce")
    programs = programs.dropna()
    # CRITICAL: merge_asof requires RIGHT df sorted GLOBALLY by merge key.
    programs = programs.sort_values("created_at")
    programs["cum_programs"] = programs.groupby("mentor_id").cumcount() + 1

    # --- Enrollment count ---
    if mentorships_hist is not None and not mentorships_hist.empty and "mentor_id" in mentorships_hist.columns:
        # Find the date column (StartDate in raw, might be mapped or not)
        enroll_time_col = None
        for c in ["start_date", "StartDate", "startDate", "created_at"]:
            if c in mentorships_hist.columns:
                enroll_time_col = c
                break
        if enroll_time_col:
            enrollments = mentorships_hist[["mentor_id", enroll_time_col]].copy()
            enrollments[enroll_time_col] = pd.to_datetime(enrollments[enroll_time_col], errors="coerce")
            enrollments["mentor_id"] = pd.to_numeric(enrollments["mentor_id"], errors="coerce")
            enrollments = enrollments.dropna()
            # CRITICAL: merge_asof requires RIGHT df sorted GLOBALLY by merge key.
            enrollments = enrollments.sort_values(enroll_time_col)
            enrollments["cum_enrollments"] = enrollments.groupby("mentor_id").cumcount() + 1
        else:
            enrollments = pd.DataFrame(columns=["mentor_id", enroll_time_col, "cum_enrollments"])
    else:
        enrollments = pd.DataFrame(columns=["mentor_id", "start_date", "cum_enrollments"])

    # --- Engagement count (interactions on mentor's posts, post creation <= event_time) ---
    if "post_id" in posts_hist.columns and "mentor_id" in posts_hist.columns:
        post_to_mentor_time = (
            posts_hist[["post_id", "mentor_id", "created_at"]]
            .drop_duplicates()
            .copy()
        )
        post_to_mentor_time["mentor_id"] = pd.to_numeric(
            post_to_mentor_time["mentor_id"], errors="coerce"
        )
        post_to_mentor_time["created_at"] = pd.to_datetime(
            post_to_mentor_time["created_at"], errors="coerce"
        )
        post_to_mentor_time = post_to_mentor_time.dropna()

        eng_events = []
        for table, time_col in [(likes_hist, "created_at"), (comments_hist, "created_at"),
                                (saves_hist, "created_at"), (shares_hist, "shared_at")]:
            if table is None or table.empty or "post_id" not in table.columns:
                continue
            t = table[["post_id", time_col]].copy()
            t = t.rename(columns={time_col: "event_time"})
            t = t.merge(post_to_mentor_time, on="post_id", how="inner")
            t = t[["mentor_id", "event_time"]]
            if not t.empty:
                eng_events.append(t)

        if eng_events:
            engagements = pd.concat(eng_events, ignore_index=True)
            engagements["event_time"] = pd.to_datetime(engagements["event_time"], errors="coerce")
            engagements = engagements.dropna()
            # CRITICAL: merge_asof requires RIGHT df sorted GLOBALLY by merge key.
            engagements = engagements.sort_values("event_time")
            engagements["cum_engagement"] = engagements.groupby("mentor_id").cumcount() + 1
        else:
            engagements = pd.DataFrame(columns=["mentor_id", "event_time", "cum_engagement"])
    else:
        engagements = pd.DataFrame(columns=["mentor_id", "event_time", "cum_engagement"])

    # --- Merge all components per pair using merge_asof ---
    pairs = pair_base[["mentee_id", "mentor_id"]].drop_duplicates().copy()
    pairs = pairs.merge(
        event_time_by_mentee.rename("event_time").reset_index(),
        left_on="mentee_id",
        right_on="mentee_id",
        how="left",
    )
    pairs["event_time"] = pd.to_datetime(pairs["event_time"], errors="coerce")
    if pairs["event_time"].isna().any():
        min_time = min(
            programs["created_at"].min() if not programs.empty else pd.Timestamp.min,
            enrollments[enroll_time_col].min() if not enrollments.empty and enroll_time_col in enrollments.columns else pd.Timestamp.min,
            engagements["event_time"].min() if not engagements.empty else pd.Timestamp.min,
        )
        pairs["event_time"] = pairs["event_time"].fillna(min_time)
    pairs = pairs.sort_values("event_time")

    # Ensure type consistency for merge_asof
    pairs["mentor_id"] = pd.to_numeric(pairs["mentor_id"], errors="coerce").astype("int64")
    programs["mentor_id"] = pd.to_numeric(programs["mentor_id"], errors="coerce").astype("int64")
    if not enrollments.empty:
        enrollments["mentor_id"] = pd.to_numeric(enrollments["mentor_id"], errors="coerce").astype("int64")
    if not engagements.empty:
        engagements["mentor_id"] = pd.to_numeric(engagements["mentor_id"], errors="coerce").astype("int64")

    # Programs
    merged = pd.merge_asof(
        pairs, programs, left_on="event_time", right_on="created_at",
        by=["mentor_id"], direction="backward",
    )
    merged["program_count"] = merged["cum_programs"].fillna(0)

    # Enrollments
    if not enrollments.empty and enroll_time_col:
        merged = pd.merge_asof(
            merged, enrollments, left_on="event_time", right_on=enroll_time_col,
            by=["mentor_id"], direction="backward",
        )
        merged["enrollment_count"] = merged["cum_enrollments"].fillna(0)
    else:
        merged["enrollment_count"] = 0.0

    # Engagements
    if not engagements.empty:
        merged = pd.merge_asof(
            merged, engagements, left_on="event_time", right_on="event_time",
            by=["mentor_id"], direction="backward",
        )
        merged["engagement_count"] = merged["cum_engagement"].fillna(0)
    else:
        merged["engagement_count"] = 0.0

    # Composite popularity score (same weights as build_mentor_popularity_features)
    # Weights: enrollments 0.55, programs 0.30, engagement 0.15
    merged["program_count_log"] = np.log1p(merged["program_count"])
    merged["enrollment_count_log"] = np.log1p(merged["enrollment_count"])
    merged["engagement_count_log"] = np.log1p(merged["engagement_count"])

    has_engagement = (merged["engagement_count"] > 0).astype(int)
    enrollment_w = 0.55 * has_engagement + 0.60 * (1 - has_engagement)
    program_w = 0.30 * has_engagement + 0.40 * (1 - has_engagement)
    engagement_w = 0.15 * has_engagement

    merged["mentor_program_popularity"] = (
        enrollment_w * merged["enrollment_count_log"]
        + program_w * merged["program_count_log"]
        + engagement_w * merged["engagement_count_log"]
    )

    # Apply same 95th-pct cap as build_mentor_popularity_features for consistency
    cap = merged["mentor_program_popularity"].quantile(0.95) if len(merged) > 10 else merged["mentor_program_popularity"].max()
    if cap > 0:
        merged["mentor_program_popularity"] = merged["mentor_program_popularity"].clip(upper=cap)

    pair_base = pair_base.merge(
        merged[["mentee_id", "mentor_id", "mentor_program_popularity"]],
        on=["mentee_id", "mentor_id"],
        how="left",
    )
    pair_base["mentor_program_popularity"] = pair_base["mentor_program_popularity"].fillna(0)
    return pair_base


# Pair-level features
# ---------------------------------------------------------------------------

def build_pair_features(
    candidate_pool: pd.DataFrame,
    mentee_features: pd.DataFrame,
    mentor_features: pd.DataFrame,
    interaction_features: pd.DataFrame,
    follows_hist: pd.DataFrame,
    cf_embeddings: Dict | None = None,
    community_sets: Dict[int, set] | None = None,
    mentor_requirement_sets: Dict[int, set] | None = None,
    event_time_by_mentee: pd.Series | None = None,
    likes_hist: pd.DataFrame | None = None,
    comments_hist: pd.DataFrame | None = None,
    saves_hist: pd.DataFrame | None = None,
    shares_hist: pd.DataFrame | None = None,
    posts_hist: pd.DataFrame | None = None,
    mentorships_hist_raw: pd.DataFrame | None = None,
    follows_hist_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute pairwise content features for every (mentee, mentor) candidate pair.

    Uses vectorized operations wherever possible.  List comprehensions for
    Jaccard/coverage are O(n_pairs) with negligible per-element cost on small
    Python sets (typically <20 elements).  For >500k pairs, consider sparse
    matrix operations.

    Time-window features (interaction, follower count, popularity) are computed
    per-pair using the mentee's event_time when raw data is provided.  This
    prevents future data leakage for early-application pairs.
    """
    pair_base = candidate_pool[["mentee_id", "mentor_id"]].drop_duplicates().copy()

    mentee_skill = (
        mentee_features[["mentee_id", "interests_set"]]
        .copy()
        .rename(columns={"interests_set": "mentee_interests"})
    )
    mentor_skill = (
        mentor_features[["mentor_id", "expertise_set"]]
        .copy()
        .rename(columns={"expertise_set": "mentor_expertise"})
    )
    mentee_sub = (
        mentee_features[["mentee_id", "subdomains_set"]]
        .copy()
        .rename(columns={"subdomains_set": "mentee_subdomains"})
    )
    mentor_sub = (
        mentor_features[["mentor_id", "subdomains_set"]]
        .copy()
        .rename(columns={"subdomains_set": "mentor_subdomains"})
    )

    pair_base = (
        pair_base
        .merge(mentee_skill, on="mentee_id", how="left")
        .merge(mentor_skill, on="mentor_id", how="left")
        .merge(mentee_sub, on="mentee_id", how="left")
        .merge(mentor_sub, on="mentor_id", how="left")
    )
    pair_base["mentee_interests"] = pair_base["mentee_interests"].apply(as_safe_set)
    pair_base["mentor_expertise"] = pair_base["mentor_expertise"].apply(as_safe_set)
    pair_base["mentee_subdomains"] = pair_base["mentee_subdomains"].apply(as_safe_set)
    pair_base["mentor_subdomains"] = pair_base["mentor_subdomains"].apply(as_safe_set)

    pair_base["skill_overlap_score"] = [
        jaccard(m_skill, n_skill)
        for m_skill, n_skill in zip(pair_base["mentee_interests"], pair_base["mentor_expertise"])
    ]
    pair_base["skill_coverage_score"] = [
        skill_coverage(m_skill, n_skill)
        for m_skill, n_skill in zip(pair_base["mentee_interests"], pair_base["mentor_expertise"])
    ]
    pair_base["subdomain_similarity"] = [
        jaccard(m_sub, n_sub)
        for m_sub, n_sub in zip(pair_base["mentee_subdomains"], pair_base["mentor_subdomains"])
    ]

    # Skill direction features — directional signals about skill set coverage
    # mentor_covers_all_skills: 1 if mentor skills ⊇ mentee skills
    pair_base["mentor_covers_all_skills"] = [
        int(mentee_s <= mentor_s) if mentee_s and mentor_s else 0
        for mentee_s, mentor_s in zip(pair_base["mentee_interests"], pair_base["mentor_expertise"])
    ]
    # extra_skill_count: number of mentor skills beyond mentee's (0 if mentor has fewer)
    pair_base["extra_skill_count"] = [
        max(len(mentor_s - mentee_s), 0) if mentee_s and mentor_s else 0
        for mentee_s, mentor_s in zip(pair_base["mentee_interests"], pair_base["mentor_expertise"])
    ]
    # skill_match_type: 2=mentor has MORE skills (ideal), 1=EXACT match, 0=mentor has LESS (bad)
    pair_base["skill_match_type"] = [
        2 if len(mentor_s) > len(mentee_s) else (1 if len(mentor_s) == len(mentee_s) else 0)
        if mentee_s and mentor_s else 0
        for mentee_s, mentor_s in zip(pair_base["mentee_interests"], pair_base["mentor_expertise"])
    ]

    pair_base = pair_base.drop(
        columns=["mentee_interests", "mentor_expertise", "mentee_subdomains", "mentor_subdomains"]
    )

    # Merge mentor quality (time-independent) and open post count (availability)
    mentor_merge_cols = ["mentor_id", "mentor_quality_score", "mentor_weighted_rating"]
    for _mc in ["mentor_open_post_count_log"]:
        if _mc in mentor_features.columns:
            mentor_merge_cols.append(_mc)
    pair_base = pair_base.merge(
        mentor_features[mentor_merge_cols],
        on="mentor_id",
        how="left",
    )
    pair_base["mentor_open_post_count_log"] = pair_base.get("mentor_open_post_count_log", 0).fillna(0)

    # ── Per-pair time-window features (causal, no future leakage) ──
    # If raw data + event_time provided, compute interaction/popularity/followers
    # as of each pair's event_time.  Otherwise fall back to pre-computed snapshots.
    has_event_time = event_time_by_mentee is not None and not event_time_by_mentee.empty
    has_raw_data = posts_hist is not None and not posts_hist.empty

    if has_event_time and has_raw_data:
        logger.info("build_pair_features: computing per-pair time-window features (causal)")

        # Interaction features per pair (post creation time as anchor)
        pair_base = _compute_pair_interactions(
            pair_base, event_time_by_mentee,
            likes_hist, comments_hist, saves_hist, shares_hist,
            posts_hist,
        )

        # Follower count per pair (follow creation time)
        if follows_hist_raw is not None and not follows_hist_raw.empty:
            pair_base = _compute_pair_follower_count(
                pair_base, event_time_by_mentee, follows_hist_raw,
            )
        else:
            pair_base["mentor_follower_count_log"] = 0.0

        # Popularity per pair (programs + enrollments + engagement up to event_time)
        pair_base = _compute_pair_popularity(
            pair_base, event_time_by_mentee,
            posts_hist, mentorships_hist_raw,
            likes_hist, comments_hist, saves_hist, shares_hist,
        )
    else:
        # Fallback: use pre-computed snapshot features from mentor_features / interaction_features
        logger.info("build_pair_features: using pre-computed snapshot features (no event_time or raw data)")

        # Popularity + follower count from mentor snapshot
        snap_cols = ["mentor_id", "mentor_program_popularity"]
        for _mc in ["mentor_follower_count_log"]:
            if _mc in mentor_features.columns:
                snap_cols.append(_mc)
        pair_base = pair_base.merge(
            mentor_features[snap_cols], on="mentor_id", how="left"
        )
        pair_base["mentor_follower_count_log"] = pair_base.get("mentor_follower_count_log", 0).fillna(0)

        # Interaction features from pre-computed table
        if interaction_features is not None and not interaction_features.empty:
            pair_base = pair_base.merge(
                interaction_features.rename(columns={"user_id": "mentee_id"})[
                    ["mentee_id", "mentor_id", "interaction_score_log", "interaction_count_log"]
                ],
                on=["mentee_id", "mentor_id"],
                how="left",
            )
        pair_base["interaction_score_log"] = pair_base.get("interaction_score_log", 0).fillna(0)
        pair_base["interaction_count_log"] = pair_base.get("interaction_count_log", 0).fillna(0)

    # Follow signal: binary only, de-duplicated to avoid inflation
    follows_df = (
        follows_hist[["follower_id", "following_id"]]
        .drop_duplicates()
        .copy()
    )
    follows_df.columns = ["mentee_id", "mentor_id"]
    follows_df["is_following"] = 1
    pair_base = pair_base.merge(follows_df, on=["mentee_id", "mentor_id"], how="left")
    pair_base["is_following"] = pair_base["is_following"].fillna(0).astype(int)

    # Country match: binary feature (same_country)
    mentee_country = mentee_features[["mentee_id", "country_code"]].copy()
    mentee_country.columns = ["mentee_id", "mentee_country"]
    mentor_country = mentor_features[["mentor_id", "country_code"]].copy()
    mentor_country.columns = ["mentor_id", "mentor_country"]
    pair_base = pair_base.merge(mentee_country, on="mentee_id", how="left")
    pair_base = pair_base.merge(mentor_country, on="mentor_id", how="left")
    mentee_country_norm = pair_base["mentee_country"].fillna("").astype(str).str.strip().str.lower()
    mentor_country_norm = pair_base["mentor_country"].fillna("").astype(str).str.strip().str.lower()
    has_country = (mentee_country_norm != "") & (mentor_country_norm != "")
    pair_base["same_country"] = (has_country & (mentee_country_norm == mentor_country_norm)).astype(int)
    pair_base = pair_base.drop(columns=["mentee_country", "mentor_country"])

    # Domain match: binary feature (mentee domain_id == mentor domain_id)
    # Both domain_id columns come from profile tables (SCD Type 1 — no time filter).
    # Safe: static profile attribute, independent of applications.
    mentee_domain = mentee_features[["mentee_id"]].copy()
    mentor_domain = mentor_features[["mentor_id"]].copy()
    if "domain_id" in mentee_features.columns:
        mentee_domain["mentee_domain_id"] = pd.to_numeric(
            mentee_features["domain_id"], errors="coerce"
        )
    else:
        mentee_domain["mentee_domain_id"] = np.nan
    if "domain_id" in mentor_features.columns:
        mentor_domain["mentor_domain_id"] = pd.to_numeric(
            mentor_features["domain_id"], errors="coerce"
        )
    else:
        mentor_domain["mentor_domain_id"] = np.nan
    pair_base = pair_base.merge(mentee_domain, on="mentee_id", how="left")
    pair_base = pair_base.merge(mentor_domain, on="mentor_id", how="left")
    valid_domain = pair_base["mentee_domain_id"].notna() & pair_base["mentor_domain_id"].notna()
    pair_base["mentor_domain_match"] = 0
    pair_base.loc[valid_domain, "mentor_domain_match"] = (
        pair_base.loc[valid_domain, "mentee_domain_id"] == pair_base.loc[valid_domain, "mentor_domain_id"]
    ).astype(int)
    pair_base = pair_base.drop(columns=["mentee_domain_id", "mentor_domain_id"])

    # Experience gap features
    mentee_exp = mentee_features[["mentee_id", "experience_level_num"]].rename(
        columns={"experience_level_num": "mentee_experience_level_num"}
    )
    mentor_exp = mentor_features[["mentor_id", "experience_level_num"]].rename(
        columns={"experience_level_num": "mentor_experience_level_num"}
    )
    pair_base = pair_base.merge(mentee_exp, on="mentee_id", how="left")
    pair_base = pair_base.merge(mentor_exp, on="mentor_id", how="left")
    pair_base["mentee_experience_level_num"] = (
        pd.to_numeric(pair_base["mentee_experience_level_num"], errors="coerce").fillna(1).astype(int)
    )
    pair_base["mentor_experience_level_num"] = (
        pd.to_numeric(pair_base["mentor_experience_level_num"], errors="coerce").fillna(1).astype(int)
    )
    pair_base["experience_gap"] = (
        pair_base["mentor_experience_level_num"] - pair_base["mentee_experience_level_num"]
    )
    pair_base["experience_gap_abs"] = pair_base["experience_gap"].abs()
    pair_base["mentor_more_experienced"] = (pair_base["experience_gap"] > 0).astype(int)

    # Experience match bucket: encodes how well experience levels align
    #   0 → mentor <= mentee (bad — no seniority advantage)
    #   2 → gap in [1, 2]    (ideal — close, meaningful mentorship)
    #   1 → gap >= 3          (acceptable — large gap but still useful)
    gap = pair_base["experience_gap"]
    pair_base["experience_match_bucket"] = np.where(
        gap <= 0, 0,
        np.where(gap <= 2, 2, 1)
    )

    # Soft gap score: smooth continuous signal, 1.0 = perfect match, decays with gap
    pair_base["soft_gap_score"] = 1.0 / (1.0 + pair_base["experience_gap_abs"])

    # Popularity: already log-scaled and 95th-pct capped in build_mentor_popularity_features.
    # Apply per-mentee-group normalization to reduce inter-group scale differences
    # and prevent popular mentors from dominating content-based signals.
    raw_pop = pair_base["mentor_program_popularity"].fillna(0).clip(lower=0)
    # Group-normalize: within each mentee's candidate set, scale popularity to [0, 1]
    pop_max = pair_base.groupby("mentee_id")["mentor_program_popularity"].transform("max")
    pop_max = pop_max.replace(0, 1)  # avoid division by zero
    pair_base["popularity_log"] = (raw_pop / pop_max).fillna(0).clip(0, 1)

    # ── Collaborative filtering score ──
    # Dot product of user and item latent factors from SVD decomposition.
    # Captures hidden preference patterns: "users like you preferred this mentor."
    if cf_embeddings and cf_embeddings.get("user_factors") and cf_embeddings.get("item_factors"):
        _uf = cf_embeddings["user_factors"]
        _if = cf_embeddings["item_factors"]
        _zero = np.zeros(len(next(iter(_uf.values())))) if _uf else np.zeros(1)
        pair_base["cf_score"] = [
            float(np.dot(_uf.get(int(m), _zero), _if.get(int(t), _zero)))
            for m, t in zip(pair_base["mentee_id"], pair_base["mentor_id"])
        ]
        logger.info(
            "build_pair_features: cf_score — non-zero: %d / %d",
            (pair_base["cf_score"] != 0).sum(), len(pair_base),
        )
    else:
        pair_base["cf_score"] = 0.0

    # ── Community overlap ──
    # Jaccard similarity of community memberships between mentee and mentor.
    # Captures social proximity: shared communities imply shared interests.
    if community_sets:
        pair_base["community_overlap"] = [
            jaccard(
                community_sets.get(int(m), set()),
                community_sets.get(int(t), set()),
            )
            for m, t in zip(pair_base["mentee_id"], pair_base["mentor_id"])
        ]
        logger.info(
            "build_pair_features: community_overlap — non-zero: %d / %d",
            (pair_base["community_overlap"] != 0).sum(), len(pair_base),
        )
    else:
        pair_base["community_overlap"] = 0.0

    # ── SCALE DOWN CF & INTERACTION SCORES (May 2026 - Hybrid Rebalancing) ──
    # CF score and interaction_score_log were dominating the model despite reweighting.
    # Scale them down AGGRESSIVELY to reduce their impact while keeping hybrid system.
    # CRITICAL: CF embeddings are highly correlated with is_following (both measure
    # social proximity). Even after removing is_following from model features, CF still
    # acts as a proxy for follows. Scale factor reduced from 0.33 to 0.15 to further
    # reduce dominance.
    CF_SCALE_FACTOR = 0.15  # AGGRESSIVE: divide by 6.7x (was 0.33)
    INTERACTION_SCALE_FACTOR = 0.15  # AGGRESSIVE: divide by 6.7x
    
    if "cf_score" in pair_base.columns:
        original_cf_mean = pair_base["cf_score"].mean()
        pair_base["cf_score"] = pair_base["cf_score"] * CF_SCALE_FACTOR
        scaled_cf_mean = pair_base["cf_score"].mean()
        logger.info(
            "build_pair_features: CF score SCALED DOWN AGGRESSIVELY (factor=%.2f) — mean %.4f → %.4f",
            CF_SCALE_FACTOR, original_cf_mean, scaled_cf_mean,
        )
    
    if "interaction_score_log" in pair_base.columns:
        original_int_mean = pair_base["interaction_score_log"].mean()
        pair_base["interaction_score_log"] = pair_base["interaction_score_log"] * INTERACTION_SCALE_FACTOR
        scaled_int_mean = pair_base["interaction_score_log"].mean()
        logger.info(
            "build_pair_features: interaction_score_log SCALED DOWN AGGRESSIVELY (factor=%.2f) — mean %.4f → %.4f",
            INTERACTION_SCALE_FACTOR, original_int_mean, scaled_int_mean,
        )

    # ── Requirement coverage ──
    # How well mentee interests match the specific technologies a mentor
    # requires in their posts. More precise than general skill_overlap_score.
    # Requirement features removed: focus on mentor-mentee compatibility only
    # (not program-mentee fit)

    pair_base = pair_base.fillna(0)

    # ── FEATURE LEAKAGE AUDIT ──
    # Label: label=1 if user applied to mentor (from mentorship_applications).
    # Each feature below is verified safe against this label definition.
    #
    # SAFE — content-based (profile data, no temporal dependency):
    #   skill_overlap_score, skill_coverage_score, subdomain_similarity,
    #   experience_gap_abs, mentor_more_experienced, same_country,
    #   mentor_domain_match (SCD Type 1 profile attribute — not temporal)
    #
    # SAFE — mentor-level aggregate (NOT pair-specific, uses data <= train_end):
    #   mentor_quality_score  (from feedback, not applications)
    #   mentor_weighted_rating (Bayesian-smoothed rating from feedback)
    #   popularity_log (mentor_program_popularity: log-based weighted sum of
    #                   enrollment, program, and engagement counts
    #                   — application counts EXCLUDED to prevent label leakage,
    #                   already in log scale, no additional log applied)
    #   mentor_follower_count_log (log1p of follower count from follows, filtered
    #                              <= train_end. Independent of applications.)
    #   mentor_open_post_count_log (log1p of open post count from posts, filtered
    #                               <= train_end. Counts availability, not applications.)
    #
    # SAFE — pair-level interaction (uses likes/comments/saves/shares only,
    #         NOT applications):
    #   interaction_score_log
    #
    # SAFE — social signal (follow relationship, not application signal):
    #   is_following
    #
    # VERIFIED ABSENT — no pair-level application features exist:
    #   No has_applied, applied_before, or any (mentee, mentor) application indicator.
    #   Applications are used ONLY for label assignment in pipeline.py.

    keep_cols = [
        "mentee_id",
        "mentor_id",
        "skill_overlap_score",
        "skill_coverage_score",
        "subdomain_similarity",
        "mentor_quality_score",
        "mentor_weighted_rating",   # Bayesian rating — model feature + explanation text
        "interaction_score_log",
        "interaction_count_log",
        "experience_gap_abs",
        "mentor_more_experienced",
        "experience_match_bucket",
        "soft_gap_score",
        # is_following: REMOVED COMPLETELY (May 2026 - CRITICAL FIX).
        # Reason: 100% correlated with positive labels (all applications are from followers).
        # Keeping it would make CF a proxy for is_following, defeating skill-first design.
        # Follow signal handled via soft reranking only (not training signal).
        "same_country",
        "popularity_log",
        "mentor_covers_all_skills",
        "extra_skill_count",
        "skill_match_type",
        # --- Validated in run_behavioral_features_experiment.py (Exp B) ---
        # NDCG@10: 0.5411 -> 0.5694 (+5.2%), HitRate@10: +0.0 (stable)
        "mentor_follower_count_log",
        "mentor_open_post_count_log",
        "mentor_domain_match",
        # --- Collaborative filtering & community features ---
        "cf_score",
        "community_overlap",
    ]
    keep_cols_present = [c for c in keep_cols if c in pair_base.columns]

    # ── Feature stability logging ──
    # Log distributions of key features for monitoring. Helps detect
    # feature drift, extreme values, or unexpected sparsity.
    _stability_cols = [
        "skill_overlap_score", "skill_coverage_score", "subdomain_similarity",
        "popularity_log", "mentor_follower_count_log", "interaction_score_log",
        "cf_score",
    ]
    for _col in _stability_cols:
        if _col in pair_base.columns:
            _s = pair_base[_col]
            _nonzero = (_s != 0).sum()
            _pct_nonzero = 100 * _nonzero / len(_s) if len(_s) > 0 else 0
            if _pct_nonzero < 5:
                logger.warning(
                    "build_pair_features: feature '%s' is %.1f%% non-zero (%d/%d) — "
                    "may be too sparse to contribute to ranking",
                    _col, _pct_nonzero, _nonzero, len(_s),
                )

    # Guard against NaN/inf in final output
    n_nan = pair_base[keep_cols_present].isna().sum().sum()
    n_inf = np.isinf(pair_base[keep_cols_present].select_dtypes(include=["number"])).sum().sum()
    if n_nan > 0 or n_inf > 0:
        logger.warning(
            "build_pair_features: detected %d NaN and %d inf values — filling with 0",
            n_nan, n_inf,
        )
        pair_base[keep_cols_present] = pair_base[keep_cols_present].replace(
            [np.inf, -np.inf], 0
        ).fillna(0)

    # ── is_following: kept as NON-MODEL metadata for reranking only ──
    # (May 2026 - Skill-First Refactor). is_following is 100% correlated with
    # positive labels, so it's EXCLUDED from DEFAULT_FEATURE_COLS (model never
    # trains on it) and from CF embeddings (no proxy leakage).
    # It survives here ONLY so apply_soft_business_boosts() can apply a
    # tiny ×1.003 reranking boost — a soft social signal that cannot
    # dominate skill-first ranking.
    out = pair_base[keep_cols_present].reset_index(drop=True)
    
    return out


# ---------------------------------------------------------------------------
# Label assignment and train/valid/test splitting
# ---------------------------------------------------------------------------

def assign_labels_and_splits(
    pair_base: pd.DataFrame,
    positive_pairs: Set[Tuple[int, int]] | Dict[str, Set[Tuple[int, int]]],
    event_time_by_mentee: pd.Series,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> pd.DataFrame:
    """Assign labels and time_split using pair-level split assignment.

    When ``positive_pairs`` is a dict keyed by split name (the primary path),
    each candidate pair is assigned to the split where it has a positive label.
    For mentees with positives in multiple splits, their candidate rows are
    duplicated so each split sees the full candidate list for evaluation.

    Mentees with no positives in any split are assigned to "train" as negatives.

    This pair-level approach prevents the "lost positives" problem where
    mentee-level splitting (based on earliest event) would discard 80%+ of
    valid/test positives from returning users.
    """
    out = pair_base.copy()

    if isinstance(positive_pairs, dict):
        # ── Pair-level split assignment ──
        # For each split, find which mentees have positives in that split
        mentees_by_split: Dict[str, set] = {}
        for split_name, pairs in positive_pairs.items():
            mentees_by_split[split_name] = {p[0] for p in pairs}

        # All mentees that have at least one positive in any split
        all_positive_mentees = set()
        for m_set in mentees_by_split.values():
            all_positive_mentees |= m_set

        # Build positive lookup per split
        pos_lookup: Dict[str, set] = {
            split_name: set(pairs) for split_name, pairs in positive_pairs.items()
        }

        split_parts = []

        for split_name in ("train", "valid", "test"):
            split_mentees = mentees_by_split.get(split_name, set())
            if not split_mentees:
                continue

            # Get all candidate rows for mentees who have positives in this split
            mask = out["mentee_id"].isin(split_mentees)
            split_df = out[mask].copy()
            split_df["time_split"] = split_name

            # Assign labels: 1 if (mentee_id, mentor_id) is a positive for this split
            split_pairs = pos_lookup.get(split_name, set())
            split_df["label"] = split_df.apply(
                lambda row: 1 if (int(row["mentee_id"]), int(row["mentor_id"])) in split_pairs else 0,
                axis=1,
            )
            split_parts.append(split_df)

        # Mentees with NO positives in any split → assign to train as negatives
        negative_only_mentees = set(out["mentee_id"].unique()) - all_positive_mentees
        if negative_only_mentees:
            neg_mask = out["mentee_id"].isin(negative_only_mentees)
            neg_df = out[neg_mask].copy()
            neg_df["time_split"] = "train"
            neg_df["label"] = 0
            split_parts.append(neg_df)

        if not split_parts:
            raise ValueError("No data produced after split assignment")

        result = pd.concat(split_parts, ignore_index=True)
        result["event_time"] = result["mentee_id"].map(event_time_by_mentee)
        result["hardness_score"] = result.get(
            "hardness_score", pd.Series(0, index=result.index)
        ).fillna(0)

        # Log split statistics
        for sn in ("train", "valid", "test"):
            s_df = result[result["time_split"] == sn]
            n_pos = int(s_df["label"].sum())
            n_mentees = s_df["mentee_id"].nunique()
            logger.info(
                "assign_labels_and_splits: %s — %d rows, %d positives, %d unique mentees",
                sn, len(s_df), n_pos, n_mentees,
            )
        return result

    # ── Fallback: flat set of positive pairs (legacy path) ──
    out["event_time"] = out["mentee_id"].map(event_time_by_mentee)

    def assign_split(dt: pd.Timestamp) -> str:
        if pd.isna(dt):
            return "train"
        if dt <= train_end:
            return "train"
        if dt <= valid_end:
            return "valid"
        return "test"

    out["time_split"] = out["mentee_id"].map(
        event_time_by_mentee.apply(assign_split)
    ).fillna("train")

    positive_df = (
        pd.DataFrame(list(positive_pairs), columns=["mentee_id", "mentor_id"])
        .drop_duplicates()
    )
    out = out.merge(positive_df.assign(label=1), on=["mentee_id", "mentor_id"], how="left")
    out["label"] = out["label"].fillna(0).astype(int)
    out["hardness_score"] = out.get("hardness_score", pd.Series(0, index=out.index)).fillna(0)
    return out


# ---------------------------------------------------------------------------
# Hard negative sampling
# ---------------------------------------------------------------------------

def sample_hard_negatives_per_group(
    group_df: pd.DataFrame,
    neg_per_pos: int = 4,
    min_candidates_per_group: int = 10,
    rng_seed: int = 42,
    include_domain_negatives: bool = False,
) -> Optional[pd.DataFrame]:
    """Sample hard negatives for a single mentee group (55% hard + 45% random).

    Ratio rationale: 70% hard negatives was too aggressive — positives and
    hard negatives became nearly indistinguishable by feature values, causing
    unstable gradient learning.  55/45 maintains ranking difficulty while
    giving the model clear "easy wins" that stabilize convergence.

    A negative is kept only if it has at least some skill OR subdomain signal.
    This avoids weak negatives that provide no learning value while still
    supporting mentees with narrow interest sets.

    Args:
        include_domain_negatives: If True, also keep negatives with
            mentor_domain_match > 0 (same domain, different skills).
            Used for eval to create realistic-difficulty negatives.
    """
    pos = group_df[group_df["label"] == 1].copy()
    neg = group_df[group_df["label"] == 0].copy()

    if len(pos) == 0:
        return None

    positive_mentor_ids = set(pos["mentor_id"].unique())
    neg = neg[~neg["mentor_id"].isin(positive_mentor_ids)].copy()
    if len(neg) == 0:
        return None

    # Compute hardness score for ALL negatives first
    interaction_component = neg["interaction_score_log"] / (1.0 + neg["interaction_score_log"])
    neg["hardness_score"] = (
        0.45 * neg["skill_overlap_score"]
        + 0.30 * neg["skill_coverage_score"]
        + 0.20 * neg.get("subdomain_similarity", pd.Series(0, index=neg.index))
        + 0.05 * interaction_component
    )

    # Classify negatives by difficulty tier
    has_skill_signal = (neg["skill_overlap_score"] > 0) | (
        neg.get("subdomain_similarity", pd.Series(0, index=neg.index)) > 0
    )
    has_domain_signal = (
        neg.get("mentor_domain_match", pd.Series(0, index=neg.index)) > 0
    )

    if include_domain_negatives:
        # 3-tier difficulty: 50% hard (skill/subdomain), 30% medium (domain),
        # 20% easy (any candidate).  Easy negatives give the model "free"
        # NDCG points; hard/medium ensure HitRate is non-trivial.
        hard_pool = neg[has_skill_signal].copy()
        medium_pool = neg[has_domain_signal & ~has_skill_signal].copy()
        easy_pool = neg[~has_skill_signal & ~has_domain_signal].copy()

        target_neg = max(len(pos) * neg_per_pos, min_candidates_per_group - len(pos))

        n_hard = min(int(np.ceil(target_neg * 0.50)), len(hard_pool))
        n_medium = min(int(np.ceil(target_neg * 0.30)), len(medium_pool))
        n_easy = max(0, min(target_neg - n_hard - n_medium, len(easy_pool)))

        # Fill shortfalls from other tiers
        shortfall = target_neg - n_hard - n_medium - n_easy
        if shortfall > 0 and len(hard_pool) > n_hard:
            extra = min(shortfall, len(hard_pool) - n_hard)
            n_hard += extra
            shortfall -= extra
        if shortfall > 0 and len(medium_pool) > n_medium:
            extra = min(shortfall, len(medium_pool) - n_medium)
            n_medium += extra
            shortfall -= extra
        if shortfall > 0 and len(easy_pool) > n_easy:
            n_easy += min(shortfall, len(easy_pool) - n_easy)

        parts = []
        if n_hard > 0:
            parts.append(hard_pool.nlargest(n_hard, "hardness_score"))
        if n_medium > 0:
            parts.append(
                medium_pool.sample(n_medium, random_state=rng_seed)
                if len(medium_pool) > n_medium else medium_pool
            )
        if n_easy > 0:
            parts.append(
                easy_pool.sample(n_easy, random_state=rng_seed)
                if len(easy_pool) > n_easy else easy_pool
            )
        if not parts:
            return None
        neg_sample = pd.concat(parts, ignore_index=True)
    else:
        # Training: strict signal filter (skill/subdomain only)
        neg = neg[has_skill_signal].copy()
        if len(neg) == 0:
            return None

        score_min = neg["hardness_score"].min()
        score_max = neg["hardness_score"].max()
        if score_max > score_min:
            neg["hardness_score"] = (neg["hardness_score"] - score_min) / (score_max - score_min)
        else:
            neg["hardness_score"] = 0.0

        target_neg = max(len(pos) * neg_per_pos, min_candidates_per_group - len(pos))
        target_neg = min(target_neg, len(neg))
        if target_neg <= 0:
            return None

        # 55% hard / 45% random: balanced difficulty for stable learning
        num_hard = int(np.ceil(target_neg * 0.55))
        num_random = target_neg - num_hard
        hard_negatives = neg.nlargest(num_hard, "hardness_score")
        remaining_neg = neg.drop(hard_negatives.index)
        random_negatives = (
            remaining_neg.sample(num_random, random_state=rng_seed)
            if len(remaining_neg) >= num_random
            else remaining_neg.copy()
        )
        neg_sample = pd.concat([hard_negatives, random_negatives], ignore_index=True)

    sampled = pd.concat([pos, neg_sample], ignore_index=True)
    sampled["hardness_score"] = sampled["hardness_score"].fillna(0)
    return sampled


# ---------------------------------------------------------------------------
# Final recommendation dataset
# ---------------------------------------------------------------------------

def build_recommendation_dataset(
    pair_base: pd.DataFrame,
    positive_pairs: Set[Tuple[int, int]] | Dict[str, Set[Tuple[int, int]]],
    event_time_by_mentee: pd.Series,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
    neg_per_pos: int = 4,
    eval_neg_per_pos: int = 20,
    min_candidates_per_group: int = 10,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Build the final training/evaluation dataset with labels and hard negatives.

    Hard negative sampling is applied to ALL splits but with different
    group sizes:
      - Train: neg_per_pos negatives (small groups for focused learning)
      - Valid/Test: eval_neg_per_pos negatives (larger groups for realistic
        HitRate@k and NDCG@k evaluation)
    """
    labeled = assign_labels_and_splits(
        pair_base, positive_pairs, event_time_by_mentee, train_end, valid_end
    )

    # ── Remove cross-split label contamination ──
    # If a (mentee, mentor) pair is positive in BOTH train and test/valid,
    # REMOVE IT FROM BOTH SPLITS to prevent data leakage entirely.
    # (Not just relabel train positives — actually exclude from eval too)
    if isinstance(positive_pairs, dict):
        train_pos = positive_pairs.get("train", set())
        eval_pos = positive_pairs.get("valid", set()) | positive_pairs.get("test", set())
        overlap = train_pos & eval_pos
        if overlap:
            logger.warning(
                "build_recommendation_dataset: REMOVING %d (mentee, mentor) pairs "
                "that appear in BOTH train and eval splits (aggressive contamination fix).",
                len(overlap),
            )
            pair_keys_series = pd.Series(
                list(zip(labeled["mentee_id"].astype(int), labeled["mentor_id"].astype(int))),
                index=labeled.index
            )
            overlap_mask = pair_keys_series.isin(overlap)
            
            # Remove from TRAIN: relabel positives to 0
            train_mask = labeled["time_split"] == "train"
            train_remove_mask = train_mask & overlap_mask & (labeled["label"] == 1)
            train_removed_count = int(train_remove_mask.sum())
            labeled.loc[train_remove_mask, "label"] = 0
            
            # Remove from EVAL: drop contaminated rows entirely
            eval_mask = labeled["time_split"].isin(["valid", "test"])
            eval_remove_mask = eval_mask & overlap_mask & (labeled["label"] == 1)
            eval_removed_count = int(eval_remove_mask.sum())
            labeled = labeled[~eval_remove_mask].reset_index(drop=True)
            
            logger.info(
                "build_recommendation_dataset: removed %d from train, %d from eval (total %d pairs cleaned)",
                train_removed_count, eval_removed_count, train_removed_count + eval_removed_count,
            )
            remaining_overlap_count = int(
                (
                    (labeled["time_split"] == "train")
                    & (labeled["label"] == 1)
                    & pd.Series(
                        [tuple(k) in overlap for k in zip(labeled["mentee_id"].astype(int), labeled["mentor_id"].astype(int))],
                        index=labeled.index,
                    )
                ).sum()
            )
            logger.info(
                "build_recommendation_dataset: remaining train/eval overlap positives = %d",
                remaining_overlap_count,
            )

    # ── Train: hard negative sampling (with domain negatives) ──
    # Include domain-level negatives so the model learns to rank positives
    # above "same domain but different skills" candidates — matching eval.
    train_data = labeled[labeled["time_split"] == "train"]
    sampled_parts = []
    skipped_groups = 0

    for _, group_df in train_data.groupby("mentee_id"):
        sampled = sample_hard_negatives_per_group(
            group_df,
            neg_per_pos=neg_per_pos,
            min_candidates_per_group=min_candidates_per_group,
            rng_seed=rng_seed,
            include_domain_negatives=True,
        )
        if sampled is not None:
            sampled_parts.append(sampled)
        else:
            skipped_groups += 1

    if not sampled_parts:
        raise ValueError("No sampled training rows — check candidate pool and positive pairs")

    train_out = pd.concat(sampled_parts, ignore_index=True)
    train_out["hardness_score"] = train_out["hardness_score"].fillna(0)

    # ── Valid/Test: use domain-level negatives for meaningful HitRate@k ──
    # Include same-domain negatives (mentor_domain_match > 0) alongside
    # skill/subdomain negatives.  These "could have chosen but didn't"
    # negatives create realistic eval difficulty without crushing NDCG.
    eval_data_raw = labeled[labeled["time_split"] != "train"]
    eval_parts: list[pd.DataFrame] = []
    eval_skipped = 0
    for _, group_df in eval_data_raw.groupby("mentee_id"):
        sampled = sample_hard_negatives_per_group(
            group_df,
            neg_per_pos=eval_neg_per_pos,
            min_candidates_per_group=min_candidates_per_group,
            rng_seed=rng_seed + 7,  # different seed from train for independence
            include_domain_negatives=True,
        )
        if sampled is not None:
            eval_parts.append(sampled)
        else:
            eval_skipped += 1
    if eval_parts:
        eval_data = pd.concat(eval_parts, ignore_index=True)
        eval_data["hardness_score"] = eval_data["hardness_score"].fillna(0)
    else:
        eval_data = eval_data_raw.copy()
        eval_data["hardness_score"] = 0.0
    if eval_skipped:
        logger.warning(
            "build_recommendation_dataset: skipped %d eval groups (no negatives or positives)",
            eval_skipped,
        )

    out = pd.concat([train_out, eval_data], ignore_index=True)

    # Validate and log
    for split_name in ("valid", "test"):
        split_df = out[out["time_split"] == split_name]
        split_pos = split_df["label"].sum()
        n_groups = split_df["mentee_id"].nunique()
        median_group = split_df.groupby("mentee_id").size().median() if n_groups > 0 else 0
        if split_pos == 0:
            logger.warning(
                "build_recommendation_dataset: %s has no positive labels!", split_name
            )
        logger.info(
            "build_recommendation_dataset: %s — %d rows, %d positives, "
            "%d groups, median_group_size=%.0f",
            split_name, len(split_df), int(split_pos), n_groups, median_group,
        )

    if "time_split" in out.columns:
        split_counts = out.groupby("time_split")["label"].agg(["sum", "count"])
        split_counts["pos_ratio"] = (split_counts["sum"] / split_counts["count"]).round(4)
        logger.info("build_recommendation_dataset split stats:\n%s", split_counts.to_string())
    if skipped_groups > 0:
        logger.warning(
            "build_recommendation_dataset: skipped %d train groups (no negatives or positives)",
            skipped_groups,
        )

    return out.drop(columns=["event_time"], errors="ignore")

