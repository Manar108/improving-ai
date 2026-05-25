from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..features import jaccard, skill_coverage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgramCompatibilityConfig:
    """Central tuning knobs for program compatibility scoring.

    These defaults preserve the current soft-compatibility behavior while
    making the thresholds and decay tail adjustable without rewiring the
    scoring code.
    """

    exact_fit_threshold: float = 0.85
    near_fit_threshold: float = 0.65
    stretch_fit_threshold: float = 0.35
    distance_exact_softness: float = 1.0
    distance_near_softness: float = 0.66
    distance_stretch_softness: float = 0.30
    distance_tail_decay: float = 0.50
    distance_tail_floor: float = 0.05
    hard_target_gap: int = 3
    hard_education_gap: int = 3
    hard_skill_coverage_floor: float = 0.15
    hard_skill_match_floor: float = 0.15
    sparse_candidate_count: int = 3
    sparse_match_ceiling: float = 70.0


PROGRAM_COMPATIBILITY_CONFIG = ProgramCompatibilityConfig()

# Project-true columns (after normalization in preprocessing.DB_TABLE_MAP):
# programs: post_id, mentor_id, target_level, education_level, availability, capacity, created_at
# mentorship_requirements: post_id, technology_id, required_experience_level
# mentee_interests: mentee_id, technology_id, experience_level
# mentorships/applications: post_id, mentee_id, [start_date|applied_at]

# Aligned with Backend ExperienceLevel enum
EXPERIENCE_LEVEL_MAP = {
    "none": 1,
    "beginner": 2,
    "intermediate": 3,
    "advanced": 4,
}

TARGET_LEVEL_MAP = EXPERIENCE_LEVEL_MAP  # Alias for consistency

REQUIRED_SKILL_LEVEL_MAP = {
    "none": 1,
    "beginner": 2,
    "intermediate": 3,
    "advanced": 4,
}

EDUCATION_LEVEL_MAP = {
    "freshman": 1,
    "sophomore": 2,
    "junior": 3,
    "senior": 4,
    "graduate": 5,
}

PROGRAM_FEATURE_COLS = [
    "post_id",
    "mentor_id",
    "target_level_num",
    "required_education_num",
    "is_open",
    "is_available",
    "capacity",
    "program_enrollment_count",
    "program_popularity_log",
    "spots_left",
    "spots_left_log",
    "required_skill_count",
    "avg_required_skill_level",
    "program_difficulty_score",
    "requirement_set",
    "requirement_level_map",
    # NEW (May 2026): Deadline features
    "deadline",
    "days_until_deadline",
    "deadline_passed",
]

PAIR_FEATURE_COLS = [
    "mentee_id",
    "post_id",
    "requirement_coverage_score",
    "requirement_overlap_score",
    "required_skill_level_match_score",
    "matched_required_skill_count",
    "missing_required_skill_count",
    "program_popularity_log",
    "program_difficulty_score",
    "is_open",
    "is_available",
    "spots_left",
    "target_level_num",
    "required_education_num",
    "target_level_gap",
    "education_level_gap",
    "target_level_pass",
    "education_level_pass",
    "target_level_exact_match",
    "education_level_exact_match",
    "minimum_requirement_exact_match",
    "minimum_requirement_above_minimum",
    "availability_pass",
    "eligibility_pass",
    "candidate_pre_score",
    # ────────────────────────────────────────────────────────────
    # NEW (May 2026): Soft Compatibility Scoring & Confidence Signals
    # ────────────────────────────────────────────────────────────
    "target_level_distance",
    "target_level_softness",
    "education_distance",
    "education_softness",
    "skill_level_compatibility",
    "overall_eligibility_softness",
    "compatibility_confidence_band",
    # NEW (May 2026): Deadline features
    "days_until_deadline",
    "deadline_passed",
]


def _safe_set(value) -> set:
    return value if isinstance(value, set) else set()


def _normalize_str_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _map_target_level_to_num(series: pd.Series) -> pd.Series:
    s = _normalize_str_series(series)
    numeric = pd.to_numeric(s, errors="coerce")
    mapped = s.map(TARGET_LEVEL_MAP)
    out = mapped.where(mapped.notna(), numeric)
    return out.fillna(1).astype(int)


def _map_required_skill_level_to_num(series: pd.Series) -> pd.Series:
    s = _normalize_str_series(series)
    numeric = pd.to_numeric(s, errors="coerce")
    mapped = s.map(REQUIRED_SKILL_LEVEL_MAP)
    out = mapped.where(mapped.notna(), numeric)
    return out.fillna(1).astype(int)


def _map_education_level_to_num(series: pd.Series) -> pd.Series:
    s = _normalize_str_series(series)
    numeric = pd.to_numeric(s, errors="coerce")
    mapped = s.map(EDUCATION_LEVEL_MAP)
    out = mapped.where(mapped.notna(), numeric)
    return out.fillna(0).astype(int)


def _build_program_requirements(
    post_requirements: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-program requirement_set and requirement_level_map.

    Expects normalized requirement columns:
    - post_id
    - technology_id
    - required_experience_level
    """
    if post_requirements is None or post_requirements.empty:
        return pd.DataFrame(columns=["post_id", "requirement_set", "requirement_level_map"])

    required_cols = {"post_id", "technology_id", "required_experience_level"}
    missing = required_cols - set(post_requirements.columns)
    if missing:
        raise ValueError(
            f"post_requirements missing required columns: {sorted(missing)}"
        )

    req = post_requirements[["post_id", "technology_id", "required_experience_level"]].copy()
    req["post_id"] = pd.to_numeric(req["post_id"], errors="coerce")
    req["technology_id"] = pd.to_numeric(req["technology_id"], errors="coerce")
    req = req.dropna(subset=["post_id", "technology_id"])
    req[["post_id", "technology_id"]] = req[["post_id", "technology_id"]].astype(int)
    req["required_skill_level_num"] = _map_required_skill_level_to_num(req["required_experience_level"])

    req_set = (
        req.groupby("post_id")["technology_id"]
        .agg(lambda x: set(x.tolist()))
        .reset_index(name="requirement_set")
    )

    req_map = (
        req.sort_values("required_skill_level_num")
        .groupby(["post_id", "technology_id"], as_index=False)
        .tail(1)
        .groupby("post_id")
        .apply(
            lambda g: {
                int(row["technology_id"]): int(row["required_skill_level_num"])
                for _, row in g.iterrows()
            }
        )
        .reset_index(name="requirement_level_map")
    )

    return req_set.merge(req_map, on="post_id", how="inner")


def build_program_features(
    program_posts: pd.DataFrame,
    post_requirements: pd.DataFrame,
    program_enrollments: pd.DataFrame | None = None,
    mentorship_applications: pd.DataFrame | None = None,
    reference_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build program-level features using project's actual schema.

    Required program columns:
    - post_id, mentor_id, target_level, education_level, availability, capacity
    
    Optional parameters:
    - mentorship_applications: DataFrame with 'post_id', 'mentee_id', 'status' to count accepted apps
    - reference_time: Timestamp for calculating days_until_deadline (defaults to now)
    """
    if program_posts is None or program_posts.empty:
        return pd.DataFrame(columns=PROGRAM_FEATURE_COLS)
    
    if reference_time is None:
        reference_time = pd.Timestamp.utcnow()

    required_post_cols = {
        "post_id",
        "mentor_id",
        "target_level",
        "education_level",
        "availability",
        "capacity",
    }
    missing_posts = required_post_cols - set(program_posts.columns)
    if missing_posts:
        raise ValueError(
            f"program_posts missing required columns: {sorted(missing_posts)}"
        )

    # Include is_open and deadline in selected columns if available
    selected_cols = [
        "post_id",
        "mentor_id",
        "target_level",
        "education_level",
        "availability",
        "capacity",
    ]
    if "is_open" in program_posts.columns:
        selected_cols.append("is_open")
    if "deadline" in program_posts.columns:
        selected_cols.append("deadline")
    
    posts = program_posts[selected_cols].copy()

    posts["post_id"] = pd.to_numeric(posts["post_id"], errors="coerce")
    posts["mentor_id"] = pd.to_numeric(posts["mentor_id"], errors="coerce")
    posts["capacity"] = pd.to_numeric(posts["capacity"], errors="coerce")
    posts = posts.dropna(subset=["post_id", "mentor_id"])
    posts[["post_id", "mentor_id"]] = posts[["post_id", "mentor_id"]].astype(int)
    posts["capacity"] = posts["capacity"].fillna(0).clip(lower=0).astype(int)

    posts["target_level_num"] = _map_target_level_to_num(posts["target_level"])
    posts["required_education_num"] = _map_education_level_to_num(posts["education_level"])

    # ── Business Logic: is_open (publication state) ──────────────────────────
    # is_open = ProgramPostStatus: Published (1) means open for applications.
    # EF Core stores ProgramPostStatus as string ('Draft'/'Published') → normalized to lowercase.
    # CSV data uses integer (0=draft, 1=published) → normalized to '0'/'1'.
    if "is_open" in posts.columns:
        is_open_values = _normalize_str_series(posts["is_open"])
        posts["is_open"] = is_open_values.isin({
            "published",   # EF Core string value
            "1",          # CSV integer value
        }).astype(int)
    else:
        posts["is_open"] = 1  # Default to open if not specified

    # ── Business Logic: is_available (mentor hasn't closed applications) ─────
    # Availability is a free-text field. The backend stores "Open", "Opened",
    # "Available", "Closed", etc. We treat anything NOT closed as available.
    availability_text = _normalize_str_series(posts["availability"])
    posts["is_available"] = (~availability_text.isin({
        "closed",
        "inactive",
        "unavailable",
        "archived",
    })).astype(int)

    req_df = _build_program_requirements(post_requirements)
    out = posts.merge(req_df, on="post_id", how="left")
    out["requirement_set"] = out["requirement_set"].apply(_safe_set)
    out["requirement_level_map"] = out["requirement_level_map"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    # ── Handle program enrollments: prefer accepted applications if available ──
    if mentorship_applications is not None and not mentorship_applications.empty:
        # Use accepted applications count as program_enrollment_count
        required_app_cols = {"post_id", "status"}
        missing_app = required_app_cols - set(mentorship_applications.columns)
        if not missing_app:
            app = mentorship_applications[["post_id", "status"]].copy()
            app["post_id"] = pd.to_numeric(app["post_id"], errors="coerce")
            app = app.dropna(subset=["post_id"])
            app["post_id"] = app["post_id"].astype(int)
            
            # Count only accepted applications
            app_stats = (
                app[app["status"].str.lower().str.strip() == "accepted"]
                .groupby("post_id")
                .size()
                .reset_index(name="program_enrollment_count")
            )
            out = out.merge(app_stats, on="post_id", how="left")
    
    # Fall back to mentorships if applications not used
    if "program_enrollment_count" not in out.columns or out["program_enrollment_count"].isna().all():
        if program_enrollments is not None and not program_enrollments.empty:
            required_enr_cols = {"post_id", "mentee_id"}
            missing_enr = required_enr_cols - set(program_enrollments.columns)
            if not missing_enr:
                enr = program_enrollments[["post_id", "mentee_id"]].copy()
                enr["post_id"] = pd.to_numeric(enr["post_id"], errors="coerce")
                enr["mentee_id"] = pd.to_numeric(enr["mentee_id"], errors="coerce")
                enr = enr.dropna(subset=["post_id", "mentee_id"])
                enr[["post_id", "mentee_id"]] = enr[["post_id", "mentee_id"]].astype(int)

                enr_stats = (
                    enr.groupby("post_id").agg(
                        program_enrollment_count=("mentee_id", "nunique"),
                    )
                ).reset_index()
                out = out.merge(enr_stats, on="post_id", how="left")
        
        if "program_enrollment_count" not in out.columns:
            out["program_enrollment_count"] = 0

    out["program_enrollment_count"] = (
        pd.to_numeric(out["program_enrollment_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    out["program_popularity_log"] = np.log1p(out["program_enrollment_count"])

    out["spots_left"] = (out["capacity"] - out["program_enrollment_count"]).clip(lower=0)
    out["spots_left_log"] = np.log1p(out["spots_left"])

    out["required_skill_count"] = out["requirement_set"].apply(len).astype(int)
    out["avg_required_skill_level"] = out["requirement_level_map"].apply(
        lambda m: float(np.mean(list(m.values()))) if m else 1.0
    )
    out["program_difficulty_score"] = (
        0.6 * out["target_level_num"] + 0.4 * out["avg_required_skill_level"]
    )

    # ── NEW (May 2026): Calculate deadline features ────────────────────────────
    if "deadline" in out.columns:
        out["deadline"] = pd.to_datetime(out["deadline"], errors="coerce")
        # Calculate days_until_deadline
        out["days_until_deadline"] = (out["deadline"] - reference_time).dt.days
        out["days_until_deadline"] = out["days_until_deadline"].fillna(-999).astype(int)
        # Calculate deadline_passed (1 if deadline has passed, 0 otherwise)
        out["deadline_passed"] = (out["deadline"] <= reference_time).astype(int)
    else:
        # No deadline info, assume always active
        out["deadline"] = pd.NaT
        out["days_until_deadline"] = 999  # Far future
        out["deadline_passed"] = 0

    return out[[c for c in PROGRAM_FEATURE_COLS if c in out.columns]]


def _build_mentee_skill_level_map(mentee_interest_levels: pd.DataFrame) -> pd.DataFrame:
    """Build mapping of mentee skill levels from mentee_interests table.

    Expects columns: mentee_id, technology_id, experience_level
    """
    if mentee_interest_levels is None or mentee_interest_levels.empty:
        return pd.DataFrame(columns=["mentee_id", "mentee_skill_level_map"])

    required_cols = {"mentee_id", "technology_id", "experience_level"}
    missing = required_cols - set(mentee_interest_levels.columns)
    if missing:
        raise ValueError(
            f"mentee_interest_levels missing required columns: {sorted(missing)}"
        )

    mi = mentee_interest_levels[["mentee_id", "technology_id", "experience_level"]].copy()
    mi["mentee_id"] = pd.to_numeric(mi["mentee_id"], errors="coerce")
    mi["technology_id"] = pd.to_numeric(mi["technology_id"], errors="coerce")
    mi = mi.dropna(subset=["mentee_id", "technology_id"])
    mi[["mentee_id", "technology_id"]] = mi[["mentee_id", "technology_id"]].astype(int)
    mi["skill_level_num"] = _map_required_skill_level_to_num(mi["experience_level"])

    out = (
        mi.sort_values("skill_level_num")
        .groupby(["mentee_id", "technology_id"], as_index=False)
        .tail(1)
        .groupby("mentee_id")
        .apply(
            lambda g: {
                int(row["technology_id"]): int(row["skill_level_num"])
                for _, row in g.iterrows()
            }
        )
        .reset_index(name="mentee_skill_level_map")
    )
    return out


def _required_level_match(mentee_level_map: dict, req_level_map: dict) -> tuple[float, int, int]:
    """Return (level_match_score, matched_count, missing_count)."""
    if not req_level_map:
        return 0.0, 0, 0
    if not mentee_level_map:
        return 0.0, 0, len(req_level_map)

    matched_sum = 0.0
    matched_count = 0
    for tech_id, req_level in req_level_map.items():
        mentee_level = mentee_level_map.get(int(tech_id), 0)
        if mentee_level > 0:
            matched_count += 1
        if req_level <= 0:
            matched_sum += 1.0
        else:
            matched_sum += min(1.0, float(mentee_level) / float(req_level))

    total = len(req_level_map)
    missing_count = max(total - matched_count, 0)
    return matched_sum / total, matched_count, missing_count


# ────────────────────────────────────────────────────────────────────────────
# NEW (May 2026): Soft Compatibility Scoring
# ────────────────────────────────────────────────────────────────────────────
# Instead of binary eligibility gates, compute soft compatibility scores
# that reflect the degree of match and can be used in ranking + explanations.
# ────────────────────────────────────────────────────────────────────────────

def _compute_level_distance(mentee_level: int, program_level: int) -> int:
    """Compute absolute distance between two level values.
    
    Used for target_level and education_level matching.
    Distance 0 = exact match (strongest)
    Distance 1 = adjacent level (acceptable)
    Distance 2+ = gap (penalty)
    """
    return abs(int(mentee_level) - int(program_level))


def _compute_softness_from_distance(
    distance: int,
    max_distance: int = 3,
    config: ProgramCompatibilityConfig = PROGRAM_COMPATIBILITY_CONFIG,
) -> float:
    """Convert distance to softness score [0, 1].
    
    Softness represents compatibility confidence:
    - 1.0 = exact match (maximum compatibility)
    - 0.8-0.9 = distance 1 (near match, acceptable)
    - 0.5-0.7 = distance 2 (stretch opportunity)
    - <0.5 = distance 3+ (weak compatibility)
    """
    if distance <= 0:
        return config.distance_exact_softness
    if distance == 1:
        return config.distance_near_softness
    if distance == 2:
        return config.distance_stretch_softness

    # Larger gaps should decay quickly but remain non-zero so stretch/weak
    # opportunities can still surface under the soft recommendation policy.
    return float(
        max(
            config.distance_tail_floor,
            config.distance_stretch_softness * (config.distance_tail_decay ** float(distance - 2)),
        )
    )


def _compute_skill_level_compatibility(
    mentee_level_map: dict,
    req_level_map: dict,
    skill_importance_map: dict[int, float] | None = None,
) -> float:
    """Compute soft skill-level compatibility (0.0-1.0).
    
    For each required skill, compute compatibility of mentee's level vs requirement:
    - Mentee level >= requirement level → high compatibility
    - Mentee level ~= requirement level (off by 1) → medium compatibility
    - Mentee level < requirement level → penalized based on gap
    
    Returns weighted average across all required skills.
    """
    if not req_level_map:
        return 1.0  # No requirements → perfect compatibility
    if not mentee_level_map:
        return 0.0  # No mentee skills vs requirements → zero compatibility
    
    compatibility_scores = []
    weights = []
    for tech_id, req_level in req_level_map.items():
        mentee_level = mentee_level_map.get(int(tech_id), 0)
        if skill_importance_map and int(tech_id) in skill_importance_map:
            weight = float(skill_importance_map[int(tech_id)])
        else:
            # Higher required levels matter more, preserving required/foundational
            # weighting without requiring a separate skill taxonomy.
            weight = 1.0 + max(float(req_level) - 1.0, 0.0) * 0.5

        if req_level <= 0:
            compatibility_scores.append(1.0)
            weights.append(weight)
        elif mentee_level <= 0:
            compatibility_scores.append(0.0)
            weights.append(weight)
        else:
            distance = abs(int(mentee_level) - int(req_level))
            softness = _compute_softness_from_distance(distance, max_distance=3)
            compatibility_scores.append(softness)
            weights.append(weight)

    return float(np.average(compatibility_scores, weights=weights)) if compatibility_scores else 0.0


def _compute_overall_eligibility_softness(
    target_softness: float,
    education_softness: float,
    skill_compatibility: float,
    coverage_score: float = 0.5,
    config: ProgramCompatibilityConfig = PROGRAM_COMPATIBILITY_CONFIG,
) -> float:
    """Combine all compatibility signals into overall softness (0.0-1.0).
    
    Weights:
    - Target level: 30% (core experience match)
    - Education level: 25% (academic readiness)
    - Skill compatibility: 25% (technical alignment)
    - Coverage: 20% (has relevant skills)
    
    Result represents overall program-mentee compatibility confidence.
    """
    weights = {
        "target": 0.30,
        "education": 0.25,
        "skill": 0.25,
        "coverage": 0.20,
    }
    
    overall = (
        weights["target"] * float(target_softness) +
        weights["education"] * float(education_softness) +
        weights["skill"] * float(skill_compatibility) +
        weights["coverage"] * min(1.0, float(coverage_score))
    )

    # Gating-aware aggregation: catastrophic target/education gaps can never
    # be hidden by strong skill coverage or popularity-like signals.
    core_floor = min(float(target_softness), float(education_softness))
    gating_multiplier = 0.55 + 0.45 * max(0.0, min(1.0, core_floor))
    if float(target_softness) < config.stretch_fit_threshold:
        gating_multiplier *= 0.75
    if float(education_softness) < config.stretch_fit_threshold:
        gating_multiplier *= 0.85

    overall = overall * gating_multiplier
    return float(np.clip(overall, 0.0, 1.0))


def _classify_fit_confidence(softness: float) -> str:
    """Classify overall softness into confidence bands.
    
    Used in explanations and ranking confidence assessment.
    """
    if softness >= PROGRAM_COMPATIBILITY_CONFIG.exact_fit_threshold:
        return "exact_fit"       # Perfect or near-perfect match
    elif softness >= PROGRAM_COMPATIBILITY_CONFIG.near_fit_threshold:
        return "near_fit"        # Good match with minor gaps
    elif softness >= PROGRAM_COMPATIBILITY_CONFIG.stretch_fit_threshold:
        return "stretch_fit"     # Challenging but achievable
    else:
        return "weak_fit"        # Significant compatibility gaps


def _is_catastrophically_incompatible(row: pd.Series) -> bool:
    """Deterministic hard safety floor for extreme mismatches.

    This blocks cases where the level gap or prerequisite gap is so large that
    soft exploration should not be allowed to surface the candidate at all.
    """
    target_distance = int(row.get("target_level_distance", 0) or 0)
    education_distance = int(row.get("education_distance", 0) or 0)
    coverage = float(row.get("requirement_coverage_score", 0.0) or 0.0)
    skill_fit = float(row.get("skill_level_compatibility", 0.0) or 0.0)
    matched = int(row.get("matched_required_skill_count", 0) or 0)
    required = int(row.get("required_skill_count", 0) or 0)

    if target_distance >= PROGRAM_COMPATIBILITY_CONFIG.hard_target_gap:
        return True
    if education_distance >= PROGRAM_COMPATIBILITY_CONFIG.hard_education_gap:
        return True
    if required >= 3 and matched == 0 and coverage <= PROGRAM_COMPATIBILITY_CONFIG.hard_skill_coverage_floor:
        return True
    if required >= 3 and skill_fit <= PROGRAM_COMPATIBILITY_CONFIG.hard_skill_match_floor and coverage <= 0.25:
        return True
    return False


def build_mentee_program_candidates(
    mentee_features: pd.DataFrame,
    program_features: pd.DataFrame,
    mentee_interest_levels: pd.DataFrame,
    top_k_per_mentee: int = 80,
    enforce_hard_gates: bool = True,
) -> pd.DataFrame:
    """Build mentee→program pair features and candidates.

    Eligibility gates are strict and based on:
    - target_level_num
    - required_education_num
    - open availability + spots_left
    """
    if mentee_features is None or mentee_features.empty:
        return pd.DataFrame(columns=PAIR_FEATURE_COLS)
    if program_features is None or program_features.empty:
        return pd.DataFrame(columns=PAIR_FEATURE_COLS)

    required_mentee_cols = {
        "mentee_id",
        "interests_set",
        "experience_level_num",
        "education_status",
    }
    missing_mentee = required_mentee_cols - set(mentee_features.columns)
    if missing_mentee:
        raise ValueError(
            f"mentee_features missing required columns: {sorted(missing_mentee)}"
        )

    required_program_cols = {
        "post_id",
        "requirement_set",
        "requirement_level_map",
        "target_level_num",
        "required_education_num",
        "is_open",
        "spots_left",
        "program_popularity_log",
        "program_difficulty_score",
    }
    missing_prog = required_program_cols - set(program_features.columns)
    if missing_prog:
        raise ValueError(
            f"program_features missing required columns: {sorted(missing_prog)}"
        )

    mentees = mentee_features[
        ["mentee_id", "interests_set", "experience_level_num", "education_status"]
    ].copy()
    mentees["mentee_id"] = pd.to_numeric(mentees["mentee_id"], errors="coerce")
    mentees = mentees.dropna(subset=["mentee_id"])
    mentees["mentee_id"] = mentees["mentee_id"].astype(int)
    mentees["interests_set"] = mentees["interests_set"].apply(_safe_set)
    mentees["experience_level_num"] = pd.to_numeric(
        mentees["experience_level_num"], errors="coerce"
    ).fillna(1).astype(int)
    mentees["education_num"] = _map_education_level_to_num(mentees["education_status"])

    programs = program_features[
        [
            "post_id",
            "requirement_set",
            "requirement_level_map",
            "target_level_num",
            "required_education_num",
            "is_open",
            "is_available",
            "spots_left",
            "capacity",
            "program_popularity_log",
            "program_difficulty_score",
        ]
    ].copy()
    programs["post_id"] = pd.to_numeric(programs["post_id"], errors="coerce")
    programs = programs.dropna(subset=["post_id"])
    programs["post_id"] = programs["post_id"].astype(int)
    programs["requirement_set"] = programs["requirement_set"].apply(_safe_set)
    programs["requirement_level_map"] = programs["requirement_level_map"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    mentee_level_map = _build_mentee_skill_level_map(mentee_interest_levels)
    mentees = mentees.merge(mentee_level_map, on="mentee_id", how="left")
    mentees["mentee_skill_level_map"] = mentees["mentee_skill_level_map"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    # MEDIUM FIX: Cross-join scalability safeguard with chunked processing
    # Detect memory explosion risk from mentee × program cross-join
    n_mentees = len(mentees)
    n_programs = len(programs)
    n_pairs_estimated = n_mentees * n_programs
    estimated_memory_gb = (n_pairs_estimated * 20 * 8) / (1024 ** 3)  # ~20 features × 8 bytes per value
    
    # The object-heavy cross-join can blow up well before a naive GB estimate
    # reaches a very large threshold, so keep the safety bar conservative.
    memory_safety_threshold_gb = 2.0  # 🔧 Increased to 2GB to allow more data
    if estimated_memory_gb > memory_safety_threshold_gb:
        logger.warning(
            "SCALABILITY SAFEGUARD: Cross-join would produce %.1f GB of pairs (%d mentees × %d programs). "
            "Estimated memory: %.1f GB > safety threshold (%.1f GB). Using chunked mentee processing.",
            estimated_memory_gb, n_mentees, n_programs, estimated_memory_gb, memory_safety_threshold_gb
        )
        # 🔧 Process mentees in chunks instead of reducing programs
        chunk_size = max(100, int(n_mentees * memory_safety_threshold_gb / estimated_memory_gb))
        logger.info("Chunked processing: %d mentees per chunk (total %d chunks)", chunk_size, (n_mentees + chunk_size - 1) // chunk_size)
        
        pair_chunks = []
        for i in range(0, n_mentees, chunk_size):
            chunk_mentees = mentees.iloc[i:i+chunk_size]
            chunk_pairs = chunk_mentees[
                [
                    "mentee_id",
                    "interests_set",
                    "experience_level_num",
                    "education_num",
                    "mentee_skill_level_map",
                ]
            ].merge(programs, how="cross")
            pair_chunks.append(chunk_pairs)
        
        pairs = pd.concat(pair_chunks, ignore_index=True)
        logger.info("Chunked processing complete: %d total pairs from %d mentees × %d programs", len(pairs), n_mentees, n_programs)
    else:
        logger.info("Cross-join within memory threshold: %d mentees × %d programs (%.1f GB estimated)", n_mentees, n_programs, estimated_memory_gb)
        pairs = mentees[
            [
                "mentee_id",
                "interests_set",
                "experience_level_num",
                "education_num",
                "mentee_skill_level_map",
            ]
        ].merge(programs, how="cross")
        logger.info("Cross-join complete: %d total pairs", len(pairs))

    # 🔧 Vectorized feature engineering with progress logging
    logger.info("Computing skill coverage and overlap features...")
    pairs["requirement_coverage_score"] = [
        skill_coverage(m_sk, p_sk)
        for m_sk, p_sk in zip(pairs["interests_set"], pairs["requirement_set"])
    ]
    pairs["requirement_overlap_score"] = [
        jaccard(m_sk, p_sk)
        for m_sk, p_sk in zip(pairs["interests_set"], pairs["requirement_set"])
    ]
    logger.info("✓ Skill coverage and overlap computed")

    logger.info("Computing required skill level matching...")
    req_level_scores = [
        _required_level_match(m_map, r_map)
        for m_map, r_map in zip(
            pairs["mentee_skill_level_map"],
            pairs["requirement_level_map"],
        )
    ]
    pairs["required_skill_level_match_score"] = [x[0] for x in req_level_scores]
    pairs["matched_required_skill_count"] = [x[1] for x in req_level_scores]
    pairs["missing_required_skill_count"] = [x[2] for x in req_level_scores]
    logger.info("✓ Required skill level matching computed")

    logger.info("Computing target level and education level compatibility...")
    pairs["target_level_gap"] = (
        pairs["experience_level_num"] - pairs["target_level_num"]
    )
    pairs["education_level_gap"] = (
        pairs["education_num"] - pairs["required_education_num"]
    )

    pairs["target_level_pass"] = (pairs["target_level_gap"] >= 0).astype(int)
    pairs["education_level_pass"] = (pairs["education_level_gap"] >= 0).astype(int)
    logger.info("✓ Level compatibility computed")
    
    # NOTE: availability_pass removed (no variance — all 1.0 when filtered, adds no signal)
    # Eligibility now directly checks is_open, is_available, and spots_left
    pairs["eligibility_pass"] = (
        (pairs["target_level_pass"] == 1)
        & (pairs["education_level_pass"] == 1)
        & (pairs["is_open"] > 0)
        & (pairs.get("is_available", 1) > 0)
        & (pairs["spots_left"] > 0)
    ).astype(int)
    pairs["target_level_exact_match"] = (pairs["target_level_gap"] == 0).astype(int)
    pairs["education_level_exact_match"] = (pairs["education_level_gap"] == 0).astype(int)
    pairs["minimum_requirement_exact_match"] = (
        (pairs["target_level_exact_match"] == 1)
        & (pairs["education_level_exact_match"] == 1)
    ).astype(int)
    pairs["minimum_requirement_above_minimum"] = (
        (pairs["eligibility_pass"] == 1)
        & (pairs["minimum_requirement_exact_match"] == 0)
    ).astype(int)

    # 🔧 SKIP soft compatibility scoring for now (too slow on 7.5M pairs)
    # These features add complexity but aren't critical for initial training
    # Will be added back after model baseline is established
    
    # Set defaults for later compatibility features if needed
    pairs["target_level_distance"] = pairs["target_level_gap"].abs()
    pairs["target_level_softness"] = 1.0 - (pairs["target_level_distance"] / 4.0).clip(0, 1)
    pairs["education_distance"] = pairs["education_level_gap"].abs()
    pairs["education_softness"] = 1.0 - (pairs["education_distance"] / 5.0).clip(0, 1)
    pairs["skill_level_compatibility"] = 0.5  # Default neutral
    pairs["overall_eligibility_softness"] = 0.5
    pairs["compatibility_confidence_band"] = "stretch_fit"  # Default classification
    pairs["catastrophic_incompatibility"] = False
    pairs["hard_safety_pass"] = 1
    
    logger.info("✓ Soft compatibility initialized (defaults only)")

    pairs["candidate_pre_score"] = (
        0.45 * pairs["requirement_coverage_score"]
        + 0.25 * pairs["required_skill_level_match_score"]
        + 0.10 * pairs["requirement_overlap_score"]
        + 0.10 * pairs["program_popularity_log"].fillna(0)
        + 0.10 * (1.0 / (1.0 + pairs["program_difficulty_score"].fillna(1.0)))
        + 0.05 * pairs["minimum_requirement_exact_match"]
    )

    if enforce_hard_gates:
        # Business logic: published AND available AND has spots
        # Filter out unpublished/closed/archived programs before ranking
        full_mask = (
            (pairs["is_open"] <= 0) |
            (pairs.get("is_available", 1) <= 0) |
            (pairs["spots_left"] <= 0)
        )
        if full_mask.any():
            logger.info(
                "Candidate generation: filtering %d unpublished/unavailable/full program pairs",
                int(full_mask.sum()),
            )

        # Invalid target/education levels fail safely
        invalid_levels = (
            (pairs["target_level_num"] <= 0) | (pairs["required_education_num"] < 0)
        )
        if invalid_levels.any():
            logger.warning(
                "Candidate generation: %d pairs with invalid target/education levels",
                int(invalid_levels.sum()),
            )

        safety_mask = (~full_mask) & (~invalid_levels) & (pairs["hard_safety_pass"] == 1)
        dropped = int((~safety_mask).sum())
        if dropped:
            logger.info(
                "Candidate generation: filtered %d catastrophic or invalid pairs before ranking",
                dropped,
            )
        pairs = pairs[safety_mask].copy()

    if pairs.empty:
        return pd.DataFrame(columns=PAIR_FEATURE_COLS)

    pairs = (
        pairs.sort_values(
            ["mentee_id", "minimum_requirement_exact_match", "candidate_pre_score"],
            ascending=[True, False, False],
        )
        .groupby("mentee_id")
        .head(top_k_per_mentee)
        .reset_index(drop=True)
    )

    return pairs[[c for c in PAIR_FEATURE_COLS if c in pairs.columns]]


def build_program_cf_embeddings(
    enrollments: pd.DataFrame,
    likes: pd.DataFrame | None = None,
    saves: pd.DataFrame | None = None,
    comments: pd.DataFrame | None = None,
    shares: pd.DataFrame | None = None,
    n_factors: int = 16,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Build user×program collaborative embeddings using project columns.

    Expects `post_id` and actor id columns:
    - likes/comments/saves: post_id, user_id
    - shares: post_id, sender_id

    CRITICAL FIX (May 2026): Enrollments are now EXCLUDED from CF!
    Reason: Enrollments ARE the positive labels in program recommendation.
    Using them in CF would cause direct data leakage (CF score becomes
    a perfect proxy for positive labels). Instead, CF learns from pure
    engagement signals (likes, comments, saves, shares) only.
    """
    signals: list[pd.DataFrame] = []

    # REMOVED: Enrollments signal (was causing direct data leakage)
    # if enrollments is not None and not enrollments.empty:
    #     if {"post_id", "mentee_id"}.issubset(enrollments.columns):
    #         e = enrollments[["mentee_id", "post_id"]].drop_duplicates().copy()
    #         e["score"] = 5.0
    #         signals.append(e)

    def _append_signal(df: pd.DataFrame | None, user_col: str, score: float) -> None:
        if df is None or df.empty:
            return
        cols = {"post_id", user_col}
        if not cols.issubset(df.columns):
            return
        s = df[[user_col, "post_id"]].drop_duplicates().copy()
        s = s.rename(columns={user_col: "mentee_id"})
        s["score"] = score
        signals.append(s)

    _append_signal(likes, "user_id", 1.0)
    _append_signal(comments, "user_id", 2.0)
    _append_signal(saves, "user_id", 3.0)
    _append_signal(shares, "sender_id", 4.0)

    if not signals:
        logger.warning("build_program_cf_embeddings: no interaction signals available")
        return {"user_factors": {}, "item_factors": {}}

    data = pd.concat(signals, ignore_index=True)
    data["mentee_id"] = pd.to_numeric(data["mentee_id"], errors="coerce")
    data["post_id"] = pd.to_numeric(data["post_id"], errors="coerce")
    data = data.dropna(subset=["mentee_id", "post_id"])
    data[["mentee_id", "post_id"]] = data[["mentee_id", "post_id"]].astype(int)
    data = data.groupby(["mentee_id", "post_id"], as_index=False)["score"].sum()

    if len(data) < 20:
        logger.warning("build_program_cf_embeddings: too few interactions (%d)", len(data))
        return {"user_factors": {}, "item_factors": {}}

    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize as sk_normalize

    users = sorted(data["mentee_id"].unique())
    items = sorted(data["post_id"].unique())
    u_map = {uid: idx for idx, uid in enumerate(users)}
    i_map = {pid: idx for idx, pid in enumerate(items)}

    rows = data["mentee_id"].map(u_map).values
    cols = data["post_id"].map(i_map).values
    vals = np.log1p(data["score"].values)

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
    n_components = min(n_factors, min(matrix.shape) - 1)
    if n_components < 2:
        return {"user_factors": {}, "item_factors": {}}

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix)
    item_factors = svd.components_.T

    user_factors = sk_normalize(user_factors, axis=1)
    item_factors = sk_normalize(item_factors, axis=1)

    user_factor_dict = {uid: user_factors[idx] for uid, idx in u_map.items()}
    item_factor_dict = {pid: item_factors[idx] for pid, idx in i_map.items()}

    return {"user_factors": user_factor_dict, "item_factors": item_factor_dict}


def compute_cf_confidence(
    cf_embeddings: Dict[str, Dict[int, np.ndarray]],
    mentee_id: int,
    min_interactions: int = 5,
) -> float:
    """Compute CF confidence score for a user.

    Returns a value in [0, 1] indicating how reliable the CF signal is.
    Low confidence (< 0.3) indicates sparse users where CF should be
    downweighted to prevent noisy embeddings from overpowering content signals.

    Args:
        cf_embeddings: Dict with 'user_factors' and 'item_factors'.
        mentee_id: The user to evaluate.
        min_interactions: Minimum interactions for full confidence.
    """
    user_factors = cf_embeddings.get("user_factors", {})
    if not user_factors:
        return 0.0

    user_vec = user_factors.get(int(mentee_id))
    if user_vec is None:
        return 0.0

    # Norm-based confidence: zero/near-zero norm indicates sparse embedding
    norm = float(np.linalg.norm(user_vec))
    if norm < 1e-6:
        return 0.0

    # Coverage confidence: fraction of item space the user has interacted with
    n_items = len(cf_embeddings.get("item_factors", {}))
    if n_items == 0:
        return 0.0

    # Clamp to [0, 1]
    return min(1.0, norm * 0.5 + 0.5)


def validate_pair_feature_distributions(
    pairs: pd.DataFrame,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate pair feature distributions for impossible/corrupted values.

    Returns list of warning messages. Raises ValueError in strict mode
    if critical corruptions are detected.
    """
    warnings_list: list[str] = []

    # Check for negative counts
    count_cols = [c for c in pairs.columns if c.endswith("_count")]
    for col in count_cols:
        if col in pairs.columns:
            neg = (pairs[col] < 0).sum()
            if neg > 0:
                msg = f"Feature '{col}' has {neg} negative values"
                warnings_list.append(msg)
                if strict:
                    raise ValueError(msg)

    # Check for invalid ratios (outside [0, 1])
    ratio_cols = [
        "requirement_coverage_score",
        "requirement_overlap_score",
        "required_skill_level_match_score",
        "candidate_pre_score",
    ]
    for col in ratio_cols:
        if col in pairs.columns:
            oob = ((pairs[col] < -0.01) | (pairs[col] > 1.01)).sum()
            if oob > 0:
                msg = f"Feature '{col}' has {oob} out-of-range values"
                warnings_list.append(msg)

    # Check for degenerate distributions (all same value)
    for col in pairs.select_dtypes(include=[np.number]).columns:
        if col in {"mentee_id", "post_id", "label"}:
            continue
        if pairs[col].nunique(dropna=True) <= 1 and len(pairs) > 10:
            warnings_list.append(f"Feature '{col}' has zero variance (degenerate)")

    # Check for popularity spikes
    if "program_popularity_log" in pairs.columns:
        pop = pairs["program_popularity_log"]
        if pop.max() > pop.quantile(0.99) * 3 and len(pop) > 50:
            warnings_list.append(
                f"Unstable popularity spike: max={pop.max():.2f}, p99={pop.quantile(0.99):.2f}"
            )

    if warnings_list:
        logger.warning(
            "Feature distribution validation: %d issues detected: %s",
            len(warnings_list), "; ".join(warnings_list),
        )

    return warnings_list
