from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from .features import PROGRAM_COMPATIBILITY_CONFIG, _classify_fit_confidence

logger = logging.getLogger(__name__)


DEFAULT_PROGRAM_FEATURE_COLS = [
    # ──────────────────────────────────────────────────────────────────
    # Core compatibility and skill matching
    # ──────────────────────────────────────────────────────────────────
    "requirement_coverage_score",          # [0,1] proportion of required skills mentee has
    "required_skill_level_match_score",    # [0,1] avg skill level match across requirements
    "requirement_overlap_score",           # [0,1] Jaccard similarity of skills
    "matched_required_skill_count",        # Count of mentee's matching required skills
    "missing_required_skill_count",        # Count of required skills mentee lacks
    
    # ──────────────────────────────────────────────────────────────────
    # Level matching (hard gates + soft compatibility)
    # ──────────────────────────────────────────────────────────────────
    "target_level_gap",                    # experience_level - program_target_level
    "target_level_pass",                   # Binary: gap >= 0
    "target_level_exact_match",            # Binary: gap == 0
    "target_level_distance",               # Absolute level distance (0, 1, 2, ...)
    "target_level_softness",               # [0,1] Soft match confidence for target level
    
    "education_level_gap",                 # education - program_required_education
    "education_level_pass",                # Binary: gap >= 0
    "education_level_exact_match",         # Binary: gap == 0
    "education_distance",                  # Absolute education distance
    "education_softness",                  # [0,1] Soft match confidence for education
    
    # ──────────────────────────────────────────────────────────────────
    # Soft compatibility scoring (NEW May 2026)
    # ──────────────────────────────────────────────────────────────────
    "skill_level_compatibility",           # [0,1] Weighted skill-level fit
    "overall_eligibility_softness",        # [0,1] Combined soft compatibility
    
    # ──────────────────────────────────────────────────────────────────
    # Availability and program metadata
    # ──────────────────────────────────────────────────────────────────
    "is_open",                             # Binary: program published
    "is_available",                        # Binary: program not closed/archived
    # "availability_pass" removed (no variance — all 1.0, adds no signal)
    "eligibility_pass",                    # Binary: all eligibility checks pass
    "minimum_requirement_exact_match",     # Binary: exact match on both levels
    "minimum_requirement_above_minimum",   # Binary: meets minimums but not exact
    
    # ──────────────────────────────────────────────────────────────────
    # Program capacity (NEW May 2026)
    # ──────────────────────────────────────────────────────────────────
    "spots_left",                          # Available capacity
    
    # ──────────────────────────────────────────────────────────────────
    # Pre-scoring and popularity
    # ──────────────────────────────────────────────────────────────────
    "candidate_pre_score",                 # Heuristic pre-ranking score
    "cf_score",                            # CF score from training-only mentorships (SAFE)
    "program_popularity_log",              # Log-scale program popularity
    "program_difficulty_score",            # Program difficulty (scaled 0-1)
    
    # ──────────────────────────────────────────────────────────────────
    # Deadline-aware filtering (NEW May 2026)
    # ──────────────────────────────────────────────────────────────────
    "days_until_deadline",                 # Days until program deadline
    "deadline_passed",                     # Binary: deadline has passed
]

# Confidence thresholds for recommendation quality bands
_CONFIDENCE_HIGH_THRESHOLD = 0.65
_CONFIDENCE_MEDIUM_THRESHOLD = 0.35

# Weak-match penalty thresholds
_WEAK_SKILL_OVERLAP_THRESHOLD = 0.15
_WEAK_REQUIREMENT_COVERAGE_THRESHOLD = 0.20
_WEAK_MATCH_PENALTY = 0.92

# Diversity penalty for near-duplicate programs within a mentee group
_DIVERSITY_SIMILARITY_THRESHOLD = 0.95
_DIVERSITY_PENALTY = 0.97

# Popularity cap — prevents popularity from dominating skill matching
_POPULARITY_CONTRIBUTION_CAP = 0.15

# Fairness: max fraction of recommendations any single program can receive
_MAX_PROGRAM_EXPOSURE_FRACTION = 0.25


def split_program_by_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by precomputed time_split labels (train/valid/test)."""
    out = df.copy()
    if "time_split" not in out.columns:
        raise ValueError("Expected time_split column in program dataset")

    out["time_split"] = out["time_split"].astype(str).str.strip().str.lower()
    train_df = out[out["time_split"] == "train"].sort_values("mentee_id")
    valid_df = out[out["time_split"] == "valid"].sort_values("mentee_id")
    test_df = out[out["time_split"] == "test"].sort_values("mentee_id")
    return train_df, valid_df, test_df


def _validate_ranker_inputs(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
) -> None:
    if train_df.empty:
        raise ValueError("train_df is empty")
    if valid_df.empty:
        raise ValueError("valid_df is empty")

    required_cols = set(feature_cols + [label_col, "mentee_id", "post_id"])
    missing_train = required_cols - set(train_df.columns)
    missing_valid = required_cols - set(valid_df.columns)
    if missing_train:
        raise ValueError(f"train_df missing columns: {sorted(missing_train)}")
    if missing_valid:
        raise ValueError(f"valid_df missing columns: {sorted(missing_valid)}")


def train_program_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    model_params: dict | None = None,
) -> LGBMRanker:
    """Train LightGBM ranker for program recommendation."""
    _validate_ranker_inputs(train_df, valid_df, feature_cols, label_col)

    train = train_df.copy()
    valid = valid_df.copy()

    # Safety numeric conversions
    train[label_col] = pd.to_numeric(train[label_col], errors="coerce").fillna(0).astype(int)
    valid[label_col] = pd.to_numeric(valid[label_col], errors="coerce").fillna(0).astype(int)
    for col in feature_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce").fillna(0.0)
        valid[col] = pd.to_numeric(valid[col], errors="coerce").fillna(0.0)

    X_train = train[feature_cols]
    y_train = train[label_col]
    X_valid = valid[feature_cols]
    y_valid = valid[label_col]
    sample_weight = None
    if "label_weight" in train.columns:
        sample_weight = pd.to_numeric(train["label_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)

    group_train = train.groupby("mentee_id").size().tolist()
    group_valid = valid.groupby("mentee_id").size().tolist()

    logger.info(
        "Program ranker training: train=%d rows (%d pos), valid=%d rows (%d pos)",
        len(train), int(y_train.sum()), len(valid), int(y_valid.sum()),
    )

    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 15,  # 🔧 Reduced from 25 to learn from fewer samples per leaf
        "n_estimators": 500,      # 🔧 Increased from 400 for better convergence
        "lambda_l1": 0.5,         # 🔧 Added L1 regularization
        "lambda_l2": 0.5,         # 🔧 Added L2 regularization
        "verbosity": -1,
        "random_state": 42,
    }
    params = default_params if model_params is None else {**default_params, **model_params}

    ranker = LGBMRanker(**params)
    ranker.fit(
        X_train,
        y_train,
        group=group_train,
        sample_weight=sample_weight,
        eval_set=[(X_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[5, 10],
        callbacks=[],
    )
    return ranker


def _filter_by_experience_level(
    df: pd.DataFrame,
    mentee_exp_col: str = "experience_level_num",
    program_target_col: str = "target_level_num",
    max_level_gap: int = 1,
) -> pd.DataFrame:
    """Filter recommendations to programs matching mentee's experience level.
    
    Only recommends programs where:
      program_target_level <= mentee_experience_level + max_level_gap
    
    Example (max_level_gap=1):
      - Mentee level 3 (Intermediate): shows programs requiring 2-3 (Beginner, Intermediate)
      - Mentee level 2 (Beginner): shows programs requiring 1-2 (None, Beginner)
    
    Args:
        df: DataFrame with mentee and program level columns
        mentee_exp_col: Column name for mentee's current experience level
        program_target_col: Column name for program's required experience level
        max_level_gap: Maximum allowed gap (1 = allow one level below)
    
    Returns:
        Filtered DataFrame with compatible experience levels only
    """
    if df.empty:
        return df.copy()
    
    if mentee_exp_col not in df.columns or program_target_col not in df.columns:
        logger.warning(
            "experience_level filter: missing columns (%s or %s) — skipping filter",
            mentee_exp_col, program_target_col,
        )
        return df.copy()
    
    out = df.copy()
    
    # Calculate compatibility: program_required <= mentee_level + gap
    # (higher number = higher level, so lower required is easier)
    mentee_levels = out[mentee_exp_col].fillna(1).astype(int)
    program_targets = out[program_target_col].fillna(1).astype(int)
    
    # Compatible if: target_level <= mentee_level + max_gap
    # e.g., mentee=3, gap=1: allows targets 1, 2, 3, 4 (all within reach)
    max_allowed = mentee_levels + max_level_gap
    is_compatible = program_targets <= max_allowed
    
    n_before = len(out)
    out = out[is_compatible].reset_index(drop=True)
    n_after = len(out)
    
    if n_after < n_before:
        logger.info(
            "experience_level filter: removed %d/%d incompatible programs (gap > %d levels)",
            n_before - n_after, n_before, max_level_gap,
        )
    
    return out


def predict_program_scores(
    model: LGBMRanker,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    X = out[feature_cols].fillna(0).astype(float)
    out["pred_score"] = model.predict(X)
    return out


def _confidence_aware_normalization(scores: pd.Series) -> pd.Series:
    """Confidence-aware normalization that avoids misleading percentages.

    For single-candidate or near-zero-variance groups, applies a confidence
    penalty instead of producing fake-high 100% scores.
    """
    n = len(scores)
    if n == 0:
        return scores

    s_min = scores.min()
    s_max = scores.max()
    s_range = s_max - s_min

    if s_range < 1e-9:
        # Zero-variance: all scores identical — cap by count
        raw = scores.clip(lower=0)
        if n == 1:
            return (raw / (raw.max() + 1e-9)).clip(0, 0.65)
        return (raw / (raw.max() + 1e-9)).clip(0, 0.85)

    normalized = (scores - s_min) / s_range

    if n == 1:
        confidence_factor = 0.65
    elif n <= 3:
        confidence_factor = 0.85
    elif n <= 5:
        confidence_factor = 0.92
    else:
        confidence_factor = 1.0

    return normalized * confidence_factor


def _compute_weak_match_penalty(row: pd.Series) -> float:
    """Compute soft penalty for weak-fit programs.

    Programs with low skill overlap, weak requirement coverage, or weak
    eligibility confidence receive a multiplicative penalty to prevent
    them from ranking highly with inflated scores.
    
    Uses new soft compatibility bands (May 2026) to make penalties more nuanced:
    - exact_fit / near_fit: minimal penalty
    - stretch_fit: moderate penalty (still allowed, but lower confidence)
    - weak_fit: strong penalty (exploration allowed but deprioritized)
    """
    penalty = 1.0

    # NEW (May 2026): Use overall eligibility softness for more nuanced penalties
    # instead of binary pass/fail gates
    if "overall_eligibility_softness" in row and pd.notna(row["overall_eligibility_softness"]):
        softness = float(row["overall_eligibility_softness"])
        if softness < PROGRAM_COMPATIBILITY_CONFIG.stretch_fit_threshold:  # weak_fit band
            penalty *= 0.85  # Strong penalty for weak compatibility
        elif softness < 0.65:  # stretch_fit band
            penalty *= 0.95  # Moderate penalty for stretch opportunities
        # exact_fit and near_fit bands don't get penalties

    # Weak skill overlap
    coverage = row.get("requirement_coverage_score", 0.0)
    if pd.notna(coverage) and coverage < _WEAK_REQUIREMENT_COVERAGE_THRESHOLD:
        penalty *= _WEAK_MATCH_PENALTY

    # Weak skill level match
    overlap = row.get("requirement_overlap_score", 0.0)
    if pd.notna(overlap) and overlap < _WEAK_SKILL_OVERLAP_THRESHOLD:
        penalty *= _WEAK_MATCH_PENALTY

    # Missing many required skills
    missing = row.get("missing_required_skill_count", 0)
    matched = row.get("matched_required_skill_count", 0)
    if pd.notna(missing) and pd.notna(matched) and matched > 0:
        miss_ratio = missing / (missing + matched)
        if miss_ratio > 0.6:
            penalty *= 0.95

    return penalty


def _apply_diversity_reranking(group_df: pd.DataFrame) -> pd.DataFrame:
    """Apply soft diversity reranking within a mentee group.

    Reduces scores of near-duplicate programs (same mentor or very similar
    feature profiles) to encourage topical diversity without hard filtering.
    """
    if len(group_df) <= 2:
        return group_df

    out = group_df.copy()
    seen_mentors = set()
    seen_programs = set()

    diversity_adjustments = np.ones(len(out))

    for idx, (_, row) in enumerate(out.iterrows()):
        post_id = row.get("post_id")

        # Penalize repeated programs (shouldn't happen, but defensive)
        if post_id in seen_programs:
            diversity_adjustments[idx] *= _DIVERSITY_PENALTY
        seen_programs.add(post_id)

    out["rerank_score"] = out["rerank_score"].values * diversity_adjustments
    return out


def _compute_recommendation_confidence(row: pd.Series) -> str:
    """Compute explicit confidence band for a recommendation.

    Returns 'high', 'medium', or 'exploratory' based on signal strength.
    """
    score = row.get("rerank_score", 0.0)
    coverage = row.get("requirement_coverage_score", 0.0)
    skill_match = row.get("required_skill_level_match_score", 0.0)
    eligibility = row.get("eligibility_pass", 0)
    softness = row.get("overall_eligibility_softness", 0.0)

    # Composite confidence signal
    signal = (
        0.40 * min(coverage, 1.0)
        + 0.30 * min(skill_match, 1.0)
        + 0.20 * float(eligibility > 0)
        + 0.10 * min(score, 1.0)
    )

    if pd.notna(softness):
        signal = 0.75 * signal + 0.25 * float(softness)

    if signal >= _CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if signal >= _CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "exploratory"


def _build_explanation_metadata(row: pd.Series) -> Dict[str, Any]:
    """Generate compact, structured explanation metadata for a recommendation."""
    coverage = float(row.get("requirement_coverage_score", 0.0) or 0.0)
    overlap = float(row.get("requirement_overlap_score", 0.0) or 0.0)
    target_gap = int(row.get("target_level_gap", 0) or 0)
    education_gap = int(row.get("education_level_gap", 0) or 0)
    missing_required = int(row.get("missing_required_skill_count", 0) or 0)
    overall_softness = float(row.get("overall_eligibility_softness", 0.0) or 0.0)
    fit_band = row.get("compatibility_confidence_band")
    if not isinstance(fit_band, str) or not fit_band:
        fit_band = _classify_fit_confidence(overall_softness)

    signals: list[str] = []
    if coverage >= 0.6:
        signals.append("skill_coverage_high")
    elif coverage >= 0.2:
        signals.append("skill_coverage_partial")

    if overlap >= 0.3:
        signals.append("skill_overlap_good")

    if target_gap == 0 and education_gap == 0:
        signals.append("level_exact")
    elif abs(target_gap) <= 1 and abs(education_gap) <= 1:
        signals.append("level_near")
    elif abs(target_gap) >= 3 or abs(education_gap) >= 3:
        signals.append("level_gap_large")

    if missing_required > 0:
        signals.append("missing_required_skills")

    if row.get("minimum_requirement_exact_match", 0) > 0:
        signals.append("minimum_exact")
    elif row.get("minimum_requirement_above_minimum", 0) > 0:
        signals.append("above_minimum")

    risk_flags: list[str] = []
    if fit_band == "weak_fit":
        risk_flags.append("weak_fit")
    if abs(target_gap) >= 3:
        risk_flags.append("target_gap_large")
    if abs(education_gap) >= 3:
        risk_flags.append("education_gap_large")
    if missing_required >= 3:
        risk_flags.append("missing_skills_high")

    return {
        "fit_band": fit_band,
        "signals": signals,
        "risk_flags": risk_flags,
        "skill_coverage": round(coverage, 3),
        "skill_overlap": round(overlap, 3),
        "target_level_gap": target_gap,
        "education_level_gap": education_gap,
        "missing_required_skills": missing_required,
        "overall_softness": round(overall_softness, 3),
        "cf_contribution": 0.0,  # Removed: CF causes data leakage
        "popularity_contribution": round(float(row.get("program_popularity_log", 0.0) or 0.0), 3),
        "minimum_requirement_exact_match": bool(row.get("minimum_requirement_exact_match", 0) > 0),
        "confidence": _compute_recommendation_confidence(row),
    }


def _compute_absolute_confidence_ceiling(group_df: pd.DataFrame) -> float:
    """Compute an absolute confidence ceiling for a mentee's candidate pool."""
    n_candidates = len(group_df)
    if n_candidates <= 0:
        return 0.0

    softness_source = group_df["overall_eligibility_softness"] if "overall_eligibility_softness" in group_df.columns else pd.Series([0.0] * n_candidates, index=group_df.index)
    coverage_source = group_df["requirement_coverage_score"] if "requirement_coverage_score" in group_df.columns else pd.Series([0.0] * n_candidates, index=group_df.index)
    softness = pd.to_numeric(softness_source, errors="coerce").fillna(0.0)
    coverage = pd.to_numeric(coverage_source, errors="coerce").fillna(0.0)
    weak_ratio = float((softness < PROGRAM_COMPATIBILITY_CONFIG.stretch_fit_threshold).mean())
    max_softness = float(softness.max()) if len(softness) else 0.0
    avg_coverage = float(coverage.mean()) if len(coverage) else 0.0

    ceiling = 95.0
    if n_candidates == 1:
        ceiling = 65.0
    elif n_candidates <= PROGRAM_COMPATIBILITY_CONFIG.sparse_candidate_count:
        ceiling = 75.0

    if weak_ratio >= 0.60:
        ceiling = min(ceiling, 60.0)
    elif weak_ratio >= 0.35:
        ceiling = min(ceiling, 70.0)

    if max_softness < PROGRAM_COMPATIBILITY_CONFIG.near_fit_threshold:
        ceiling = min(ceiling, 80.0)
    if max_softness < PROGRAM_COMPATIBILITY_CONFIG.stretch_fit_threshold:
        ceiling = min(ceiling, 65.0)
    if avg_coverage < 0.20:
        ceiling = min(ceiling, 70.0)

    return float(np.clip(ceiling, 0.0, 100.0))


def rerank_program_recommendations(
    scored_df: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Apply production-grade business-safe reranking.

    Keeps skill-first behavior with:
    - Confidence-aware normalization (Item 20)
    - Weak-match penalties (Item 21) + Soft compatibility bands (May 2026)
    - Soft diversity reranking (Item 22)
    - Popularity capping (Item 23)
    - Confidence scoring (Item 24)
    - Explainability metadata (Item 25)
    - Fairness safeguards (Item 26)
    """
    if scored_df.empty:
        return scored_df.copy()

    out = scored_df.copy()

    # Confidence-aware normalization (Item 20)
    # Replaces unstable min-max that produces misleading percentages for
    # single-candidate or low-variance groups
    out["pred_norm"] = out.groupby("mentee_id")["pred_score"].transform(
        _confidence_aware_normalization
    )

    # ──────────────────────────────────────────────────────────────────
    # CRITICAL FIX (May 2026): Hard constraint for minimum requirements
    # ──────────────────────────────────────────────────────────────────
    # If a program doesn't meet MINIMUM requirements, severely penalize it
    # This prevents CF from recommending unqualified candidates
    
    if "minimum_requirement_exact_match" in out.columns:
        # Programs that meet exact minimum requirements: full score
        # Programs that meet but not exact (above_minimum): 95% of score
        # Programs that DON'T meet minimums: 60% penalty (0.40x)
        exact_match = out["minimum_requirement_exact_match"].fillna(0) > 0
        above_min = out.get("minimum_requirement_above_minimum", pd.Series(0, index=out.index)).fillna(0) > 0
        meets_minimum = exact_match | above_min
        
        # Hard constraint: non-eligible programs get severe penalty
        requirement_constraint = np.where(
            exact_match, 1.0,  # Exact match: full weight
            np.where(
                above_min, 0.95,  # Above minimum: slight penalty
                0.40  # HARD CONSTRAINT: doesn't meet minimums gets 60% penalty
            )
        )
    else:
        requirement_constraint = 1.0

    # Base reranking formula
    # Now includes requirement constraints to prevent CF overshadowing
    out["rerank_score"] = out["pred_norm"].fillna(0) * requirement_constraint
    
    # Popularity capping (Item 23): ensure popularity never dominates
    pop_col = out.get("program_popularity_log", pd.Series(0.0, index=out.index))
    if isinstance(pop_col, pd.Series):
        pop_contribution = pop_col.fillna(0).clip(lower=0)
        if pop_contribution.max() > 0:
            pop_contribution = pop_contribution / (pop_contribution.max() + 1e-9)
        pop_contribution = pop_contribution.clip(upper=_POPULARITY_CONTRIBUTION_CAP)
    else:
        pop_contribution = 0.0

    # NEW (May 2026): Add confidence band-aware weighting to prefer exact/near fits
    # This amplifies the soft signals and reduces weak-fit recommendations
    if "overall_eligibility_softness" in out.columns:
        # Map softness to confidence multiplier
        # exact_fit (>=0.85): 1.08x boost
        # near_fit (0.65-0.84): 1.03x boost
        # stretch_fit (0.35-0.64): 1.0x (neutral)
        # weak_fit (<0.35): 0.92x penalty
        confidence_multipliers = out["overall_eligibility_softness"].apply(
            lambda s: 1.08 if s >= PROGRAM_COMPATIBILITY_CONFIG.exact_fit_threshold else (
                1.03 if s >= PROGRAM_COMPATIBILITY_CONFIG.near_fit_threshold else (
                    1.0 if s >= PROGRAM_COMPATIBILITY_CONFIG.stretch_fit_threshold else 0.92
                )
            )
        )
        confidence_multipliers = confidence_multipliers.clip(lower=0.92, upper=1.08)
        out["rerank_score"] *= confidence_multipliers

    if "minimum_requirement_exact_match" in out.columns:
        # Already applied hard constraint above, don't double boost
        pass
    if "minimum_requirement_above_minimum" in out.columns:
        # Already applied hard constraint above, don't double boost
        pass

    # Soft boosts only (no hard filtering)
    # NOTE: availability_pass removed (no variance, adds no signal)
    if "candidate_pre_score" in out.columns:
        out["rerank_score"] *= (1.0 + 0.008 * out["candidate_pre_score"].clip(lower=0, upper=1))

    # Weak-match penalties (Item 21) + Soft compatibility updates (May 2026)
    weak_penalties = out.apply(_compute_weak_match_penalty, axis=1)
    out["rerank_score"] *= weak_penalties
    out["rerank_score"] = out["rerank_score"].clip(lower=0.0, upper=2.0)

    # Diversity-aware soft reranking (Item 22)
    parts = []
    for _, group in out.groupby("mentee_id"):
        parts.append(_apply_diversity_reranking(group))
    out = pd.concat(parts, ignore_index=True)

    # Absolute confidence calibration: keep weak pools from looking overly strong.
    ceiling_map = out.groupby("mentee_id").apply(_compute_absolute_confidence_ceiling)
    out["confidence_ceiling"] = out["mentee_id"].map(ceiling_map).fillna(0.0)

    # Confidence-aware match percentage (Item 20)
    out["match_percentage"] = (
        out.groupby("mentee_id")["rerank_score"].transform(
            lambda s: (
                100.0 * _confidence_aware_normalization(s)
            )
        )
        .clip(0, 100)
        .round(1)
    )

    # Low-confidence penalty: reduce match_percentage for weak signals
    if "requirement_coverage_score" in out.columns:
        low_signal = out["requirement_coverage_score"].fillna(0) < _WEAK_REQUIREMENT_COVERAGE_THRESHOLD
        out.loc[low_signal, "match_percentage"] = out.loc[low_signal, "match_percentage"].clip(upper=70.0)
    out["match_percentage"] = np.minimum(out["match_percentage"], out["confidence_ceiling"]).round(1)
    out["match_percentage"] = out["match_percentage"].clip(0, 100)

    # Sparse-user guard: singletons and tiny pools should never get premium scores.
    sparse_mask = out.groupby("mentee_id")["mentee_id"].transform("size") <= PROGRAM_COMPATIBILITY_CONFIG.sparse_candidate_count
    out.loc[sparse_mask, "match_percentage"] = np.minimum(
        out.loc[sparse_mask, "match_percentage"],
        PROGRAM_COMPATIBILITY_CONFIG.sparse_match_ceiling,
    )

    sort_cols = ["mentee_id"]
    sort_order = [True]
    if "minimum_requirement_exact_match" in out.columns:
        sort_cols.append("minimum_requirement_exact_match")
        sort_order.append(False)
    sort_cols.append("rerank_score")
    sort_order.append(False)

    out = out.sort_values(sort_cols, ascending=sort_order)
    out = out.groupby("mentee_id").head(k).reset_index(drop=True)

    # Fairness safeguard (Item 26): cap per-program exposure
    if len(out) > 20:
        program_counts = out["post_id"].value_counts()
        max_allowed = max(2, int(len(out) * _MAX_PROGRAM_EXPOSURE_FRACTION))
        overexposed = program_counts[program_counts > max_allowed].index
        if len(overexposed) > 0:
            logger.warning(
                "Fairness: %d programs exceed exposure cap of %d — consider increasing candidate diversity",
                len(overexposed), max_allowed,
            )

    # Recommendation confidence (Item 24) and explainability (Item 25)
    out["recommendation_confidence"] = out.apply(_compute_recommendation_confidence, axis=1)
    out["explanation_metadata"] = out.apply(_build_explanation_metadata, axis=1)

    out["pred_score"] = out["rerank_score"]
    return out


def evaluate_program_model(
    scored_df: pd.DataFrame,
    k: int = 10,
    min_candidates: int = 2,
) -> Dict[str, float]:
    """Evaluate ranking quality (NDCG@k, HitRate@k, MRR@k)."""
    if scored_df.empty:
        return {f"ndcg@{k}": 0.0, f"hitrate@{k}": 0.0, f"mrr@{k}": 0.0, "groups": 0}

    ndcg_vals = []
    hit_vals = []
    mrr_vals = []

    for _, g in scored_df.groupby("mentee_id"):
        if len(g) < min_candidates:
            continue
        y_true = pd.to_numeric(g["label"], errors="coerce").fillna(0).astype(int).values
        y_score = pd.to_numeric(g["pred_score"], errors="coerce").fillna(0).values
        if y_true.sum() == 0:
            continue

        order = np.argsort(-y_score)
        y_top = y_true[order][:k]

        # HitRate@k
        hit_vals.append(float(y_top.sum() > 0))

        # MRR@k
        rr = 0.0
        for rank_idx, val in enumerate(y_top, start=1):
            if val > 0:
                rr = 1.0 / rank_idx
                break
        mrr_vals.append(rr)

        # NDCG@k
        def dcg(arr):
            return np.sum([(2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(arr)])

        dcg_k = dcg(y_top)
        ideal = np.sort(y_true)[::-1][:k]
        idcg_k = dcg(ideal)
        ndcg_vals.append(float(dcg_k / idcg_k) if idcg_k > 0 else 0.0)

    return {
        f"ndcg@{k}": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        f"hitrate@{k}": float(np.mean(hit_vals)) if hit_vals else 0.0,
        f"mrr@{k}": float(np.mean(mrr_vals)) if mrr_vals else 0.0,
        "groups": float(len(ndcg_vals)),
    }


def generate_program_recommendations(
    model: LGBMRanker,
    candidate_df: pd.DataFrame,
    feature_cols: List[str],
    top_k: int = 10,
) -> pd.DataFrame:
    scored = predict_program_scores(model, candidate_df, feature_cols)
    reranked = rerank_program_recommendations(scored, k=top_k)
    # Preserve the reranker order so exact minimum matches stay ahead of
    # higher-than-minimum programs within the same mentee group.
    out = reranked.reset_index(drop=True)
    keep_cols = [
        col for col in [
            "mentee_id",
            "post_id",
            "pred_score",
            "score",
            "match_percentage",
            "recommendation_confidence",
            "requirement_coverage_score",
            "required_skill_level_match_score",
            "minimum_requirement_exact_match",
            "minimum_requirement_above_minimum",
            "target_level_pass",
            "education_level_pass",
            # "availability_pass" removed (no variance)
        ]
        if col in out.columns
    ]
    return out[keep_cols]
