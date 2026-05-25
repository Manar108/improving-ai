from __future__ import annotations

import logging
import ast
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

from .io import (
    get_project_paths,
    load_features,
    load_json_artifact,
    load_model,
    load_scaler,
    save_feature_artifact,
    save_json_artifact,
    save_model,
    save_scaler,
)
from .features import (
    BINARY_FEATURE_COLS,
    build_cf_embeddings,
    build_community_membership_sets,
    build_interaction_features,
    build_mentee_features,
    build_mentor_features,
    build_pair_features,
    build_recommendation_dataset,
    build_requirement_sets,
    generate_candidate_pool,
)
from .preprocessing import build_time_split_config, load_db_datasets, load_db_datasets_from_db, load_raw_datasets, prepare_processed_tables, save_time_split_config
from .ranking import (
    apply_multi_signal_rerank,
    apply_soft_business_boosts,
    evaluate_model,
    generate_top_k_recommendations,
    prepare_ranking_features,
    scale_features,
    split_by_time,
    train_model,
)


def _get_user_frame_cached(bundle: Dict[str, object], user_id) -> pd.DataFrame:
    """Retrieve pre-computed recommendation features for a user, with caching."""
    recommendation_features = bundle["recommendation_features"]

    try:
        user_id_int = int(user_id)
    except (ValueError, TypeError):
        return recommendation_features.iloc[0:0].copy()

    if "_user_row_index" not in bundle:
        user_row_index = recommendation_features.groupby("mentee_id").indices
        bundle["_user_row_index"] = {int(k): v for k, v in user_row_index.items()}

    if "_user_prefilter_row_index" in bundle:
        user_idx = bundle["_user_prefilter_row_index"].get(user_id_int)
        if user_idx is not None:
            return recommendation_features.loc[user_idx].copy()

    user_idx = bundle["_user_row_index"].get(user_id_int)
    if user_idx is None:
        return recommendation_features.iloc[0:0].copy()
    return recommendation_features.loc[user_idx].copy()


# ---------------------------------------------------------------------------
# Temporal diversity: in-memory cache of recently shown mentors per user.
# Used by predict_for_user() to soft-decay mentors shown in recent sessions,
# improving discovery and UX freshness.  Max 100 users tracked; entries
# auto-evict oldest when limit is reached.  Memory-only — not persisted.
# ---------------------------------------------------------------------------
_recent_recommendations: dict[int, list[set]] = {}
_RECENT_RECS_MAX_USERS = 100


def _track_and_trim_recent_recs(user_id: int, shown_mentors: set) -> None:
    """Track shown mentors for a user, trimming cache to max size."""
    history = _recent_recommendations.get(user_id, [])
    history.append(shown_mentors)
    _recent_recommendations[user_id] = history[-3:]  # keep last 3 sessions
    # Evict oldest users if cache exceeds max
    if len(_recent_recommendations) > _RECENT_RECS_MAX_USERS:
        oldest = next(iter(_recent_recommendations))
        del _recent_recommendations[oldest]


# ---------------------------------------------------------------------------
# Business Rule Post-Filters
# ---------------------------------------------------------------------------

EVAL_K = 10
TRAIN_NEG_PER_POS = 9
EVAL_NEG_PER_POS = 14
MIN_CANDIDATES_PER_GROUP = 5
MIN_EVAL_CANDIDATES = 5
DEFAULT_FOLLOW_RERANK_RATIO = 3  # kept for manifest backward compatibility

# Skill-First Reranking Weights (May 2026 — Production).
# ─────────────────────────────────────────────────────────────────────────────
# CORE COMPATIBILITY SIGNALS ONLY — no serving/business signals in this layer.
# ─────────────────────────────────────────────────────────────────────────────
#
# These weights control the multi-signal reranking layer that refines
# the ML model's predictions.  Only genuine compatibility signals belong here.
# Availability, popularity, and follow are in SOFT_BUSINESS_BOOSTS instead.
#
# SIGNAL SEPARATION PHILOSOPHY:
#   Layer A — Core Compatibility (this dict):
#     pred_score:      ML model's learned compatibility (dominant)
#     skill_coverage:  fraction of mentee's skills covered by mentor
#     subdomain:       specialization alignment (same field/specialty)
#     domain_match:    broader domain alignment (same field)
#
#   Layer B — Quality (inside SOFT_BUSINESS_BOOSTS):
#     high_rating_same_subdomain: quality + compatibility overlap
#
#   Layer C — Serving/Business (inside SOFT_BUSINESS_BOOSTS):
#     open_programs, completed_programs, followed, popularity
#
# NOTE: open_programs was removed from this layer — it was dead code anyway
# (checked for "open_programs" column but actual feature is "mentor_open_post_count_log")
# and it's a serving-time signal, not mentor-mentee compatibility.
SKILL_FIRST_RERANK_WEIGHTS = {
    "pred_score": 0.60,    # ML model — dominant but shared with compatibility signals
    "skill_coverage": 0.20,  # Coverage — PRIMARY compatibility signal
    "subdomain": 0.12,       # Subdomain — specialization alignment
    "domain_match": 0.08,    # Domain — broader field alignment
}

# Soft business-rule boosts (multiplicative, NEVER remove mentors).
# Applied AFTER core compatibility reranking as conditional multipliers.
#
# Three tiers, ordered from strongest to weakest:
#
#   Tier A — Quality (compatibility-adjacent, strongest boost):
#     Rewards mentors who combine quality with compatibility (e.g. high
#     rating in the SAME subdomain).  These are nearest to core compatibility.
#
#   Tier B — Serving/Availability (moderate boost):
#     Rewards mentors who are currently available for mentorship.
#     Availability is a serving-time state, not compatibility — it should
#     NEVER override a strong skill match.
#
#   Tier C — Social/Business (weakest boost):
#     Follow relationship and popularity.  These are trust/familiarity
#     signals, not compatibility.  Should only help when candidates are
#     already close in score.
#
# Max compound boost: ~1.048 (4.8% score increase).
SOFT_BUSINESS_BOOSTS = {
    # Tier A — Quality (compatibility-adjacent)
    "high_rating_same_subdomain": 1.015,   # High rating + same subdomain
    # Tier B — Serving/Availability
    "open_programs": 1.02,                 # Has open programs (reduced from 1.025)
    "completed_programs": 1.008,           # Has completed programs (reduced from 1.01)
    # Tier C — Social/Business (weakest)
    "followed": 1.003,                     # Mentee follows mentor (minimal boost)
    "active_popular_same_subdomain": 1.005, # Active/popular + same subdomain
}





def apply_safe_filters(
    recommendations: pd.DataFrame,
    mentor_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply ONLY safe hard filters (banned, deleted, inactive accounts).

    Aggressive business rules (open posts, capacity, rating, profile picture)
    are REMOVED from hard filtering.  They are now soft reranking boosts only.

    Returns:
        Filtered recommendations DataFrame with only unsafe mentors removed.
    """
    if recommendations.empty:
        return recommendations

    before_count = len(recommendations)
    filtered = recommendations.copy()

    # Safe filter 1: inactive / deleted / banned accounts
    if mentor_profile is not None and "mentor_id" in filtered.columns:
        for status_col in ["is_active", "is_deleted", "is_banned", "account_status"]:
            if status_col in mentor_profile.columns:
                if status_col == "is_active":
                    active_ids = set(
                        mentor_profile[mentor_profile[status_col] == 1]["user_id"]
                        .dropna().astype(int).unique()
                    )
                    mask = filtered["mentor_id"].isin(active_ids)
                elif status_col in ("is_deleted", "is_banned"):
                    bad_ids = set(
                        mentor_profile[mentor_profile[status_col] == 1]["user_id"]
                        .dropna().astype(int).unique()
                    )
                    mask = ~filtered["mentor_id"].isin(bad_ids)
                else:
                    # account_status — keep only "active"
                    active_ids = set(
                        mentor_profile[mentor_profile[status_col].astype(str).str.lower() == "active"]["user_id"]
                        .dropna().astype(int).unique()
                    )
                    mask = filtered["mentor_id"].isin(active_ids)
                removed = (~mask).sum()
                if removed > 0:
                    logger.info("Safe filter [%s]: removed %d recommendations", status_col, removed)
                filtered = filtered[mask]

    after_count = len(filtered)
    total_removed = before_count - after_count
    if total_removed > 0:
        logger.info(
            "Safe filters: %d -> %d recommendations (%d removed, %.1f%% kept)",
            before_count, after_count, total_removed,
            100 * after_count / max(1, before_count),
        )

    return filtered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Centralized Label Filtering (May 2026)
# ---------------------------------------------------------------------------

def _apply_label_filters(
    positive_pairs_by_split: dict[str, set],
    pair_base: pd.DataFrame,
) -> tuple[dict[str, set], dict[str, int]]:
    """Apply all positive-pair label filters in a single centralized pass.

    Filters applied:
      1. Zero skill coverage: pairs where skill_coverage_score == 0 are removed.
         These represent mentors with no listed requirements or users who applied
         due to non-skill reasons (interactions, following). Including them as
         positives teaches the model to recommend unsuitable mentors.

    Returns:
        Tuple of (filtered positive_pairs_by_split, filter_log dict).
    """
    filter_log: dict[str, int] = {}
    total_before = sum(len(v) for v in positive_pairs_by_split.values())

    # Filter 1: Zero skill coverage
    zero_coverage_pairs = set(
        zip(
            pair_base[pair_base["skill_coverage_score"] == 0]["mentee_id"],
            pair_base[pair_base["skill_coverage_score"] == 0]["mentor_id"],
        )
    )
    for split_name in ("train", "valid", "test"):
        positive_pairs_by_split[split_name] -= zero_coverage_pairs

    total_after = sum(len(v) for v in positive_pairs_by_split.values())
    removed = total_before - total_after
    filter_log["zero_coverage_removed"] = removed

    logger.info(
        "Label filters: removed %d positive pairs total (%.2f%% of %d)",
        removed, 100.0 * removed / total_before if total_before > 0 else 0,
        total_before,
    )
    logger.info(
        "  zero_coverage: %d removed", filter_log["zero_coverage_removed"],
    )
    logger.info(
        "  After all filters: train=%d, valid=%d, test=%d",
        len(positive_pairs_by_split.get("train", set())),
        len(positive_pairs_by_split.get("valid", set())),
        len(positive_pairs_by_split.get("test", set())),
    )

    return positive_pairs_by_split, filter_log


# ---------------------------------------------------------------------------
# Diagnostics Helpers
# ---------------------------------------------------------------------------

def _log_rerank_impact(
    stage_name: str,
    raw_scored: pd.DataFrame,
    reranked_scored: pd.DataFrame,
    k: int = 10,
) -> None:
    """Log how reranking changed the top-k ordering vs raw model scores.

    Reports:
      - Fraction of top-k that changed after reranking
      - Score spread (std) before and after
      - Average rank displacement of positive labels
    """
    if raw_scored.empty or reranked_scored.empty:
        return

    reorder_fracs = []
    for mentee_id in raw_scored["mentee_id"].unique():
        raw_group = raw_scored[raw_scored["mentee_id"] == mentee_id].nlargest(k, "pred_score")
        reranked_group = reranked_scored[reranked_scored["mentee_id"] == mentee_id].nlargest(k, "pred_score")
        raw_top = set(raw_group["mentor_id"].values)
        reranked_top = set(reranked_group["mentor_id"].values)
        if raw_top:
            reorder_fracs.append(1.0 - len(raw_top & reranked_top) / len(raw_top))

    if reorder_fracs:
        import numpy as np
        avg_reorder = np.mean(reorder_fracs)
        max_reorder = np.max(reorder_fracs)
        logger.info(
            "Rerank impact [%s]: avg top-%d reorder=%.1f%%, max=%.1f%% "
            "(0%%=no change, 100%%=completely different)",
            stage_name, k, avg_reorder * 100, max_reorder * 100,
        )
        if avg_reorder > 0.5:
            logger.warning(
                "Rerank impact [%s]: >50%% of top-%d changed — reranking may be "
                "overriding model predictions. Consider reducing rerank weights.",
                stage_name, k,
            )


def run_full_pipeline(raw_data=None):
    """Run the full recommendation pipeline from data to trained model.

    Pipeline flow:
      1. Load & preprocess data (from DB by default)
      2. Build features (mentee, mentor, interaction)
      3. Resolve positive pairs from applications (label source)
      4. Generate candidate pool + inject positive pairs (high recall)
      5. Build pair features → dataset → train → evaluate

    Labels: label=1 if user applied to mentor (user interest signal).
    Applications are high-quality — users cannot apply without meeting
    skill/level requirements, so they represent informed interest.

    Args:
        raw_data: One of:
            - None → loads directly from SQL Server database (production)
            - Path → auto-detects DB vs raw format based on file presence
            - dict of DataFrames → uses directly

    Returns:
        Dict containing all pipeline artifacts.
    """
    if raw_data is None:
        # Default: load directly from SQL Server database (production mode)
        logger.info("run_full_pipeline: loading directly from SQL Server database")
        raw_tables = load_db_datasets_from_db()
    elif isinstance(raw_data, (str, Path)):
        data_path = Path(raw_data)
        # Auto-detect: DB format has programs.csv, raw has mentorship_posts.csv
        if (data_path / "programs.csv").exists():
            logger.info("run_full_pipeline: detected DB format in %s", data_path)
            raw_tables = load_db_datasets(data_path)
        else:
            logger.info("run_full_pipeline: detected raw CSV format in %s", data_path)
            raw_tables = load_raw_datasets(data_path)
    elif isinstance(raw_data, dict):
        raw_tables = raw_data
    else:
        raise TypeError("raw_data must be None (default DB), a Path, or a dict of DataFrames")

    config = build_time_split_config(
        raw_tables["mentorships"],
        applications=raw_tables.get("mentorship_applications"),
    )
    processed = prepare_processed_tables(raw_tables, config)
    project_root = get_project_paths()["root"]
    save_time_split_config(config, project_root / "config" / "time_split_config.csv")

    # ── Mentorships: used for time boundaries and enrollment features ──
    mentorships_split_col = processed["mentorships"].get(
        "time_split", pd.Series("train", index=processed["mentorships"].index)
    )
    mentorships_train = processed["mentorships"][mentorships_split_col == "train"].copy()

    # ── Applications: label source (label=1 if user applied) ──
    # For FEATURES (popularity, reliability): use train-only apps.
    # For LABELS: use all apps with their time_split.
    # IMPORTANT (May 2026): Filter out "alerted" status applications.
    # Reason: alerted = requirements mismatch, insufficient skills/level, NOT reliable mentor-suitability signals.
    # These create noisy positive labels that force the model to learn coverage tricks.
    apps_all = processed["mentorship_applications"].copy()
    apps_before_filter = len(apps_all)
    apps_all = apps_all[apps_all.get("status", "").str.lower() != "alerted"].copy()
    apps_filtered = apps_before_filter - len(apps_all)
    logger.info(
        "Label filtering: removed %d 'alerted' status applications (%.1f%% of %d total)",
        apps_filtered, 100.0 * apps_filtered / apps_before_filter if apps_before_filter > 0 else 0, apps_before_filter,
    )
    
    apps_split_col = apps_all.get(
        "time_split", pd.Series("train", index=apps_all.index)
    )
    apps_train = apps_all[apps_split_col == "train"].copy()

    # Post→mentor mapping (referential lookup — includes all posts so valid/test
    # applications can be resolved).
    posts_mapping = raw_tables["mentorship_posts"][["post_id", "mentor_id"]].copy()
    posts_mapping["post_id"] = pd.to_numeric(posts_mapping["post_id"], errors="coerce")
    posts_mapping["mentor_id"] = pd.to_numeric(posts_mapping["mentor_id"], errors="coerce")
    posts_mapping = posts_mapping.dropna().drop_duplicates().astype({"post_id": int, "mentor_id": int})

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Build features
    # ══════════════════════════════════════════════════════════════════════
    mentee_features = build_mentee_features(
        processed["mentee_profile"],
        processed["mentee_subdomains"],
        processed["mentee_interests"],
    )
    mentor_features = build_mentor_features(
        processed["mentor_profile"],
        processed["mentor_subdomains"],
        processed["mentor_expertise"],
        processed["mentors_feedback"],
        apps_train,
        processed["mentorship_posts"],
        processed["mentorship_cancellation"],
        mentorships_train,
        processed["follows"],
        train_end=config.train_end,
        likes_hist=processed.get("posts_likes_dataset"),
        comments_hist=processed.get("posts_comments"),
        saves_hist=processed.get("saved_posts_dataset"),
        shares_hist=processed.get("shared_posts_dataset"),
    )
    interaction_features = build_interaction_features(
        processed.get("posts_likes_dataset", pd.DataFrame()),
        processed.get("posts_comments", pd.DataFrame()),
        processed.get("saved_posts_dataset", pd.DataFrame()),
        processed.get("shared_posts_dataset", pd.DataFrame()),
        processed["follows"],
        processed["mentorship_posts"],
    )

    # ── Collaborative filtering embeddings ──
    mentor_ids_for_cf = set(mentor_features["mentor_id"].dropna().astype(int).unique())
    cf_embeddings = build_cf_embeddings(
        interaction_features,
        processed["follows"],
        mentorships_train,
        processed["mentorship_posts"],
        mentor_ids_set=mentor_ids_for_cf,
        n_factors=16,
    )

    # ── Community co-membership ──
    community_sets = build_community_membership_sets(
        processed.get("community_members", pd.DataFrame()),
    )

    # ── Post requirement sets ──
    mentor_requirement_sets = build_requirement_sets(
        processed.get("mentorship_requirements", pd.DataFrame()),
        processed["mentorship_posts"],
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Resolve positive pairs BEFORE candidate pool generation
    # ══════════════════════════════════════════════════════════════════════
    apps_with_mentor = apps_all.merge(posts_mapping, on="post_id", how="inner")
    if "mentee_id" not in apps_with_mentor.columns and "user_id" in apps_with_mentor.columns:
        apps_with_mentor = apps_with_mentor.rename(columns={"user_id": "mentee_id"})
    apps_with_mentor["mentee_id"] = pd.to_numeric(apps_with_mentor["mentee_id"], errors="coerce")
    apps_with_mentor["mentor_id"] = pd.to_numeric(apps_with_mentor["mentor_id"], errors="coerce")
    apps_with_mentor = apps_with_mentor.dropna(subset=["mentee_id", "mentor_id"]).copy()
    apps_with_mentor[["mentee_id", "mentor_id"]] = (
        apps_with_mentor[["mentee_id", "mentor_id"]].astype(int)
    )

    positive_pairs_by_split = {}
    for split_name in ("train", "valid", "test"):
        split_df = apps_with_mentor[apps_with_mentor["time_split"] == split_name]
        positive_pairs_by_split[split_name] = set(
            zip(split_df["mentee_id"], split_df["mentor_id"])
        )
    total_positives = sum(len(v) for v in positive_pairs_by_split.values())
    logger.info(
        "Label source: mentorship_applications (filtered: no 'alerted' status) — %d total positive (mentee, mentor) pairs "
        "(train=%d, valid=%d, test=%d)",
        total_positives,
        len(positive_pairs_by_split.get("train", set())),
        len(positive_pairs_by_split.get("valid", set())),
        len(positive_pairs_by_split.get("test", set())),
    )

    # Event time = first application date (user interest time, not mentorship start)
    event_time_by_mentee = (
        apps_with_mentor
        .assign(applied_at=pd.to_datetime(apps_with_mentor["applied_at"], errors="coerce"))
        .dropna(subset=["applied_at"])
        .groupby("mentee_id")["applied_at"]
        .min()
    )

    # All positive pairs as a DataFrame for injection
    all_positive_pairs_df = pd.DataFrame(
        list(set().union(*positive_pairs_by_split.values())),
        columns=["mentee_id", "mentor_id"],
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Generate candidate pool + inject positive pairs
    # ══════════════════════════════════════════════════════════════════════
    subdomains_map = processed.get("subdomains")
    candidate_pool = generate_candidate_pool(
        mentee_features, mentor_features,
        subdomains_map=subdomains_map,
        top_k=50, min_candidates_per_mentee=15,
        high_priority_cap=80, low_priority_cap=40,
        exploration_pct=0.15,
    )

    # CRITICAL: Inject ALL positive pairs into candidate pool.
    # Candidate stage = HIGH RECALL (every applied pair present).
    # Ranking stage = PRECISION (ordering within pool).
    pool_before = len(candidate_pool)
    candidate_pool = pd.concat(
        [candidate_pool, all_positive_pairs_df[["mentee_id", "mentor_id"]]],
        ignore_index=True,
    ).drop_duplicates(["mentee_id", "mentor_id"]).reset_index(drop=True)
    injected = len(candidate_pool) - pool_before
    logger.info(
        "Positive pair injection: %d new pairs added to candidate pool "
        "(pool: %d -> %d)",
        injected, pool_before, len(candidate_pool),
    )

    # ── Coverage validation ──
    pool_set = set(zip(candidate_pool["mentee_id"], candidate_pool["mentor_id"]))
    coverage_log = {}
    for split_name in ("train", "valid", "test"):
        split_positives = positive_pairs_by_split.get(split_name, set())
        if not split_positives:
            coverage_log[split_name] = {"total": 0, "in_pool": 0, "coverage": 0.0}
            continue
        in_pool = len(split_positives & pool_set)
        coverage_pct = 100 * in_pool / len(split_positives)
        missing = len(split_positives) - in_pool
        coverage_log[split_name] = {
            "total": len(split_positives),
            "in_pool": in_pool,
            "coverage": round(coverage_pct, 1),
        }
        logger.info(
            "Coverage %s: %d/%d positives in pool (%.1f%%) — %d missing",
            split_name, in_pool, len(split_positives), coverage_pct, missing,
        )

    # Sanity check: missing should be ~0 after injection
    total_missing = sum(c["total"] - c["in_pool"] for c in coverage_log.values())
    if total_missing > 0:
        logger.error(
            "SANITY CHECK FAILED: %d positive pairs still missing after injection! "
            "These pairs have mentee/mentor IDs not in feature tables.",
            total_missing,
        )

    # Hard failure warning if valid/test positives are too low
    for _sname in ("valid", "test"):
        _in_pool = coverage_log.get(_sname, {}).get("in_pool", 0)
        if _in_pool < 30:
            logger.warning(
                "LOW COVERAGE: %s has only %d positives in pool (< 30 threshold). "
                "Model may not learn meaningful ranking for this split.",
                _sname, _in_pool,
            )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Build pair features
    # ══════════════════════════════════════════════════════════════════════
    pair_base = build_pair_features(
        candidate_pool, mentee_features, mentor_features,
        interaction_features, processed["follows"],
        cf_embeddings=cf_embeddings,
        community_sets=community_sets,
        mentor_requirement_sets=mentor_requirement_sets,
        event_time_by_mentee=event_time_by_mentee,
        likes_hist=raw_tables.get("posts_likes_dataset"),
        comments_hist=raw_tables.get("posts_comments"),
        saves_hist=raw_tables.get("saved_posts_dataset"),
        shares_hist=raw_tables.get("shared_posts_dataset"),
        posts_hist=raw_tables.get("mentorship_posts"),
        mentorships_hist_raw=raw_tables.get("mentorships"),
        follows_hist_raw=raw_tables.get("follows"),
    )

    # ══════════════════════════════════════════════════════════════════════
    # CRITICAL: LEAKAGE DETECTION (May 2026)
    # ══════════════════════════════════════════════════════════════════════
    # CRITICAL FIX: Detect and block potential leakage columns before training.
    # Same regex + blacklist patterns used in program recommender.
    # Leakage columns invalidate model training and test metrics.
    from src.hybrid_recommender.program_recommender.preprocessing import detect_leakage_columns
    
    leakage_cols = detect_leakage_columns(pair_base.columns)
    if leakage_cols:
        logger.error("LEAKAGE DETECTION FAILED: Found suspicious columns that may leak future data: %s", leakage_cols)
        logger.error("  Examples: future_*, post_split_*, eval_*, leakage")
        logger.error("  Failing training to prevent model contamination.")
        raise ValueError(f"CRITICAL: Leakage columns detected in pair_base: {leakage_cols}. Training aborted.")
    
    logger.info("Leakage detection PASSED: no suspicious columns detected in pair_base")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: Apply ALL label filters BEFORE dataset generation
    # ══════════════════════════════════════════════════════════════════════
    # Centralized here to avoid the previous double-build pattern where
    # build_recommendation_dataset was called twice (wasting compute and
    # causing inconsistent negative sampling due to same rng_seed on
    # different positive sets).
    positive_pairs_by_split, filter_log = _apply_label_filters(
        positive_pairs_by_split, pair_base,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: Build dataset (SINGLE call — after all filters)
    # ══════════════════════════════════════════════════════════════════════
    recommendation_features = build_recommendation_dataset(
        pair_base,
        positive_pairs_by_split,
        event_time_by_mentee,
        config.train_end,
        config.valid_end,
        neg_per_pos=TRAIN_NEG_PER_POS,
        eval_neg_per_pos=EVAL_NEG_PER_POS,
        min_candidates_per_group=MIN_CANDIDATES_PER_GROUP,
        rng_seed=42,
    )

    # ══════════════════════════════════════════════════════════════════════
    # TIME-SPLIT DIAGNOSTICS (May 2026): Verify after label filtering
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n=== TIME-SPLIT DIAGNOSTICS (after 'alerted' filtering) ===")
    for split_name in ("train", "valid", "test"):
        split_data = recommendation_features[recommendation_features["time_split"] == split_name]
        n_rows = len(split_data)
        n_positives = int(split_data["label"].sum())
        n_users = split_data["mentee_id"].nunique()
        positive_ratio = 100.0 * n_positives / n_rows if n_rows > 0 else 0
        logger.info(
            "  %s: %d rows, %d positives (%.2f%%), %d unique mentees",
            split_name, n_rows, n_positives, positive_ratio, n_users,
        )
    
    # Check for time leakage
    test_data = recommendation_features[recommendation_features["time_split"] == "test"]
    valid_data = recommendation_features[recommendation_features["time_split"] == "valid"]
    if len(test_data) > 0 and len(valid_data) > 0:
        min_test_time = test_data.get("event_time", pd.Series()).min() if "event_time" in test_data.columns else None
        max_valid_time = valid_data.get("event_time", pd.Series()).max() if "event_time" in valid_data.columns else None
        if min_test_time is not None and max_valid_time is not None and pd.notna(min_test_time) and pd.notna(max_valid_time):
            if min_test_time <= max_valid_time:
                logger.warning("  WARNING: Time leakage detected! Test min_time=%s <= valid max_time=%s", min_test_time, max_valid_time)
            else:
                logger.info("  Time boundaries OK: valid_max_time=%s < test_min_time=%s", max_valid_time, min_test_time)

    recommendation_features, feature_cols = prepare_ranking_features(recommendation_features)

    # Log label distribution before splitting
    if "time_split" in recommendation_features.columns:
        _pre_split = recommendation_features.groupby("time_split")["label"].agg(["sum", "count"])
        _pre_split["pos_ratio"] = (_pre_split["sum"] / _pre_split["count"]).round(4)
        logger.info("\n=== DISTRIBUTION BEFORE TRAIN SPLIT ===")
        logger.info("\n%s", _pre_split.to_string())

        for _sname in ("valid", "test"):
            if _sname in _pre_split.index and _pre_split.loc[_sname, "sum"] == 0:
                raise ValueError(
                    f"CRITICAL: {_sname} split has ZERO positive labels — "
                    f"evaluation cannot proceed."
                )

    train_df, valid_df, test_df = split_by_time(
        recommendation_features,
        {"train_end": config.train_end, "valid_end": config.valid_end},
    )

    logger.info("\n=== DISTRIBUTION AFTER TRAIN SPLIT ===")
    for _sname, _sdf in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        _n = len(_sdf)
        _pos = int(_sdf["label"].sum()) if "label" in _sdf.columns else 0
        _ratio = round(_pos / _n, 4) if _n > 0 else 0
        logger.info("  %s: %d rows, %d positives, ratio=%.4f", _sname, _n, _pos, _ratio)

    scale_cols = [
        col for col in feature_cols
        if col in recommendation_features.columns and col not in BINARY_FEATURE_COLS
    ]
    train_df, valid_df, test_df, scaler = scale_features(train_df, valid_df, test_df, scale_cols)
    model = train_model(train_df, valid_df, feature_cols)

    train_scored = train_df.copy()
    train_scored["pred_score"] = model.predict(train_scored[feature_cols])
    valid_scored = valid_df.copy()
    valid_scored["pred_score"] = model.predict(valid_scored[feature_cols])
    test_scored = test_df.copy()
    test_scored["pred_score"] = model.predict(test_scored[feature_cols])

    metrics_train = evaluate_model(train_scored, k=EVAL_K, min_candidates=MIN_EVAL_CANDIDATES)
    metrics_valid = evaluate_model(valid_scored, k=EVAL_K, min_candidates=MIN_EVAL_CANDIDATES)
    metrics_raw = evaluate_model(test_scored, k=EVAL_K, min_candidates=MIN_EVAL_CANDIDATES)

    # DEBUG: Verify skill coverage is actually driving rankings (May 2026)
    from .ranking import debug_skill_coverage_verification
    debug_skill_coverage_verification(test_scored, k=5, sample_size=5)

    # ── Score spread analysis (detect tight clustering) ──
    import numpy as np
    _test_pred_std = test_scored.groupby("mentee_id")["pred_score"].std().median()
    _test_pred_range = (
        test_scored.groupby("mentee_id")["pred_score"].apply(lambda x: x.max() - x.min()).median()
    )
    logger.info(
        "Score spread (test): median_std=%.4f, median_range=%.4f",
        _test_pred_std, _test_pred_range,
    )
    if _test_pred_range < 0.5:
        logger.warning(
            "Score spread WARNING: median intra-group range=%.4f is very tight. "
            "Reranking additive signals may disproportionately reorder rankings. "
            "Consider increasing ML weight or reducing rerank weights.",
            _test_pred_range,
        )

    # Apply skill-first multi-signal reranking (May 2026).
    # Priority: Skills > Subdomain > Follow > Domain > Quality.
    test_reranked = apply_multi_signal_rerank(test_scored, weights=SKILL_FIRST_RERANK_WEIGHTS)
    test_reranked["pred_score"] = test_reranked["rerank_score"]
    metrics_reranked = evaluate_model(test_reranked, k=EVAL_K, min_candidates=MIN_EVAL_CANDIDATES)

    # Rerank impact: how much did multi-signal reranking change top-k?
    _log_rerank_impact("multi_signal_rerank", test_scored, test_reranked, k=EVAL_K)

    valid_reranked = apply_multi_signal_rerank(valid_scored, weights=SKILL_FIRST_RERANK_WEIGHTS)
    valid_reranked["pred_score"] = valid_reranked["rerank_score"]
    metrics_valid_reranked = evaluate_model(valid_reranked, k=EVAL_K, min_candidates=MIN_EVAL_CANDIDATES)



    recommendations = generate_top_k_recommendations(
        model,
        test_df,
        feature_cols,
        k=EVAL_K,
        rerank=True,
    )

    # ── Apply soft business boosts to served recommendations ──
    from .ranking import apply_soft_business_boosts
    test_reranked_boosted = apply_soft_business_boosts(test_reranked)
    test_reranked_boosted["pred_score"] = test_reranked_boosted["rerank_score"]
    metrics_reranked_boosted = evaluate_model(test_reranked_boosted, k=EVAL_K, min_candidates=MIN_EVAL_CANDIDATES)

    # Rerank impact: how much did soft boosts change top-k beyond multi-signal?
    _log_rerank_impact("soft_business_boosts", test_reranked, test_reranked_boosted, k=EVAL_K)

    # ── Safe production filters ONLY (no aggressive hard filters) ──
    # Only remove banned / deleted / inactive accounts.  All other business
    # rules (open posts, capacity, rating, profile picture) are SOFT BOOSTS.
    recommendations = apply_safe_filters(
        recommendations,
        mentor_profile=processed.get("mentor_profile"),
    )
    test_served = apply_safe_filters(
        test_reranked_boosted,
        mentor_profile=processed.get("mentor_profile"),
    )
    metrics_served = evaluate_model(test_served, k=EVAL_K, min_candidates=MIN_EVAL_CANDIDATES)
    metrics_served_global = evaluate_model(test_served, k=EVAL_K, min_candidates=MIN_EVAL_CANDIDATES, include_skipped=True)
    metrics = metrics_served

    # ── 3-STAGE EVALUATION LOGGING ──
    logger.info("\n=== MODEL EVALUATION METRICS (train raw) ===")
    for mk, mv in metrics_train.items():
        logger.info("  %s: %.4f", mk, mv) if isinstance(mv, float) else logger.info("  %s: %s", mk, mv)
    logger.info("\n=== MODEL EVALUATION METRICS (valid raw) ===")
    for mk, mv in metrics_valid.items():
        logger.info("  %s: %.4f", mk, mv) if isinstance(mv, float) else logger.info("  %s: %s", mk, mv)
    logger.info("\n=== MODEL EVALUATION METRICS (valid reranked) ===")
    for mk, mv in metrics_valid_reranked.items():
        logger.info("  %s: %.4f", mk, mv) if isinstance(mv, float) else logger.info("  %s: %s", mk, mv)
    logger.info("\n=== MODEL EVALUATION METRICS (test raw — pure model) ===")
    for mk, mv in metrics_raw.items():
        logger.info("  %s: %.4f", mk, mv) if isinstance(mv, float) else logger.info("  %s: %s", mk, mv)
    logger.info("\n=== MODEL EVALUATION METRICS (test reranked — skill-first soft boosts) ===")
    for mk, mv in metrics_reranked.items():
        logger.info("  %s: %.4f", mk, mv) if isinstance(mv, float) else logger.info("  %s: %s", mk, mv)
    logger.info("\n=== MODEL EVALUATION METRICS (test reranked + business soft boosts) ===")
    for mk, mv in metrics_reranked_boosted.items():
        logger.info("  %s: %.4f", mk, mv) if isinstance(mv, float) else logger.info("  %s: %s", mk, mv)
    logger.info("\n=== MODEL EVALUATION METRICS (test served — safe filters only) ===")
    for mk, mv in metrics_served.items():
        logger.info("  %s: %.4f", mk, mv) if isinstance(mv, float) else logger.info("  %s: %s", mk, mv)
    logger.info("\n=== MODEL EVALUATION METRICS (test served GLOBAL — all users) ===")
    for mk, mv in metrics_served_global.items():
        logger.info("  %s: %.4f", mk, mv) if isinstance(mv, float) else logger.info("  %s: %s", mk, mv)

    # ── POSITIVE COVERAGE DIAGNOSTICS ──
    test_positives = positive_pairs_by_split.get("test", set())
    for stage_name, stage_df in [
        ("raw model", test_scored),
        ("reranked", test_reranked),
        ("reranked + soft boosts", test_reranked_boosted),
        ("served (safe filters)", test_served),
    ]:
        pool_pairs = set(zip(stage_df["mentee_id"], stage_df["mentor_id"]))
        in_pool = len(test_positives & pool_pairs)
        mentees_with_pos = stage_df.groupby("mentee_id")["label"].sum()
        mentees_with_pos_count = (mentees_with_pos > 0).sum()
        logger.info(
            "  Positives %s: %d/%d in pool, %d mentees have positives, %d candidates",
            stage_name, in_pool, len(test_positives), mentees_with_pos_count, len(stage_df),
        )

    # Candidate pool statistics
    test_mentees_total = test_df["mentee_id"].nunique()
    test_mentees_served = test_served["mentee_id"].nunique()
    candidates_per_mentee_served = test_served.groupby("mentee_id").size().mean() if not test_served.empty else 0
    impossible_users = metrics_served.get("skipped_no_positive", 0)
    small_group_users = metrics_served.get("skipped_small_group", 0)
    test_positives_in_served = len(test_positives & set(zip(test_served["mentee_id"], test_served["mentor_id"])))
    logger.info("\n=== CANDIDATE POOL STATISTICS ===")
    logger.info("  Test mentees total: %d", test_mentees_total)
    logger.info("  Test mentees after safe filters: %d", test_mentees_served)
    logger.info("  Impossible users (no positive in pool): %d", impossible_users)
    logger.info("  Small group users (< %d candidates): %d", MIN_EVAL_CANDIDATES, small_group_users)
    logger.info("  Avg candidates per mentee (served): %.1f", candidates_per_mentee_served)
    logger.info("  Test positives total: %d", len(test_positives))
    logger.info("  Test positives in served pool: %d", test_positives_in_served)
    logger.info("  Mentor coverage (served): %d / %d mentors",
                test_served["mentor_id"].nunique(),
                processed["mentor_profile"]["user_id"].nunique())

    ndcg_key = f"ndcg@{EVAL_K}"
    hitrate_key = f"hitrate@{EVAL_K}"
    if ndcg_key in metrics_train and ndcg_key in metrics_valid:
        logger.info(
            "Overfit gap (train-valid) %s: %.4f",
            ndcg_key,
            metrics_train.get(ndcg_key, 0.0) - metrics_valid.get(ndcg_key, 0.0),
        )
    if hitrate_key in metrics_train and hitrate_key in metrics_valid:
        logger.info(
            "Overfit gap (train-valid) %s: %.4f",
            hitrate_key,
            metrics_train.get(hitrate_key, 0.0) - metrics_valid.get(hitrate_key, 0.0),
        )
    if hitrate_key in metrics_valid_reranked and hitrate_key in metrics_served:
        logger.info(
            "Generalization gap (valid reranked-test served) %s: %.4f",
            hitrate_key,
            metrics_valid_reranked.get(hitrate_key, 0.0) - metrics_served.get(hitrate_key, 0.0),
        )


    artifacts_dir = project_root / "data" / "artifacts"
    save_model(model, artifacts_dir / "model.joblib")
    save_scaler(scaler, artifacts_dir / "scaler.joblib")
    save_feature_artifact(recommendation_features, "recommendation_features.csv")
    save_feature_artifact(mentee_features, "mentee_features.csv")
    save_feature_artifact(mentor_features, "mentor_features.csv")
    save_json_artifact(
        {
            "feature_cols": feature_cols,
            "follow_rerank_ratio": DEFAULT_FOLLOW_RERANK_RATIO,
            "eval_k": EVAL_K,
            "min_eval_candidates": MIN_EVAL_CANDIDATES,
        },
        artifacts_dir / "feature_manifest.json",
    )

    return {
        "config": config,
        "processed": processed,
        "mentee_features": mentee_features,
        "mentor_features": mentor_features,
        "interaction_features": interaction_features,
        "recommendation_features": recommendation_features,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "model": model,
        "metrics": metrics,
        "metrics_global": metrics_served_global,
        "metrics_reranked_boosted": metrics_reranked_boosted,
        "metrics_raw": metrics_raw,
        "metrics_reranked": metrics_reranked,
        "recommendations": recommendations,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "artifacts_dir": artifacts_dir,
        "coverage": coverage_log,
    }



def load_inference_artifacts(artifacts_dir: Path) -> Dict[str, object]:
    """Load pre-trained artifacts from disk for inference.

    Returns a bundle dict containing model, scaler, features, and pre-built
    indexes for fast user lookup.
    """
    model = load_model(artifacts_dir / "model.joblib")
    scaler = load_scaler(artifacts_dir / "scaler.joblib")
    recommendation_features = load_features("recommendation_features.csv")
    mentee_features = load_features("mentee_features.csv")
    mentor_features = load_features("mentor_features.csv")
    manifest = load_json_artifact(artifacts_dir / "feature_manifest.json")
    feature_cols = manifest["feature_cols"]
    follow_rerank_ratio = int(manifest.get("follow_rerank_ratio", DEFAULT_FOLLOW_RERANK_RATIO))

    def parse_set_value(value):
        if isinstance(value, set):
            return value
        if pd.isna(value) or value == "":
            return set()
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return set()
            if isinstance(parsed, set):
                return parsed
            if isinstance(parsed, (list, tuple)):
                return set(parsed)
        return set()

    for frame, id_column in ((mentee_features, "mentee_id"), (mentor_features, "mentor_id")):
        if id_column in frame.columns:
            frame[id_column] = pd.to_numeric(frame[id_column], errors="coerce")
        for column in ["subdomains_set", "interests_set", "expertise_set"]:
            if column in frame.columns:
                frame[column] = frame[column].apply(parse_set_value)

    recommendation_features["mentee_id"] = pd.to_numeric(recommendation_features["mentee_id"], errors="coerce")
    recommendation_features = recommendation_features.dropna(subset=["mentee_id"]).copy()
    recommendation_features["mentee_id"] = recommendation_features["mentee_id"].astype(int)

    user_row_index = recommendation_features.groupby("mentee_id").groups
    user_row_index = {int(k): list(v) for k, v in user_row_index.items()}

    prefilter_col = None
    if "interaction_score_log" in recommendation_features.columns:
        prefilter_col = "interaction_score_log"
    elif "popularity_log" in recommendation_features.columns:
        prefilter_col = "popularity_log"

    user_prefilter_row_index = {}
    if prefilter_col is not None:
        prefiltered = (
            recommendation_features
            .sort_values(["mentee_id", prefilter_col], ascending=[True, False])
            .groupby("mentee_id", sort=False)
            .head(50)
        )
        user_prefilter_row_index = {
            int(k): list(v)
            for k, v in prefiltered.groupby("mentee_id").groups.items()
        }

    bundle = {
        "model": model,
        "scaler": scaler,
        "recommendation_features": recommendation_features,
        "mentee_features": mentee_features,
        "mentor_features": mentor_features,
        "feature_cols": feature_cols,
        "follow_rerank_ratio": follow_rerank_ratio,
        "artifacts_dir": artifacts_dir,
        "_user_row_index": user_row_index,
        "_user_prefilter_row_index": user_prefilter_row_index,
    }

    # Warm up model prediction to avoid cold-start latency
    if feature_cols and not recommendation_features.empty:
        warmup = recommendation_features[feature_cols].head(1)
        try:
            model.predict(warmup)
        except Exception:
            pass

    return bundle


def _content_based_scores(user_row: pd.Series, mentor_features: pd.DataFrame) -> pd.DataFrame:
    """Compute content-based similarity scores for cold-start users.

    Used when a user has a profile but no pre-computed recommendation features.
    Returns signal columns so downstream formatting and reason generation
    work consistently with the warm-start model path.
    """
    def normalize_set(value):
        if isinstance(value, set):
            return value
        if pd.isna(value) or value == "":
            return set()
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return set()
            if isinstance(parsed, set):
                return parsed
            if isinstance(parsed, (list, tuple)):
                return set(parsed)
        if isinstance(value, (list, tuple)):
            return set(value)
        return set()

    mentor_base = mentor_features[["mentor_id", "expertise_set", "subdomains_set", "experience_level_num", "domain_id", "country_code"]].copy()
    mentor_base["expertise_set"] = mentor_base["expertise_set"].apply(normalize_set)
    mentor_base["subdomains_set"] = mentor_base["subdomains_set"].apply(normalize_set)

    user_interests = normalize_set(user_row.get("interests_set", set()))
    user_subdomains = normalize_set(user_row.get("subdomains_set", set()))
    user_level = float(user_row.get("experience_level_num", 1))
    user_domain = pd.to_numeric(user_row.get("domain_id"), errors="coerce")
    user_country = str(user_row.get("country_code", "")).strip().lower()

    mentor_base["skill_overlap_score"] = mentor_base["expertise_set"].apply(lambda values: len(user_interests & values) / len(user_interests | values) if user_interests and values else 0.0)
    mentor_base["skill_coverage_score"] = mentor_base["expertise_set"].apply(lambda values: len(user_interests & values) / len(user_interests) if user_interests else 0.0)
    mentor_base["subdomain_similarity"] = mentor_base["subdomains_set"].apply(lambda values: len(user_subdomains & values) / len(user_subdomains | values) if user_subdomains and values else 0.0)
    mentor_base["experience_gap"] = mentor_base["experience_level_num"].fillna(1).astype(float) - user_level

    # Domain match (cold-start approximation)
    mentor_base["mentor_domain_match"] = 0
    if pd.notna(user_domain):
        mentor_base["mentor_domain_match"] = (mentor_base["domain_id"].fillna(-1) == float(user_domain)).astype(int)

    # Country match
    mentor_base["same_country"] = 0
    if user_country:
        mentor_base["same_country"] = (mentor_base["country_code"].fillna("").str.strip().str.lower() == user_country).astype(int)

    # Mentor more experienced
    mentor_base["mentor_more_experienced"] = (mentor_base["experience_level_num"].fillna(1).astype(float) > user_level).astype(int)

    # ── Experience level alignment ──
    # Mentees benefit most from mentors 1-2 levels above them (not too far):
    # beginner → intermediate, intermediate → advanced.
    # Perfect gap (1 level) gets full bonus, 2 levels gets partial, 0 or >2 gets none.
    exp_gap_raw = mentor_base["experience_level_num"].fillna(1).astype(float) - user_level
    level_alignment = pd.Series(0.0, index=mentor_base.index)
    level_alignment[exp_gap_raw == 1] = 1.0   # ideal: 1 level above
    level_alignment[exp_gap_raw == 2] = 0.6   # good: 2 levels above
    level_alignment[exp_gap_raw == 0] = 0.3   # peer mentoring (still useful)
    mentor_base["level_alignment"] = level_alignment

    # ── Mentorship availability (cold-start) ──
    # Mentors with open programs should rank slightly higher even in cold-start.
    has_open = pd.Series(0.0, index=mentor_base.index)
    if "mentor_open_post_count_log" in mentor_features.columns:
        open_vals = pd.to_numeric(mentor_features["mentor_open_post_count_log"], errors="coerce").fillna(0)
        has_open = (open_vals > 0).astype(float).values[:len(mentor_base)]
        if len(has_open) == len(mentor_base):
            mentor_base["has_open_programs"] = has_open
        else:
            mentor_base["has_open_programs"] = 0.0
    else:
        mentor_base["has_open_programs"] = 0.0

    # Cold-start scoring weights aligned with main reranker priority:
    # skill_overlap > skill_coverage > subdomain > domain > level_alignment > experience > availability
    # More signals than before to reduce popularity-only fallback behavior.
    experience_gap_score = mentor_base["experience_gap"].clip(-3, 3) / 3.0
    mentor_base["pred_score"] = (
        0.30 * mentor_base["skill_overlap_score"]
        + 0.22 * mentor_base["skill_coverage_score"]
        + 0.18 * mentor_base["subdomain_similarity"]
        + 0.10 * mentor_base["mentor_domain_match"].astype(float)
        + 0.08 * mentor_base["level_alignment"]
        + 0.07 * experience_gap_score
        + 0.05 * mentor_base["has_open_programs"]
    )

    # Keep signal columns for downstream formatting
    out_cols = ["mentor_id", "pred_score", "skill_overlap_score", "skill_coverage_score",
                "subdomain_similarity", "mentor_domain_match", "same_country", "mentor_more_experienced"]
    return mentor_base[out_cols].sort_values("pred_score", ascending=False).reset_index(drop=True)


def _global_mentor_fallback(user_id, mentor_features: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Return top-k globally popular mentors as fallback recommendations."""
    score_column = None
    for candidate in ("mentor_popularity_score", "mentor_program_popularity", "mentor_weighted_rating"):
        if candidate in mentor_features.columns:
            score_column = candidate
            break

    if score_column is None:
        return pd.DataFrame(columns=["mentee_id", "mentor_id", "pred_score"])

    fallback = mentor_features[["mentor_id", score_column]].copy()
    fallback["pred_score"] = pd.to_numeric(fallback[score_column], errors="coerce").fillna(0)
    fallback = fallback.sort_values(["pred_score", "mentor_id"], ascending=[False, True]).head(top_k)
    fallback.insert(0, "mentee_id", user_id)

    # Add empty signal columns so downstream formatting works consistently
    for sig_col in ["skill_overlap_score", "skill_coverage_score", "subdomain_similarity",
                    "mentor_domain_match", "mentor_quality_score", "mentor_weighted_rating",
                    "mentor_more_experienced", "same_country"]:
        if sig_col not in fallback.columns:
            fallback[sig_col] = 0.0

    return fallback.reset_index(drop=True)


def predict_for_user(user_id, data, top_k: int = 10):
    """Generate top-k recommendations for a single user.

    Handles three scenarios:
      1. Existing user with pre-computed features → model prediction
      2. Cold-start user with profile → content-based scoring
      3. Unknown user → global popularity fallback

    Features (May 2026):
      - Confidence-aware reranking (via apply_multi_signal_rerank)
      - Weak match penalty (via apply_multi_signal_rerank)
      - Temporal diversity: soft decay for recently shown mentors
      - Score calibration: sigmoid compression for confidence realism

    Args:
        user_id: The user ID to generate recommendations for (int or str).
        data: Either a loaded artifacts bundle (dict) or path to artifacts dir.
        top_k: Number of recommendations to return (default: 10).
    """
    if isinstance(data, dict) and {"model", "recommendation_features", "feature_cols"}.issubset(data.keys()):
        bundle = data
    elif isinstance(data, (str, Path)):
        artifacts_dir = Path(data)
        bundle = load_inference_artifacts(artifacts_dir)
    else:
        raise ValueError("predict_for_user expects loaded artifacts dict or an artifacts directory path")

    # Normalize user_id to integer (artifacts use integer IDs)
    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        logger.info("User ID '%s' is not an integer — skipping artifact lookup", user_id)
        user_id = None

    model = bundle["model"]

    user_frame = pd.DataFrame()
    if user_id is not None:
        user_frame = _get_user_frame_cached(bundle, user_id)

    if not user_frame.empty:
        feature_cols = bundle["feature_cols"]

        # CRITICAL: defensive copy before scaling to prevent mutating cached
        # recommendation_features. Without this, repeated predict_for_user()
        # calls would double-scale the same rows.
        user_frame = user_frame.copy()
        
        # MEDIUM FIX: Apply NaN imputation defaults before inference
        # Missing optional features get sensible defaults (not NaN or 0)
        from src.hybrid_recommender.ranking import OPTIONAL_FEATURE_DEFAULTS
        for feature, default_value in OPTIONAL_FEATURE_DEFAULTS.items():
            if feature in user_frame.columns:
                nan_count = user_frame[feature].isna().sum()
                if nan_count > 0:
                    user_frame[feature] = user_frame[feature].fillna(default_value)
                    logger.debug("NaN imputation: filled %d missing %s values with default=%.2f", 
                               nan_count, feature, default_value)
            else:
                # If column doesn't exist, add it with default value
                user_frame[feature] = default_value

        # Apply scaler to match training preprocessing
        scaler = bundle.get("scaler")
        if scaler is not None:
            scale_cols = [c for c in feature_cols if c in user_frame.columns and c not in BINARY_FEATURE_COLS]
            if scale_cols:
                user_frame[scale_cols] = scaler.transform(user_frame[scale_cols])

        user_frame["pred_score"] = model.predict(user_frame[feature_cols])

        # Apply skill-first multi-signal reranking (includes confidence-aware
        # weight adjustment and weak-match penalty as of May 2026).
        user_frame = apply_multi_signal_rerank(user_frame, weights=SKILL_FIRST_RERANK_WEIGHTS)

        # ── Temporal diversity: soft decay for recently shown mentors ──
        # Mentors shown in recent sessions get a small multiplicative decay
        # so fresh mentors can surface. Strong mentors still rank high —
        # decay is soft (×0.95 for last session, ×0.97 for 2 sessions ago).
        if user_id is not None and "rerank_score" in user_frame.columns:
            recent = _recent_recommendations.get(user_id, [])
            if recent:
                # Most recent session (last entry) gets strongest decay
                for session_idx, shown_mentors in enumerate(reversed(recent)):
                    decay = 0.95 if session_idx == 0 else 0.97
                    mask = user_frame["mentor_id"].isin(shown_mentors)
                    user_frame.loc[mask, "rerank_score"] *= decay

        # ── Score calibration: confidence_score for internal use ──
        # Produces a separate `confidence_score` column with sigmoid-compressed
        # values for internal confidence interpretation.  Does NOT modify
        # rerank_score/pred_score — the UI match_percentage pipeline is
        # untouched.  Downstream consumers can use confidence_score for
        # confidence-aware decisions without affecting displayed percentages.
        import numpy as np
        if "rerank_score" in user_frame.columns and len(user_frame) > 1:
            rs = user_frame["rerank_score"]
            rs_median = rs.median()
            rs_std = rs.std()
            if rs_std > 0:
                z = 2.0 * (rs - rs_median) / rs_std
                user_frame["confidence_score"] = 1.0 / (1.0 + np.exp(-z))
            else:
                user_frame["confidence_score"] = 0.5
        else:
            user_frame["confidence_score"] = 0.5

        # Include signal columns for reason generation downstream
        _signal_cols = [
            "skill_overlap_score", "skill_coverage_score", "subdomain_similarity",
            "mentor_domain_match", "mentor_quality_score", "mentor_weighted_rating",
            "mentor_more_experienced", "same_country",
            "mentor_covers_all_skills", "confidence_score",
        ]
        keep_cols = ["mentee_id", "mentor_id", "rerank_score"]
        keep_cols += [c for c in _signal_cols if c in user_frame.columns]

        result = (
            user_frame.sort_values("rerank_score", ascending=False)
            .drop_duplicates(subset=["mentor_id"])
            .head(top_k)[keep_cols]
            .rename(columns={"rerank_score": "pred_score"})
            .reset_index(drop=True)
        )

        # Track shown mentors for temporal diversity (keep last 3 sessions)
        if user_id is not None:
            shown = set(result["mentor_id"].tolist())
            _track_and_trim_recent_recs(user_id, shown)

        return result

    mentee_features = bundle.get("mentee_features")
    mentor_features = bundle.get("mentor_features")
    if mentee_features is None or mentor_features is None:
        return _global_mentor_fallback(user_id, bundle.get("mentor_features", pd.DataFrame()), top_k)

    if "mentee_id" not in mentee_features.columns or "mentor_id" not in mentor_features.columns:
        return _global_mentor_fallback(user_id, mentor_features, top_k)

    if user_id is not None:
        user_row_df = mentee_features[mentee_features["mentee_id"] == user_id]
    else:
        user_row_df = pd.DataFrame()

    if user_row_df.empty:
        return _global_mentor_fallback(user_id, mentor_features, top_k)

    content_scores = _content_based_scores(user_row_df.iloc[0], mentor_features)
    if content_scores.empty:
        return _global_mentor_fallback(user_id, mentor_features, top_k)

    content_scores = content_scores.head(top_k).copy()
    content_scores.insert(0, "mentee_id", user_id)
    # Return all available columns (including signal columns for downstream formatting)
    return content_scores.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Auto-tuning
# ---------------------------------------------------------------------------

def tune_pipeline(raw_data=None, configs=None):
    """Run a parameter search with positive pair injection and coverage tracking.

    Tests each configuration end-to-end, injects positive pairs into every
    candidate pool, validates coverage, and applies early stopping.

    Early stop condition: coverage ≥ 60% AND NDCG improvement < 1% over
    previous best.

    Args:
        raw_data: None (default DB), Path to data directory, or dict of DataFrames.
        configs: Optional list of config dicts (max 8).  Each may contain:
            - high_priority (int): candidates from top tiers
            - low_priority (int): candidates from lower tiers
            - neg_per_pos (int): negative samples per positive
            - eval_k (int): k for NDCG/HitRate evaluation

    Returns:
        Dict with "results", "best", and "coverage_log".
    """
    import time as _time
    import numpy as np

    if configs is None:
        configs = [
            {"high_priority": 30, "low_priority": 10, "neg_per_pos": 4, "eval_k": 10},
            {"high_priority": 40, "low_priority": 15, "neg_per_pos": 4, "eval_k": 10},
            {"high_priority": 40, "low_priority": 15, "neg_per_pos": 6, "eval_k": 10},
            {"high_priority": 50, "low_priority": 15, "neg_per_pos": 6, "eval_k": 10},
            {"high_priority": 50, "low_priority": 20, "neg_per_pos": 8, "eval_k": 10},
            {"high_priority": 70, "low_priority": 20, "neg_per_pos": 6, "eval_k": 10},
        ]

    # Cap at 8 configs
    configs = configs[:8]

    if raw_data is None:
        raw_tables = load_db_datasets_from_db()
    elif isinstance(raw_data, (str, Path)):
        data_path = Path(raw_data)
        if (data_path / "programs.csv").exists():
            raw_tables = load_db_datasets(data_path)
        else:
            raw_tables = load_raw_datasets(data_path)
    elif isinstance(raw_data, dict):
        raw_tables = raw_data
    else:
        raise TypeError("raw_data must be None (default DB), a Path, or dict of DataFrames")

    config = build_time_split_config(
        raw_tables["mentorships"],
        applications=raw_tables.get("mentorship_applications"),
    )
    processed = prepare_processed_tables(raw_tables, config)

    mentorships_split_col = processed["mentorships"].get(
        "time_split", pd.Series("train", index=processed["mentorships"].index)
    )
    mentorships_train = processed["mentorships"][mentorships_split_col == "train"].copy()

    apps_all = processed["mentorship_applications"]
    apps_split_col = apps_all.get("time_split", pd.Series("train", index=apps_all.index))
    apps_train = apps_all[apps_split_col == "train"].copy()

    posts_mapping = raw_tables["mentorship_posts"][["post_id", "mentor_id"]].copy()
    posts_mapping["post_id"] = pd.to_numeric(posts_mapping["post_id"], errors="coerce")
    posts_mapping["mentor_id"] = pd.to_numeric(posts_mapping["mentor_id"], errors="coerce")
    posts_mapping = posts_mapping.dropna().drop_duplicates().astype({"post_id": int, "mentor_id": int})

    mentee_features = build_mentee_features(
        processed["mentee_profile"],
        processed["mentee_subdomains"],
        processed["mentee_interests"],
    )
    mentor_features = build_mentor_features(
        processed["mentor_profile"],
        processed["mentor_subdomains"],
        processed["mentor_expertise"],
        processed["mentors_feedback"],
        apps_train,
        processed["mentorship_posts"],
        processed["mentorship_cancellation"],
        mentorships_train,
        processed["follows"],
        train_end=config.train_end,
        likes_hist=processed.get("posts_likes_dataset"),
        comments_hist=processed.get("posts_comments"),
        saves_hist=processed.get("saved_posts_dataset"),
        shares_hist=processed.get("shared_posts_dataset"),
    )
    interaction_features = build_interaction_features(
        processed.get("posts_likes_dataset", pd.DataFrame()),
        processed.get("posts_comments", pd.DataFrame()),
        processed.get("saved_posts_dataset", pd.DataFrame()),
        processed.get("shared_posts_dataset", pd.DataFrame()),
        processed["follows"],
        processed["mentorship_posts"],
    )
    subdomains_map = processed.get("subdomains")

    # Build CF, community, requirement features (shared across configs)
    mentor_ids_for_cf = set(mentor_features["mentor_id"].dropna().astype(int).unique())
    cf_embeddings = build_cf_embeddings(
        interaction_features, processed["follows"], mentorships_train,
        processed["mentorship_posts"], mentor_ids_set=mentor_ids_for_cf, n_factors=16,
    )
    community_sets = build_community_membership_sets(
        processed.get("community_members", pd.DataFrame()),
    )
    mentor_requirement_sets = build_requirement_sets(
        processed.get("mentorship_requirements", pd.DataFrame()),
        processed["mentorship_posts"],
    )

    # Build positive pairs (shared across all configs)
    apps_with_mentor = apps_all.merge(posts_mapping, on="post_id", how="inner")
    if "mentee_id" not in apps_with_mentor.columns and "user_id" in apps_with_mentor.columns:
        apps_with_mentor = apps_with_mentor.rename(columns={"user_id": "mentee_id"})
    apps_with_mentor["mentee_id"] = pd.to_numeric(apps_with_mentor["mentee_id"], errors="coerce")
    apps_with_mentor["mentor_id"] = pd.to_numeric(apps_with_mentor["mentor_id"], errors="coerce")
    apps_with_mentor = apps_with_mentor.dropna(subset=["mentee_id", "mentor_id"]).copy()
    apps_with_mentor[["mentee_id", "mentor_id"]] = apps_with_mentor[["mentee_id", "mentor_id"]].astype(int)

    positive_pairs_by_split = {}
    for split_name in ("train", "valid", "test"):
        split_df = apps_with_mentor[apps_with_mentor["time_split"] == split_name]
        positive_pairs_by_split[split_name] = set(zip(split_df["mentee_id"], split_df["mentor_id"]))

    all_positive_pairs_df = pd.DataFrame(
        list(set().union(*positive_pairs_by_split.values())),
        columns=["mentee_id", "mentor_id"],
    )

    event_time_by_mentee = (
        apps_with_mentor
        .assign(applied_at=pd.to_datetime(apps_with_mentor["applied_at"], errors="coerce"))
        .dropna(subset=["applied_at"])
        .groupby("mentee_id")["applied_at"]
        .min()
    )

    results = []
    best_ndcg_so_far = 0.0
    hitrate_overfit_tolerance = 0.03
    logger.info("=" * 70)
    logger.info("AUTO-TUNING: testing %d configurations (with positive injection)", len(configs))
    logger.info("=" * 70)

    for i, cfg in enumerate(configs):
        hp = cfg.get("high_priority", 30)
        lp = cfg.get("low_priority", 10)
        npp = cfg.get("neg_per_pos", 4)
        ek = cfg.get("eval_k", 10)
        label = f"hp={hp} lp={lp} neg={npp} k={ek}"
        logger.info("\n--- Config %d/%d: %s ---", i + 1, len(configs), label)

        t0 = _time.time()
        try:
            # Generate candidate pool + inject positive pairs
            candidate_pool = generate_candidate_pool(
                mentee_features, mentor_features,
                subdomains_map=subdomains_map,
                top_k=30, min_candidates_per_mentee=10,
                high_priority_cap=hp, low_priority_cap=lp,
            )
            pool_before = len(candidate_pool)
            candidate_pool = pd.concat(
                [candidate_pool, all_positive_pairs_df[["mentee_id", "mentor_id"]]],
                ignore_index=True,
            ).drop_duplicates(["mentee_id", "mentor_id"]).reset_index(drop=True)
            injected = len(candidate_pool) - pool_before

            # Coverage check
            pool_set = set(zip(candidate_pool["mentee_id"], candidate_pool["mentor_id"]))
            cfg_coverage = {}
            for sn in ("train", "valid", "test"):
                sp = positive_pairs_by_split.get(sn, set())
                ip = len(sp & pool_set) if sp else 0
                cfg_coverage[sn] = round(100 * ip / len(sp), 1) if sp else 0.0
            logger.info(
                "  Coverage: train=%.1f%% valid=%.1f%% test=%.1f%% (injected=%d)",
                cfg_coverage["train"], cfg_coverage["valid"], cfg_coverage["test"], injected,
            )

            pair_base = build_pair_features(
                candidate_pool, mentee_features, mentor_features,
                interaction_features, processed["follows"],
                cf_embeddings=cf_embeddings,
                community_sets=community_sets,
                mentor_requirement_sets=mentor_requirement_sets,
                event_time_by_mentee=event_time_by_mentee,
                likes_hist=raw_tables.get("posts_likes_dataset"),
                comments_hist=raw_tables.get("posts_comments"),
                saves_hist=raw_tables.get("saved_posts_dataset"),
                shares_hist=raw_tables.get("shared_posts_dataset"),
                posts_hist=raw_tables.get("mentorship_posts"),
                mentorships_hist_raw=raw_tables.get("mentorships"),
                follows_hist_raw=raw_tables.get("follows"),
            )
            recommendation_features = build_recommendation_dataset(
                pair_base, positive_pairs_by_split, event_time_by_mentee,
                config.train_end, config.valid_end,
                neg_per_pos=npp,
                eval_neg_per_pos=EVAL_NEG_PER_POS,
                min_candidates_per_group=MIN_CANDIDATES_PER_GROUP,
                rng_seed=42,
            )
            recommendation_features, feature_cols = prepare_ranking_features(recommendation_features)
            train_df, valid_df, test_df = split_by_time(
                recommendation_features,
                {"train_end": config.train_end, "valid_end": config.valid_end},
            )
            scale_cols = [
                col for col in feature_cols
                if col in recommendation_features.columns and col not in BINARY_FEATURE_COLS
            ]
            train_df, valid_df, test_df, _ = scale_features(train_df, valid_df, test_df, scale_cols)
            model = train_model(train_df, valid_df, feature_cols)

            valid_scored = valid_df.copy()
            valid_scored["pred_score"] = model.predict(valid_scored[feature_cols])
            test_scored = test_df.copy()
            test_scored["pred_score"] = model.predict(test_scored[feature_cols])
            metrics_valid = evaluate_model(valid_scored, k=ek, min_candidates=MIN_EVAL_CANDIDATES)
            metrics_test = evaluate_model(test_scored, k=ek, min_candidates=MIN_EVAL_CANDIDATES)

            elapsed = _time.time() - t0
            ndcg_key = f"ndcg@{ek}"
            hit_key = f"hitrate@{ek}"
            hitrate_gap = metrics_valid.get(hit_key, 0.0) - metrics_test.get(hit_key, 0.0)
            result = {
                "config": cfg,
                "label": label,
                **{f"valid_{k}": v for k, v in metrics_valid.items()},
                **{f"test_{k}": v for k, v in metrics_test.items()},
                "hitrate_valid_test_gap": hitrate_gap,
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "test_rows": len(test_df),
                "train_positives": int(train_df["label"].sum()),
                "valid_positives": int(valid_df["label"].sum()),
                "test_positives": int(test_df["label"].sum()),
                "coverage": cfg_coverage,
                "elapsed_seconds": round(elapsed, 2),
            }
            results.append(result)

            logger.info(
                "  valid %s=%.4f valid %s=%.4f | test %s=%.4f test %s=%.4f | gap=%.4f | time=%.1fs",
                ndcg_key,
                metrics_valid.get(ndcg_key, 0),
                hit_key,
                metrics_valid.get(hit_key, 0),
                ndcg_key,
                metrics_test.get(ndcg_key, 0),
                hit_key,
                metrics_test.get(hit_key, 0),
                hitrate_gap,
                elapsed,
            )

            # Early stopping: coverage strong + no meaningful valid NDCG improvement.
            current_ndcg = metrics_valid.get(ndcg_key, 0)
            min_coverage = min(cfg_coverage.get("valid", 0), cfg_coverage.get("test", 0))
            if (
                min_coverage >= 60
                and best_ndcg_so_far > 0
                and current_ndcg <= best_ndcg_so_far * 1.01
                and i >= 2  # run at least 3 configs
            ):
                logger.info(
                    "  EARLY STOP: coverage=%.1f%% ≥ 60%%, valid NDCG improvement < 1%% (%.4f → %.4f)",
                    min_coverage, best_ndcg_so_far, current_ndcg,
                )
                break
            best_ndcg_so_far = max(best_ndcg_so_far, current_ndcg)

        except Exception as e:
            logger.error("  Config %s FAILED: %s", label, e)
            results.append({"config": cfg, "label": label, "error": str(e)})

    # Determine best config (valid NDCG first, then test NDCG, then stable HitRate gap)
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        stability_filtered = [
            r for r in valid_results
            if abs(float(r.get("hitrate_valid_test_gap", 0.0))) <= hitrate_overfit_tolerance
        ]
        pool = stability_filtered or valid_results

        def _score_key(r):
            eval_k = r["config"].get("eval_k", EVAL_K)
            ndcg_k = f"ndcg@{eval_k}"
            hit_k = f"hitrate@{eval_k}"
            return (
                r.get(f"valid_{ndcg_k}", 0),
                r.get(f"test_{ndcg_k}", 0),
                -abs(float(r.get("hitrate_valid_test_gap", 0.0))),
                r.get(f"test_{hit_k}", 0),
                min(r.get("coverage", {}).get("valid", 0), r.get("coverage", {}).get("test", 0)),
            )

        best = max(pool, key=_score_key)
        logger.info("\n" + "=" * 70)
        logger.info("BEST CONFIG: %s", best["label"])
        ndcg_k = f"ndcg@{best['config'].get('eval_k', EVAL_K)}"
        hit_k = f"hitrate@{best['config'].get('eval_k', EVAL_K)}"
        logger.info(
            "  valid %s=%.4f | test %s=%.4f",
            ndcg_k,
            best.get(f"valid_{ndcg_k}", 0),
            ndcg_k,
            best.get(f"test_{ndcg_k}", 0),
        )
        logger.info(
            "  valid %s=%.4f | test %s=%.4f | gap=%.4f",
            hit_k,
            best.get(f"valid_{hit_k}", 0),
            hit_k,
            best.get(f"test_{hit_k}", 0),
            float(best.get("hitrate_valid_test_gap", 0.0)),
        )
        cov = best.get("coverage", {})
        logger.info("  Coverage: train=%.1f%% valid=%.1f%% test=%.1f%%",
                     cov.get("train", 0), cov.get("valid", 0), cov.get("test", 0))
        logger.info("=" * 70)
    else:
        best = None
        logger.error("ALL configurations failed!")

    return {"results": results, "best": best}


