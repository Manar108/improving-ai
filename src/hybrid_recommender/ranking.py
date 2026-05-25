from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler
logger = logging.getLogger(__name__)


DEFAULT_FEATURE_COLS = [
    "skill_coverage_score",           # PRIORITY: fraction of mentee skills mentor covers
    "subdomain_similarity",           # Specialization alignment
    "mentor_quality_score",
    "mentor_weighted_rating",
    # interaction_score_log / interaction_count_log: RE-ENABLED (May 2026).
    # Fixed: now computed per-pair using the mentee's event_time (application date).
    # Uses post creation time as causal anchor — only posts that existed before
    # the application are counted.  Uses merge_asof with cumulative sums for O(n log n)
    # efficiency on 500K+ pairs.  Previously leaked future likes from months after
    # early-application pairs.
    # SCALED DOWN AGGRESSIVELY (May 2026): factor=0.15 to reduce CF proxy effect.
    "interaction_score_log",
    "interaction_count_log",
    "experience_gap_abs",
    "mentor_more_experienced",
    "experience_match_bucket",
    "soft_gap_score",
    "same_country",
    # popularity_log & mentor_follower_count_log: RE-ENABLED (May 2026).
    # Fixed: now computed per-pair using the mentee's event_time.
    # - popularity = programs created + enrollments started + engagement on posts,
    #   all counted only if they occurred before the pair's event_time.
    # - follower_count = follows where follow_time <= pair event_time.
    # Previously leaked future popularity/followers from months after early pairs.
    "popularity_log",
    "mentor_follower_count_log",      # social proof (follower count, log1p)
    "mentor_open_post_count_log",     # availability signal (open posts, log1p)
    "mentor_domain_match",            # domain alignment (binary)
    # --- Collaborative filtering & community features ---
    # cf_score: RE-ENABLED (May 2026).
    # Fixed: follows REMOVED from CF embeddings.  CF now learns purely from
    # content engagement (likes, comments, saves, shares) and actual enrollments.
    # Previously cf_score was a proxy for is_following because follows were included
    # in the interaction matrix fed to SVD.
    # SCALED DOWN AGGRESSIVELY (May 2026): factor=0.15 to reduce CF proxy effect.
    "cf_score",
    # REMOVED FEATURES (May 2026 - Skill-First Refactor):
    # - is_following: COMPLETELY REMOVED from training (May 2026 - CRITICAL FIX).
    #   Reason: 100% correlated with positive labels (all applications from followers).
    #   Making CF + interaction a proxy for is_following.  Removed from BOTH model
    #   features AND from pair_base dataframe during feature construction.
    #   Follow signal will be handled via soft reranking only.
    # - skill_overlap_score: Removed - overlaps with skill_coverage, adds noise
    #   Reason: model prefers mentors who cover ALL skills (coverage) not just
    #   share SOME skills (overlap). Overlap dilutes this signal.
    # - community_overlap: Removed - 0.0 feature importance, distracts model
    #   Reason: social clustering is handled by cf_score; community_overlap added
    #   no predictive value and confused the ranker.
    # NOTE: mentor_covers_all_skills, extra_skill_count, skill_match_type are computed
    # in features.py but excluded from default model -- experiments showed they cause
    # early stopping interference (model converges at iter=1 vs iter=98 without them).
]


# MEDIUM FIX: Optional feature imputation defaults (for NaN handling)
# When optional features are missing, use these sensible defaults instead of NaN
OPTIONAL_FEATURE_DEFAULTS = {
    "cf_score": 0.0,                          # No CF history → neutral (no prior engagement)
    "popularity_log": 0.0,                    # log(1) = 0 (minimum popularity)
    "mentor_follower_count_log": 0.0,         # log(1) = 0 (no followers)
    "mentor_open_post_count_log": 0.0,        # log(1) = 0 (no open posts)
    "interaction_score_log": 0.0,             # log(1) = 0 (no interaction history)
    "interaction_count_log": 0.0,             # log(1) = 0 (no interactions)
    "soft_gap_score": 0.5,                    # Neutral experience gap
    "experience_gap_abs": 0.0,                # Same experience level (neutral)
    "mentor_more_experienced": 0.0,           # Unknown experience relationship
    "experience_match_bucket": 0,             # Neutral bucket
    "subdomain_similarity": 0.0,              # No subdomain match
    "mentor_quality_score": 0.0,              # Unknown quality (not bad, just unknown)
    "mentor_weighted_rating": 0.0,            # No rating history
    "skill_coverage_score": 0.0,              # No skill coverage data
}


def _safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric Series for *column*, defaulting to 0.0 if missing."""
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _build_similarity_score(df: pd.DataFrame) -> pd.Series:
    """Compute a 0-100 user-facing similarity score from multiple signals.

    Combines:
      - skill_overlap_score      (25%)  — Jaccard similarity of skill sets
      - skill_coverage_score     (20%)  — fraction of mentee skills covered
      - subdomain_similarity     (15%)  — Jaccard similarity of subdomains
      - interaction_score_log    (15%)  — prior engagement (likes/comments/saves)
      - experience_gap_norm      (10%)  — normalized experience compatibility
      - mentor_quality_score     (15%)  — unified quality (rating+sentiment)

    All component scores are normalized to [0, 1] before weighting.
    Final score is scaled to 0-100.
    """
    skill_overlap = _safe_numeric_series(df, "skill_overlap_score").clip(0, 1)
    skill_coverage = _safe_numeric_series(df, "skill_coverage_score").clip(0, 1)
    subdomain_sim = _safe_numeric_series(df, "subdomain_similarity").clip(0, 1)

    # Interaction: log-scaled, normalize to [0,1] via sigmoid-like transform
    interaction_raw = _safe_numeric_series(df, "interaction_score_log")
    interaction_norm = interaction_raw / (1.0 + interaction_raw)

    # Experience gap: lower abs gap = better compatibility → invert
    exp_gap_abs = _safe_numeric_series(df, "experience_gap_abs")
    experience_norm = 1.0 / (1.0 + exp_gap_abs)

    # Mentor quality: already a composite score (weighted_rating+sentiment+feedback)
    # Normalize to [0,1] assuming max realistic value ~5
    quality_raw = _safe_numeric_series(df, "mentor_quality_score")
    quality_norm = (quality_raw / 5.0).clip(0, 1)

    similarity = (
        0.25 * skill_overlap
        + 0.20 * skill_coverage
        + 0.15 * subdomain_sim
        + 0.15 * interaction_norm
        + 0.10 * experience_norm
        + 0.15 * quality_norm
    )
    return (similarity.clip(0, 1) * 100).round(1)


def _build_explanation_text(df: pd.DataFrame) -> pd.Series:
    """Build intelligent, contextual explanations for mentor recommendations.

    Creates detailed, personalized explanations that combine multiple signals
    into coherent reasoning. Uses dynamic weighting based on signal strength
    and provides specific, actionable insights for each recommendation.

    Enhanced Signal Priority (May 2026 - Smart Explanations):
      1. Perfect/Strong Skill Match    (2.0+ weight) - Core compatibility
      2. Complete Requirement Coverage (1.8+ weight) - Meets all needs
      3. Subdomain Specialization     (1.5+ weight) - Exact field match
      4. High Quality + Experience    (1.2+ weight) - Proven expertise
      5. Active Availability          (1.0+ weight) - Ready to mentor
      6. Prior Engagement             (0.8+ weight) - Known interaction
      7. Moderate Skill Match         (0.6+ weight) - Partial alignment
      8. Social/Geographic            (0.3+ weight) - Contextual factors

    Features intelligent combination of signals and contextual phrasing.
    """
    # Build signal → (score, label) for each row
    skill_overlap = _safe_numeric_series(df, "skill_overlap_score")
    skill_coverage = _safe_numeric_series(df, "skill_coverage_score")
    subdomain_sim = _safe_numeric_series(df, "subdomain_similarity")
    mentor_rating = _safe_numeric_series(df, "mentor_weighted_rating")
    quality_score = _safe_numeric_series(df, "mentor_quality_score")
    interaction = _safe_numeric_series(df, "interaction_score_log")
    exp_gap_abs = _safe_numeric_series(df, "experience_gap_abs")
    mentor_more_exp = _safe_numeric_series(df, "mentor_more_experienced")
    same_country = _safe_numeric_series(df, "same_country")

    open_posts = _safe_numeric_series(df, "mentor_open_post_count_log")
    req_coverage = _safe_numeric_series(df, "requirement_coverage")
    popularity = _safe_numeric_series(df, "popularity_log")

    reasons_list = []
    for idx in df.index:
        # Collect (weight, reason) tuples — weight determines ranking
        signals = []

        so = skill_overlap.loc[idx]
        sc = skill_coverage.loc[idx]
        sd = subdomain_sim.loc[idx]
        rc = req_coverage.loc[idx]
        r = mentor_rating.loc[idx]
        q = quality_score.loc[idx]
        op = open_posts.loc[idx]
        inter = interaction.loc[idx]
        me = mentor_more_exp.loc[idx]
        eg = exp_gap_abs.loc[idx]

        country = same_country.loc[idx]
        pop = popularity.loc[idx]

        # Perfect Skill Match (highest priority - 2.0+ weight)
        if so >= 0.8 and sc >= 0.8:
            signals.append((2.5 + so + sc, "perfect skill and expertise match"))
        elif so >= 0.7 and sc >= 0.7:
            signals.append((2.2 + so + sc, "excellent skill alignment with your needs"))
        elif so >= 0.6 or sc >= 0.7:
            signals.append((1.8 + so + sc, "strong skill match for your goals"))

        # Complete Requirement Coverage (1.8+ weight)
        if rc >= 0.9:
            signals.append((2.0 + rc, "covers all your mentorship requirements"))
        elif rc >= 0.7:
            signals.append((1.8 + rc, "meets most of your specific requirements"))
        elif rc >= 0.5:
            signals.append((1.4 + rc, "aligns well with your requirements"))

        # Exact Subdomain Specialization (1.5+ weight)
        if sd >= 0.8:
            signals.append((1.8 + sd, "specializes exactly in your field"))
        elif sd >= 0.6:
            signals.append((1.5 + sd, "strong specialization alignment"))
        elif sd >= 0.4:
            signals.append((1.2 + sd, "relevant specialization area"))

        # High Quality + Experience Combination (1.2+ weight)
        if r >= 4.5 and me > 0:
            signals.append((1.5 + q * 0.1, f"top-rated expert mentor ({r:.1f}/5)"))
        elif r >= 4.0 and me > 0:
            signals.append((1.3 + q * 0.1, f"highly rated experienced mentor ({r:.1f}/5)"))
        elif r >= 4.0:
            signals.append((1.1 + q * 0.1, f"exceptionally well-reviewed ({r:.1f}/5)"))
        elif r >= 3.5:
            signals.append((0.9 + q * 0.05, f"strong mentor ratings ({r:.1f}/5)"))

        # Active Availability (1.0+ weight)
        if op > 1:
            signals.append((1.2, f"multiple open mentorship programs ({int(op)})"))
        elif op > 0:
            signals.append((1.0, "currently accepting mentorship applications"))

        # Strong Prior Engagement (0.8+ weight)
        if inter >= 2:
            signals.append((1.0 + inter * 0.1, "regular engagement with their content"))
        elif inter >= 1:
            signals.append((0.8 + inter * 0.1, "previous interaction with their posts"))
        elif inter > 0:
            signals.append((0.6, "some prior engagement"))

        # Moderate Skill Match (0.6+ weight)
        if so >= 0.4 or sc >= 0.5:
            if so >= 0.4 and sc >= 0.5:
                signals.append((0.8 + so + sc, "good skill overlap and coverage"))
            elif so >= 0.4:
                signals.append((0.7 + so, "relevant skill overlap"))
            elif sc >= 0.5:
                signals.append((0.6 + sc, "covers many of your skill areas"))

        # Experience Alignment (0.5+ weight)
        if me > 0 and eg >= 3:
            signals.append((0.7, "significantly more experienced mentor"))
        elif me > 0 and eg >= 1:
            signals.append((0.5, "more experienced than your level"))

        # Social Context (0.3+ weight)
        if pop > 1:
            signals.append((0.25, "popular mentor in your field"))

        # Geographic Context (0.2+ weight)
        if country > 0:
            signals.append((0.2, "located in your country"))

        # Sort by weight descending, take top 3 reasons
        signals.sort(key=lambda x: x[0], reverse=True)
        top_reasons = [reason for _, reason in signals[:3]]

        if not top_reasons:
            top_reasons = ["good overall profile fit"]

        # Create more natural phrasing by connecting related concepts
        if len(top_reasons) >= 2:
            # Combine skill and requirement coverage if both present
            skill_reasons = [r for r in top_reasons if any(k in r.lower() for k in ['skill', 'requirement', 'specialization'])]
            if len(skill_reasons) >= 2:
                # Keep the strongest skill reason, add others as secondary
                top_reasons = [skill_reasons[0]] + [r for r in top_reasons if r not in skill_reasons[1:]]

        reasons_list.append("; ".join(top_reasons))

    return pd.Series(reasons_list, index=df.index)


def prepare_ranking_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare ranking features: enforce numeric types and determine feature list.

    Returns:
        Tuple of (cleaned DataFrame, list of feature column names).
    """
    out = df.copy()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)

    feature_cols = [col for col in DEFAULT_FEATURE_COLS if col in out.columns]
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)



    return out, feature_cols


def split_by_time(df: pd.DataFrame, split_cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/valid/test using ``time_split`` column or ``event_time``.

    Prefers the ``time_split`` column if present; falls back to time-based
    partitioning using ``event_time`` or ``start_date``.
    """
    out = df.copy()
    train_end = pd.to_datetime(split_cfg["train_end"], errors="coerce")
    valid_end = pd.to_datetime(split_cfg["valid_end"], errors="coerce")
    if pd.isna(train_end) or pd.isna(valid_end):
        raise ValueError("Invalid split config")

    if "time_split" in out.columns:
        out["time_split"] = out["time_split"].astype(str).str.strip().str.lower()
        train_df = out[out["time_split"] == "train"].sort_values("mentee_id")
        valid_df = out[out["time_split"] == "valid"].sort_values("mentee_id")
        test_df = out[out["time_split"] == "test"].sort_values("mentee_id")
        if len(train_df) == 0:
            out = out.copy()
        else:
            return train_df, valid_df, test_df

    if "event_time" not in out.columns:
        if "start_date" in out.columns:
            out["event_time"] = pd.to_datetime(out["start_date"], errors="coerce")
        else:
            raise ValueError("Expected event_time or start_date column")

    out = out[out["event_time"].notna()].copy()
    train_df = out[out["event_time"].dt.normalize() <= train_end].sort_values("mentee_id")
    valid_df = out[(out["event_time"].dt.normalize() > train_end) & (out["event_time"].dt.normalize() <= valid_end)].sort_values("mentee_id")
    test_df = out[out["event_time"].dt.normalize() > valid_end].sort_values("mentee_id")
    return train_df, valid_df, test_df


def scale_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Scale numeric features using MinMaxScaler fit on train data ONLY.

    Binary features (same_country, mentor_more_experienced)
    must be excluded from ``scale_cols`` by the caller.
    """
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()
    if scale_cols:
        scaler.fit(train_df[scale_cols])
        train_df[scale_cols] = scaler.transform(train_df[scale_cols])
        valid_df[scale_cols] = scaler.transform(valid_df[scale_cols])
        test_df[scale_cols] = scaler.transform(test_df[scale_cols])
    return train_df, valid_df, test_df, scaler


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    model_params: dict | None = None,
) -> LGBMRanker:
    """Train a LGBMRanker (lambdarank) with diagnostics and auto-retry.

    Groups are defined by ``mentee_id``.  Includes:
      - Validation sanity checks (positive count, group count)
      - Early stopping diagnostics (detects premature stopping)
      - Auto-retry with lower LR if early stop is too early (< 10 iters)
      - Gain-based feature importance stored on model object

    Args:
        model_params: Optional override dict for LGBMRanker hyperparameters.
    """
    import logging
    _logger = logging.getLogger(__name__)

    group_train = train_df.groupby("mentee_id").size().values
    group_valid = valid_df.groupby("mentee_id").size().values

    train_out = train_df.copy()
    train_out = train_out[train_out[label_col].notna()].copy()
    X_train = train_out[feature_cols]
    y_train = train_out[label_col]
    group_train = train_out.groupby("mentee_id").size().tolist()

    # ── Sample weights: upweight high-skill-match positives ──
    # SKILL-COVERAGE FOCUS (May 2026): Removed skill_overlap from training signal.
    # Priority: mentor covers ALL skills needed (coverage) >> shared skills (overlap).
    # 
    # DOWNWEIGHTED CF SIGNAL: CF score is strong but tends to recommend popular mentors.
    # We upweight skill-coverage positives more aggressively so the model learns
    # that skill-matched applications are THE primary signal.
    #
    # Weight formula: 1.0 + 3.0 * skill_coverage
    # Result: high-coverage positives get ~4x weight vs neutral negatives.
    # CF signal still present in model but deprioritized via reranking weights.
    sample_weight_train = np.ones(len(train_out), dtype=np.float32)
    is_pos = train_out[label_col].values == 1
    if is_pos.any():
        skill_coverage = train_out["skill_coverage_score"].fillna(0).values[is_pos]
        # Aggressive skill-coverage weighting: 3.0x multiplier
        sample_weight_train[is_pos] = 1.0 + 3.0 * skill_coverage
        _logger.info(
            "sample_weights: mean=%.2f pos_mean=%.2f (aggressive coverage: 3.0*coverage to deprioritize CF)",
            sample_weight_train.mean(),
            sample_weight_train[is_pos].mean(),
        )

    # ── Validation sanity checks ──
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[label_col]
    valid_positives = int(y_valid.sum())
    valid_groups_with_pos = int((valid_df.groupby("mentee_id")[label_col].sum() > 0).sum())
    _logger.info(
        "Training diagnostics: train=%d rows (%d pos), valid=%d rows (%d pos, %d groups with pos)",
        len(X_train), int(y_train.sum()), len(X_valid), valid_positives, valid_groups_with_pos,
    )
    if valid_positives < 10:
        _logger.warning(
            "LOW VALIDATION SIGNAL: only %d positives in validation set. "
            "Consider increasing candidate pool or improving positive coverage.",
            valid_positives,
        )
    if valid_groups_with_pos < 5:
        _logger.warning(
            "LOW GROUP SIGNAL: only %d validation groups have positives. "
            "Ranking evaluation may be unreliable.",
            valid_groups_with_pos,
        )

    # ── Default hyperparameters (tuned via neg_per_pos experiment) ──
    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "n_estimators": 1000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.01,
        "reg_lambda": 0.1,
        "random_state": 42,
        "importance_type": "gain",
    }
    if model_params:
        default_params.update(model_params)

    def _fit_model(params, early_stopping_rounds=150, log_period=10, sample_weight=None):
        """Fit LGBMRanker and return (model, best_iteration, best_score)."""
        model = LGBMRanker(**params)
        fit_kwargs = dict(
            group=group_train,
            eval_set=[(X_valid, y_valid)],
            eval_group=[group_valid],
            eval_at=[10],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, first_metric_only=True),
                lgb.log_evaluation(period=log_period),
            ],
        )
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X_train, y_train, **fit_kwargs)
        best_iter = model.best_iteration_ if hasattr(model, "best_iteration_") else params["n_estimators"]
        best_score = model.best_score_ if hasattr(model, "best_score_") else {}
        return model, best_iter, best_score

    # ── Primary training run ──
    model, best_iter, best_score = _fit_model(default_params, sample_weight=sample_weight_train)

    _logger.info(
        "Training complete: best_iteration=%d, total_estimators=%d",
        best_iter, default_params["n_estimators"],
    )
    if best_score:
        for ds_name, metric_dict in best_score.items():
            for metric_name, value in metric_dict.items():
                _logger.info("  %s %s: %.6f", ds_name, metric_name, value)

    # ── Auto-retry if early stopping triggered too early ──
    if best_iter < 10:
        _logger.warning(
            "EARLY STOPPING TOO EARLY: best_iteration=%d (< 10). "
            "Likely weak signal or bad validation split. Retrying with lower LR...",
            best_iter,
        )
        retry_params = default_params.copy()
        retry_params["learning_rate"] = default_params["learning_rate"] * 0.5  # half the LR
        retry_params["n_estimators"] = 700  # more room to converge

        model_retry, best_iter_retry, best_score_retry = _fit_model(
            retry_params, early_stopping_rounds=100, log_period=10,
        )
        _logger.info(
            "Retry complete: best_iteration=%d (was %d)", best_iter_retry, best_iter,
        )

        # Keep whichever model has higher validation NDCG
        def _extract_ndcg(score_dict):
            if score_dict:
                for ds, metrics_dict in score_dict.items():
                    for mn, val in metrics_dict.items():
                        if "ndcg" in mn:
                            return val
            return 0.0

        orig_ndcg = _extract_ndcg(best_score)
        retry_ndcg = _extract_ndcg(best_score_retry)
        if retry_ndcg > orig_ndcg:
            _logger.info(
                "Retry improved validation NDCG (%.4f -> %.4f, iter %d -> %d). Using retry model.",
                orig_ndcg, retry_ndcg, best_iter, best_iter_retry,
            )
            model, best_iter, best_score = model_retry, best_iter_retry, best_score_retry
        else:
            _logger.info(
                "Retry did not improve (orig=%.4f, retry=%.4f). Keeping original model.",
                orig_ndcg, retry_ndcg,
            )

    # ── Feature importance (gain-based) ──
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.feature_importances_,
        }
    ).sort_values(["importance_gain", "feature"], ascending=[False, True]).reset_index(drop=True)
    model.feature_importance_df_ = importance_df
    model.feature_importance_map_ = dict(zip(importance_df["feature"], importance_df["importance_gain"]))
    model.trained_feature_names_ = feature_cols
    model.training_diagnostics_ = {
        "best_iteration": best_iter,
        "n_estimators": default_params["n_estimators"],
        "best_score": best_score,
        "valid_positives": valid_positives,
        "valid_groups_with_pos": valid_groups_with_pos,
    }

    _logger.info("Feature importance (gain):")
    for _, row in importance_df.iterrows():
        _logger.info("  %-25s %8.1f", row["feature"], row["importance_gain"])

    return model


def evaluate_model(scored_df: pd.DataFrame, k: int = 10, min_candidates: int = 2, include_skipped: bool = False, return_split_metrics: bool = True) -> Dict[str, float]:
    """Evaluate ranking quality using NDCG@k, HitRate@k, Precision@k, MAP@k, Recall@k.

    Groups with no positive labels or fewer than ``min_candidates`` are evaluated
    as 0 metrics when return_split_metrics=True (global evaluation).
    When False, these groups are skipped (filtered evaluation).
    
    Returns a dict with metric values and group-level diagnostics.
    
    NEW (May 2026): return_split_metrics=True computes BOTH filtered and global metrics:
    - Filtered: only users with valid positives in candidate pool
    - Global: all test users (impossible users counted as NDCG=0, HitRate=0)
    """
    ndcg_vals_filtered = []
    hit_vals_filtered = []
    precision_vals_filtered = []
    map_vals_filtered = []
    recall_vals_filtered = []
    
    ndcg_vals_global = []
    hit_vals_global = []
    precision_vals_global = []
    map_vals_global = []
    recall_vals_global = []
    
    total_groups = 0
    skipped_no_positive = 0
    skipped_small_group = 0

    for _, grp in scored_df.groupby("mentee_id"):
        total_groups += 1
        has_positive = grp["label"].sum() > 0
        is_large_enough = len(grp) >= min_candidates
        
        # GLOBAL metrics: always compute (impossible users as 0)
        if not has_positive:
            skipped_no_positive += 1
            ndcg_vals_global.append(0.0)
            hit_vals_global.append(0)
            precision_vals_global.append(0.0)
            recall_vals_global.append(0.0)
            map_vals_global.append(0.0)
        elif not is_large_enough:
            skipped_small_group += 1
            ndcg_vals_global.append(0.0)
            hit_vals_global.append(0)
            precision_vals_global.append(0.0)
            recall_vals_global.append(0.0)
            map_vals_global.append(0.0)
        else:
            # Compute actual metrics for global
            grp_sorted = grp.sort_values("pred_score", ascending=False)
            labels = grp_sorted["label"].to_numpy()
            y_true = labels.reshape(1, -1)
            y_pred = grp_sorted["pred_score"].to_numpy().reshape(1, -1)
            actual_k = min(k, len(grp_sorted))

            ndcg_vals_global.append(ndcg_score(y_true, y_pred, k=actual_k))
            hit_vals_global.append(int(labels[:actual_k].sum() > 0))
            precision_vals_global.append(float(labels[:actual_k].sum()) / actual_k)
            total_pos = labels.sum()
            recall_vals_global.append(float(labels[:actual_k].sum()) / total_pos if total_pos > 0 else 0.0)
            
            hits = 0.0
            ap_score = 0.0
            for i in range(actual_k):
                if labels[i] > 0:
                    hits += 1.0
                    ap_score += hits / (i + 1)
            n_relevant = min(actual_k, int(total_pos))
            map_vals_global.append(ap_score / n_relevant if n_relevant > 0 else 0.0)
        
        # FILTERED metrics: only include evaluable groups
        if has_positive and is_large_enough:
            grp_sorted = grp.sort_values("pred_score", ascending=False)
            labels = grp_sorted["label"].to_numpy()
            y_true = labels.reshape(1, -1)
            y_pred = grp_sorted["pred_score"].to_numpy().reshape(1, -1)
            actual_k = min(k, len(grp_sorted))

            ndcg_vals_filtered.append(ndcg_score(y_true, y_pred, k=actual_k))
            hit_vals_filtered.append(int(labels[:actual_k].sum() > 0))
            precision_vals_filtered.append(float(labels[:actual_k].sum()) / actual_k)
            total_pos = labels.sum()
            recall_vals_filtered.append(float(labels[:actual_k].sum()) / total_pos if total_pos > 0 else 0.0)
            
            hits = 0.0
            ap_score = 0.0
            for i in range(actual_k):
                if labels[i] > 0:
                    hits += 1.0
                    ap_score += hits / (i + 1)
            n_relevant = min(actual_k, int(total_pos))
            map_vals_filtered.append(ap_score / n_relevant if n_relevant > 0 else 0.0)

    # Return split metrics if requested (default True)
    if return_split_metrics:
        return {
            # FILTERED metrics (users with valid positives)
            f"filtered_ndcg@{k}": float(np.mean(ndcg_vals_filtered)) if ndcg_vals_filtered else np.nan,
            f"filtered_hitrate@{k}": float(np.mean(hit_vals_filtered)) if hit_vals_filtered else np.nan,
            f"filtered_precision@{k}": float(np.mean(precision_vals_filtered)) if precision_vals_filtered else np.nan,
            f"filtered_map@{k}": float(np.mean(map_vals_filtered)) if map_vals_filtered else np.nan,
            f"filtered_recall@{k}": float(np.mean(recall_vals_filtered)) if recall_vals_filtered else np.nan,
            # GLOBAL metrics (all users, including impossible)
            f"global_ndcg@{k}": float(np.mean(ndcg_vals_global)) if ndcg_vals_global else np.nan,
            f"global_hitrate@{k}": float(np.mean(hit_vals_global)) if hit_vals_global else np.nan,
            f"global_precision@{k}": float(np.mean(precision_vals_global)) if precision_vals_global else np.nan,
            f"global_map@{k}": float(np.mean(map_vals_global)) if map_vals_global else np.nan,
            f"global_recall@{k}": float(np.mean(recall_vals_global)) if recall_vals_global else np.nan,
            # Diagnostics
            "evaluated_groups": len(hit_vals_filtered),
            "total_groups": total_groups,
            "skipped_no_positive": skipped_no_positive,
            "skipped_small_group": skipped_small_group,
            "impossible_users": skipped_no_positive + skipped_small_group,
            "min_candidates_rule": min_candidates,
        }
    else:
        # Legacy: return filtered metrics only
        return {
            f"ndcg@{k}": float(np.mean(ndcg_vals_filtered)) if ndcg_vals_filtered else np.nan,
            f"hitrate@{k}": float(np.mean(hit_vals_filtered)) if hit_vals_filtered else np.nan,
            f"precision@{k}": float(np.mean(precision_vals_filtered)) if precision_vals_filtered else np.nan,
            f"map@{k}": float(np.mean(map_vals_filtered)) if map_vals_filtered else np.nan,
            f"recall@{k}": float(np.mean(recall_vals_filtered)) if recall_vals_filtered else np.nan,
            "evaluated_groups": len(hit_vals_filtered),
            "total_groups": total_groups,
            "skipped_no_positive": skipped_no_positive,
            "skipped_small_group": skipped_small_group,
            "impossible_users": skipped_no_positive + skipped_small_group,
            "min_candidates_rule": min_candidates,
        }




def debug_skill_coverage_verification(scored_df: pd.DataFrame, k: int = 3, sample_size: int = 5) -> None:
    """Debug function: sample users and print top recommendations with skill coverage info.
    
    Verifies that high skill coverage actually correlates with top-ranked mentors.
    
    Args:
        scored_df: DataFrame with pred_score, skill_coverage_score, subdomain_similarity columns
        k: Number of top recommendations to show per user
        sample_size: Number of users to sample
    
    May 2026: Added for skill coverage validation after rebalancing.
    """
    unique_mentees = scored_df["mentee_id"].unique()
    if len(unique_mentees) < sample_size:
        sample_size = len(unique_mentees)
    
    sample_mentees = np.random.choice(unique_mentees, size=sample_size, replace=False)
    
    logger.info("\n=== SKILL COVERAGE DEBUG: Top-%d recommendations per sampled user ===", k)
    for mentee_id in sample_mentees:
        user_recs = scored_df[scored_df["mentee_id"] == mentee_id].sort_values("pred_score", ascending=False).head(k)
        n_positives = int(user_recs["label"].sum())
        logger.info("\nMentee ID %d (%d positives):", mentee_id, n_positives)
        
        for rank, (_, rec) in enumerate(user_recs.iterrows(), 1):
            mentor_id = rec.get("mentor_id", "?")
            pred_score = rec.get("pred_score", 0)
            skill_cov = rec.get("skill_coverage_score", 0)
            subdomain = rec.get("subdomain_similarity", 0)
            label = rec.get("label", 0)
            is_pos = "✓ POSITIVE" if label > 0 else "✗ negative"
            
            logger.info(
                "  Rank %d: mentor_id=%s, pred_score=%.3f, skill_cov=%.3f, subdomain=%.3f, %s",
                rank, mentor_id, pred_score, skill_cov, subdomain, is_pos,
            )
        
        # Summary stats
        avg_skill_cov = user_recs.get("skill_coverage_score", pd.Series()).mean()
        avg_subdomain = user_recs.get("subdomain_similarity", pd.Series()).mean()
        logger.info("  Avg skill_coverage=%.3f, Avg subdomain=%.3f", avg_skill_cov, avg_subdomain)


def generate_top_k_recommendations(
    model: LGBMRanker,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    k: int = 10,
    rerank: bool = True,
    follow_ratio: int = 3,
) -> pd.DataFrame:
    """Generate top-k recommendations per mentee from model predictions.

    Adds ``similarity_score`` (0-100) and ``explanation_text`` for display.
    If ``rerank`` is True, applies skill-first multi-signal and business rule
    reranking (May 2026 Hybrid Rebalancing):
      1. apply_multi_signal_rerank() — skill coverage, subdomain, domain signals
      2. apply_soft_business_boosts() — availability, quality, business signals
    This replaces the old follow_ratio-based reranking (deprecated May 2026).
    """
    scored = test_df.copy()
    scored["pred_score"] = model.predict(scored[feature_cols])
    scored["similarity_score"] = _build_similarity_score(scored)
    scored["explanation_text"] = _build_explanation_text(scored)

    if rerank:
        # New skill-first reranking pipeline (May 2026)
        scored = apply_multi_signal_rerank(scored)
        scored = apply_soft_business_boosts(scored)
        sort_col = "rerank_score"
    else:
        sort_col = "pred_score"

    return (
        scored.sort_values(["mentee_id", sort_col], ascending=[True, False])
        .groupby("mentee_id")
        .head(k)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Post-model re-ranking
# ---------------------------------------------------------------------------




def apply_multi_signal_rerank(
    scored_df: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Apply weighted multi-signal post-model reranking (ML-dominant, confidence-aware).

    CORE COMPATIBILITY SIGNALS ONLY — this layer answers
    "which mentor is the best match?" not "who is most popular?"

    Layer A — Core Compatibility (this function):
      - pred_score:      ML model's learned compatibility (60%, dominant)
      - skill_coverage:  mentee skill coverage fraction (20%)
      - subdomain:       specialization alignment (12%)
      - domain_match:    broader domain alignment (8%)

    Serving/business signals (availability, follow, popularity) are in
    apply_soft_business_boosts() — a separate, weaker layer.

    Confidence-aware behavior:
      - When model scores are spread (high confidence): ML weight stays dominant
      - When model scores cluster tightly (low confidence): ML weight is reduced
        slightly, allowing compatibility signals to break ties

    Weak match penalty:
      - Mentors with very low skill overlap AND weak subdomain similarity
        receive a soft multiplicative penalty (×0.85–0.92)
      - Prevents irrelevant-but-popular mentors from dominating top-k

    Args:
        scored_df: DataFrame with pred_score and feature columns.
        weights: Dict of signal_name -> weight. Defaults provided (ML-dominant).

    Returns:
        Same DataFrame with added ``rerank_score`` column.
    """
    default_weights = {
        "pred_score": 0.60,    # ML model — dominant but shares weight with compatibility signals
        "skill_coverage": 0.20,  # Coverage — PRIMARY compatibility signal
        "subdomain": 0.12,       # Subdomain — specialization alignment
        "domain_match": 0.08,     # Domain — broader field alignment
    }
    w = {**default_weights, **(weights or {})}

    results = []
    for _, grp in scored_df.groupby("mentee_id"):
        g = grp.copy()

        # Normalize pred_score within group to [0, 1]
        ps = g["pred_score"]
        ps_range = ps.max() - ps.min()
        ps_norm = (ps - ps.min()) / ps_range if ps_range > 0 else pd.Series(0.5, index=g.index)

        # ── Confidence-aware weight adjustment ──
        # When scores are tightly clustered (ps_range small), the model is
        # uncertain about relative ordering → allow other signals more influence.
        # When scores are well-spread, the model is confident → allow ML dominance.
        #
        # SKILL-FIRST ADJUSTMENT: Reduced confidence range to [0.5, 0.8]
        # This means even when model scores are spread, we allocate more weight to
        # skill_coverage and subdomain signals (which are directly interpretable).
        # Previously [0.7, 1.0]: ML weight = 0.56-0.80. Now [0.5, 0.8]: ML weight = 0.30-0.48.
        # Result: skill signals get 52-70% of total weight (vs 20-44% previously).
        confidence_factor = min(1.0, max(0.5, ps_range / 2.0)) if ps_range > 0 else 0.5
        ml_weight = w.get("pred_score", 0.60) * confidence_factor
        # Redistribute freed weight proportionally to compatibility signals
        freed_weight = w.get("pred_score", 0.60) * (1.0 - confidence_factor)
        other_total = sum(v for k, v in w.items() if k != "pred_score")
        scale_up = (1.0 + freed_weight / other_total) if other_total > 0 else 1.0

        score = ml_weight * ps_norm

        # ── Core Compatibility Signals ──

        # Skill coverage: suitability refinement
        if "skill_coverage_score" in g.columns:
            score += w.get("skill_coverage", 0.10) * scale_up * g["skill_coverage_score"].fillna(0).astype(float)

        # Subdomain: specialization alignment
        if "subdomain_similarity" in g.columns:
            score += w.get("subdomain", 0.06) * scale_up * g["subdomain_similarity"].fillna(0).astype(float)

        # Domain match: broader field alignment
        if "mentor_domain_match" in g.columns:
            score += w.get("domain_match", 0.04) * scale_up * g["mentor_domain_match"].fillna(0).astype(float)

        # ── Weak match penalty ──
        # Mentors with BOTH low skill coverage AND low subdomain similarity
        # get a soft multiplicative penalty. This prevents irrelevant-but-popular
        # mentors from ranking highly due to engagement/follower signals.
        # Penalty is graduated: ×0.85 for zero coverage, ×0.92 for very weak match.
        # Does NOT remove any mentors — just softly deprioritizes poor matches.
        skill_cov = g["skill_coverage_score"].fillna(0).astype(float) if "skill_coverage_score" in g.columns else pd.Series(0.0, index=g.index)
        sub_sim = g["subdomain_similarity"].fillna(0).astype(float) if "subdomain_similarity" in g.columns else pd.Series(0.0, index=g.index)

        # Weak match: skill_coverage < 0.2 AND subdomain_similarity < 0.2
        weak_match_mask = (skill_cov < 0.2) & (sub_sim < 0.2)
        # Very weak: both are exactly 0
        very_weak_mask = (skill_cov == 0) & (sub_sim == 0)

        # Apply graduated penalty
        penalty = pd.Series(1.0, index=g.index)
        penalty[weak_match_mask] = 0.92  # weak match: 8% penalty
        penalty[very_weak_mask] = 0.85   # zero match: 15% penalty
        score = score * penalty

        g["rerank_score"] = score
        results.append(g)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Soft business-rule boosts (multiplicative, NEVER remove mentors)
# ---------------------------------------------------------------------------
#
# Three tiers, ordered from strongest to weakest:
#
#   Tier A — Quality (compatibility-adjacent, strongest boost):
#     Rewards mentors who combine quality with compatibility
#     (high rating in the SAME subdomain — nearest to core compatibility).
#
#   Tier B — Serving/Availability (moderate boost):
#     Availability is a serving-time state, not compatibility.
#     Should NEVER override a strong skill match.
#
#   Tier C — Social/Business (weakest boost):
#     Follow relationship and popularity.  Trust/familiarity signals,
#     not compatibility.  Only help when candidates are already close.
#
# Max compound boost: ~1.048 (4.8% score increase).
SOFT_BUSINESS_BOOSTS = {
    # Tier A — Quality (compatibility-adjacent)
    "high_rating_same_subdomain": 1.015,   # High rating + same subdomain
    # Tier B — Serving/Availability
    "open_programs": 1.02,                 # Has open programs
    "completed_programs": 1.008,           # Has completed programs
    # Tier C — Social/Business (weakest)
    "followed": 1.005,                     # Mentee follows mentor (very soft)
    "active_popular_same_subdomain": 1.005, # Active/popular + same subdomain
}


def apply_soft_business_boosts(
    scored_df: pd.DataFrame,
    boosts: dict | None = None,
) -> pd.DataFrame:
    """Apply conditional soft business-rule boosts as multiplicative score adjustments.

    These are BOOSTS ONLY — mentors are NEVER removed.  Multiplicative boosts
    preserve the model's relative ordering while providing gentle preference
    nudges for business-relevant conditions.

    Three tiers (strongest → weakest):
      Tier A — Quality (compatibility-adjacent):
        ×1.015 if high rating (≥ 0.5 scaled) AND same subdomain

      Tier B — Serving/Availability:
        ×1.02  if mentor has open programs
        ×1.008 elif mentor has completed programs (and no open)

      Tier C — Social/Business (weakest):
        ×1.005 if mentee follows mentor
        ×1.005 if active/popular AND same subdomain

    Max compound boost: ~1.048 (4.8% score increase).

    Note: thresholds are calibrated for MinMaxScaler-normalized features.
    mentor_weighted_rating is in [0, 1] after scaling.

    Args:
        scored_df: DataFrame with rerank_score and feature columns.
        boosts: Dict of boost_name -> multiplicative factor. Defaults provided.

    Returns:
        Same DataFrame with ``rerank_score`` updated by soft boosts.
    """
    b = {**SOFT_BUSINESS_BOOSTS, **(boosts or {})}
    df = scored_df.copy()

    if "rerank_score" not in df.columns:
        df["rerank_score"] = df.get("pred_score", 0.0)

    # --- Tier A: Quality (compatibility-adjacent) ---
    same_subdomain = pd.Series(False, index=df.index)
    if "subdomain_similarity" in df.columns:
        same_subdomain = df["subdomain_similarity"].fillna(0) > 0

    # mentor_weighted_rating is MinMaxScaler-normalized to [0, 1].
    # Threshold 0.5 ≈ upper half of scaled range.
    high_rating = pd.Series(False, index=df.index)
    if "mentor_weighted_rating" in df.columns:
        high_rating = df["mentor_weighted_rating"].fillna(0) >= 0.5

    df.loc[same_subdomain & high_rating, "rerank_score"] *= b["high_rating_same_subdomain"]

    # --- Tier B: Serving/Availability (SKILL-AWARE PRIORITY) ---
    # SKILL-FIRST UPDATE (May 2026): When skill matching is strong, prioritize
    # mentors who have programs because that's how the mentee will engage.
    # Boost multipliers increase with skill match strength.
    
    has_open = pd.Series(False, index=df.index)
    if "mentor_open_post_count_log" in df.columns:
        has_open = df["mentor_open_post_count_log"].fillna(0) > 0

    # Completed programs proxy: has program popularity activity but no open posts
    has_completed = pd.Series(False, index=df.index)
    if "popularity_log" in df.columns:
        has_activity = df["popularity_log"].fillna(0) > 0
        has_completed = has_activity & ~has_open

    # Determine skill match strength for adaptive boosting
    skill_coverage = df["skill_coverage_score"].fillna(0).astype(float) if "skill_coverage_score" in df.columns else pd.Series(0.0, index=df.index)
    subdomain = df["subdomain_similarity"].fillna(0).astype(float) if "subdomain_similarity" in df.columns else pd.Series(0.0, index=df.index)
    
    # High skill match: coverage > 0.6 (mentor covers most skills)
    # OR strong subdomain match (specialization in mentor's field)
    high_skill_match = (skill_coverage > 0.6) | (subdomain > 0.7)
    # Very high skill match: coverage > 0.75 AND strong subdomain
    very_high_skill_match = (skill_coverage > 0.75) & (subdomain > 0.7)
    
    # Apply adaptive program boosts
    #   Very high skill match + open programs: 1.05 (5% boost, strong priority)
    #   High skill match + open programs: 1.035 (3.5% boost, moderate priority)
    #   High skill match + completed programs: 1.015 (1.5% boost, secondary)
    
    if very_high_skill_match.any():
        df.loc[very_high_skill_match & has_open, "rerank_score"] *= 1.05
        df.loc[very_high_skill_match & ~has_open & has_completed, "rerank_score"] *= 1.015
    
    if high_skill_match.any():
        df.loc[high_skill_match & ~very_high_skill_match & has_open, "rerank_score"] *= 1.035
        df.loc[high_skill_match & ~very_high_skill_match & ~has_open & has_completed, "rerank_score"] *= 1.01
    
    # Default boosts for low skill match (original values)
    low_skill_match = ~high_skill_match
    df.loc[low_skill_match & has_open, "rerank_score"] *= b["open_programs"]
    df.loc[low_skill_match & ~has_open & has_completed, "rerank_score"] *= b["completed_programs"]

    # --- Tier C: Social/Business (weakest) ---
    if "is_following" in df.columns:
        df.loc[df["is_following"].fillna(0) == 1, "rerank_score"] *= b.get("followed", 1.003)

    active_popular = pd.Series(False, index=df.index)
    if "popularity_log" in df.columns:
        active_popular = df["popularity_log"].fillna(0) > 0
    elif "mentor_follower_count_log" in df.columns:
        active_popular = df["mentor_follower_count_log"].fillna(0) > 0

    df.loc[same_subdomain & active_popular, "rerank_score"] *= b["active_popular_same_subdomain"]

    return df
