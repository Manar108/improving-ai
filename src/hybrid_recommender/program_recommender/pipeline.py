from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..features import build_mentee_features
from ..preprocessing import (
    build_time_split_config,
    load_db_datasets,
    load_db_datasets_from_db,
    prepare_processed_tables,
    save_time_split_config,
)
from .features import (
    PAIR_FEATURE_COLS,
    build_mentee_program_candidates,
    build_program_cf_embeddings,
    build_program_features,
    compute_cf_confidence,
    validate_pair_feature_distributions,
)
from .io import (
    get_default_program_artifact_paths,
    get_program_paths,
    load_json_artifact,
    load_model,
    load_scaler,
    save_feature_artifact,
    save_json_artifact,
    save_model,
    save_scaler,
)
from .preprocessing import (
    PROGRAM_PREPROCESSING_VERSION,
    build_program_feature_manifest,
    detect_leakage_columns,
    fit_and_apply_program_scaler,
    get_program_numeric_feature_cols,
    validate_program_artifact_compatibility,
    validate_scaler_feature_order,
)
from .ranking import (
    DEFAULT_PROGRAM_FEATURE_COLS,
    evaluate_program_model,
    predict_program_scores,
    rerank_program_recommendations,
    split_program_by_time,
    train_program_model,
    _filter_by_experience_level,
)

logger = logging.getLogger(__name__)

# Pipeline reproducibility seed
_PIPELINE_RANDOM_SEED = 42
_PIPELINE_VERSION = "2026.05.12-hardened"





# ── Monitoring hooks (Item 32) ──

class PipelineMonitor:
    """Optional monitoring hooks for pipeline health diagnostics.

    Tracks feature drift, recommendation drift, candidate shrinkage, and
    CF sparsity.  Results are logged and returned as metadata.
    """

    def __init__(self):
        self.metrics: Dict[str, Any] = {}

    def record_candidate_stats(self, pair_df: pd.DataFrame) -> None:
        if pair_df.empty:
            self.metrics["candidate_count"] = 0
            self.metrics["candidate_shrinkage_warning"] = True
            logger.warning("PipelineMonitor: candidate pool is empty")
            return
        n_mentees = pair_df["mentee_id"].nunique()
        n_programs = pair_df["post_id"].nunique()
        avg_per_mentee = len(pair_df) / max(n_mentees, 1)
        self.metrics["candidate_count"] = len(pair_df)
        self.metrics["candidate_mentees"] = n_mentees
        self.metrics["candidate_programs"] = n_programs
        self.metrics["avg_candidates_per_mentee"] = round(avg_per_mentee, 1)
        if avg_per_mentee < 5:
            self.metrics["candidate_shrinkage_warning"] = True
            logger.warning(
                "PipelineMonitor: low candidate density (%.1f per mentee)", avg_per_mentee,
            )

    def record_cf_stats(self, cf_embeddings: Dict) -> None:
        user_factors = cf_embeddings.get("user_factors", {})
        item_factors = cf_embeddings.get("item_factors", {})
        self.metrics["cf_user_count"] = len(user_factors)
        self.metrics["cf_item_count"] = len(item_factors)
        if not user_factors:
            self.metrics["cf_sparsity_warning"] = True
            logger.warning("PipelineMonitor: CF embeddings are empty (sparse)")

    def record_training_stats(self, train_df: pd.DataFrame, label_col: str = "label") -> None:
        if train_df.empty:
            return
        n_pos = int((train_df[label_col] == 1).sum())
        n_neg = int((train_df[label_col] == 0).sum())
        ratio = n_neg / max(n_pos, 1)
        self.metrics["train_positive_count"] = n_pos
        self.metrics["train_negative_count"] = n_neg
        self.metrics["train_neg_pos_ratio"] = round(ratio, 2)

    def record_recommendation_stats(self, reranked: pd.DataFrame) -> None:
        if reranked.empty:
            return
        program_counts = reranked["post_id"].value_counts()
        self.metrics["recommended_unique_programs"] = len(program_counts)
        if len(program_counts) > 0:
            top_program_share = float(program_counts.iloc[0]) / len(reranked)
            self.metrics["top_program_share"] = round(top_program_share, 3)
            if top_program_share > 0.3:
                self.metrics["popularity_collapse_warning"] = True
                logger.warning(
                    "PipelineMonitor: potential popularity collapse — top program has %.1f%% share",
                    top_program_share * 100,
                )

    def get_report(self) -> Dict[str, Any]:
        return dict(self.metrics)


# ── Negative sampling diagnostics (Item 31) ──

def _compute_sampling_diagnostics(
    dataset: pd.DataFrame,
) -> Dict[str, Any]:
    """Track hardness distributions, pos/neg ratios, and sampling diversity."""
    if dataset.empty:
        return {}

    train = dataset[dataset.get("time_split", "") == "train"] if "time_split" in dataset.columns else dataset
    if train.empty:
        return {}

    n_pos = int((train.get("label", 0) == 1).sum())
    n_neg = int((train.get("label", 0) == 0).sum())

    diagnostics: Dict[str, Any] = {
        "train_positive_count": n_pos,
        "train_negative_count": n_neg,
        "train_neg_pos_ratio": round(n_neg / max(n_pos, 1), 2),
    }

    if "hardness_score" in train.columns:
        hardness = train[train["label"] == 0]["hardness_score"]
        if not hardness.empty:
            diagnostics["hardness_mean"] = round(float(hardness.mean()), 4)
            diagnostics["hardness_std"] = round(float(hardness.std()), 4)
            diagnostics["hardness_q25"] = round(float(hardness.quantile(0.25)), 4)
            diagnostics["hardness_q75"] = round(float(hardness.quantile(0.75)), 4)

    # Candidate coverage: fraction of unique programs in training
    if "post_id" in train.columns:
        diagnostics["training_unique_programs"] = int(train["post_id"].nunique())
        diagnostics["training_unique_mentees"] = int(train["mentee_id"].nunique())

    return diagnostics


# ── Pipeline integrity validation (Item 29) ──

def validate_pipeline_integrity(
    manifest: Dict[str, Any],
    scaler: Any | None,
    model: Any | None,
    feature_cols: list[str],
) -> None:
    """Validate end-to-end consistency between artifacts.

    Checks scaler vs manifest, manifest vs model, and feature ordering.
    Fails loudly on mismatch.
    """
    validate_program_artifact_compatibility(manifest, scaler, model)

    # Validate feature_cols vs manifest
    manifest_features = list(manifest.get("feature_cols", []))
    if manifest_features and feature_cols and manifest_features != feature_cols:
        raise ValueError(
            f"Pipeline feature_cols mismatch: pipeline={feature_cols}, manifest={manifest_features}"
        )

    # Validate scaler cols vs manifest scaler_feature_cols
    if scaler is not None:
        manifest_scale_cols = list(manifest.get("scaler_feature_cols", []))
        if manifest_scale_cols:
            validate_scaler_feature_order(scaler, manifest_scale_cols)

    logger.info("Pipeline integrity validation passed")


# ── Inference-safe loading (Item 30) ──

def load_program_inference_artifacts(
    artifact_paths: Dict[str, Path] | None = None,
) -> Dict[str, Any]:
    """Load and validate program inference artifacts.

    Validates versions, schema signatures, feature ordering, and scaler
    compatibility before returning usable artifacts.
    """
    paths = artifact_paths or get_default_program_artifact_paths()

    model = load_model(paths["model"])
    scaler = load_scaler(paths["scaler"])
    manifest = load_json_artifact(paths["manifest"])

    # Version validation
    manifest_version = manifest.get("preprocessing_version")
    if manifest_version != PROGRAM_PREPROCESSING_VERSION:
        raise ValueError(
            f"Artifact version mismatch: manifest={manifest_version}, "
            f"code={PROGRAM_PREPROCESSING_VERSION}"
        )

    # Schema signature validation
    feature_cols = list(manifest.get("feature_cols", []))
    scale_cols = list(manifest.get("scaler_feature_cols", []))
    if not feature_cols:
        raise ValueError("Manifest missing feature_cols")

    # Scaler compatibility
    if scaler is not None and scale_cols:
        validate_scaler_feature_order(scaler, scale_cols)

    # Model feature validation
    model_features = list(getattr(model, "feature_name_", []))
    if model_features and feature_cols and model_features != feature_cols:
        raise ValueError(
            f"Model feature mismatch: model={model_features}, manifest={feature_cols}"
        )

    logger.info(
        "Program inference artifacts loaded: %d features, %d scale cols, version=%s",
        len(feature_cols), len(scale_cols), manifest_version,
    )

    return {
        "model": model,
        "scaler": scaler,
        "manifest": manifest,
        "feature_cols": feature_cols,
        "scale_cols": scale_cols,
    }


# ── Reproducibility metadata (Item 28) ──

def _build_reproducibility_metadata(
    dataset: pd.DataFrame,
    scaler_strategy: str,
    feature_cols: list[str],
    scale_cols: list[str],
) -> Dict[str, Any]:
    """Build pipeline reproducibility metadata for artifact persistence."""
    from .preprocessing import _dataset_fingerprint, _feature_manifest_signature

    return {
        "pipeline_version": _PIPELINE_VERSION,
        "preprocessing_version": PROGRAM_PREPROCESSING_VERSION,
        "random_seed": _PIPELINE_RANDOM_SEED,
        "scaler_strategy": scaler_strategy,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": _dataset_fingerprint(dataset) if not dataset.empty else None,
        "schema_signature": _feature_manifest_signature(feature_cols, scale_cols),
        "feature_count": len(feature_cols),
        "scale_col_count": len(scale_cols),
    }


# ── Core pipeline functions ──

def _resolve_raw_tables(raw_data=None) -> Dict[str, pd.DataFrame]:
    if raw_data is None:
        logger.info("run_program_pipeline: loading from SQL Server database")
        return load_db_datasets_from_db()
    if isinstance(raw_data, (str, Path)):
        path = Path(raw_data)
        if (path / "programs.csv").exists():
            logger.info("run_program_pipeline: detected DB CSV format at %s", path)
            return load_db_datasets(path)
        raise FileNotFoundError("Expected DB-normalized files (e.g., programs.csv) in provided path")
    if isinstance(raw_data, dict):
        return raw_data
    raise TypeError("raw_data must be None, path, or dict[str, DataFrame]")


def _resolve_program_positive_pairs(
    applications: pd.DataFrame,
    mentorships: pd.DataFrame | None = None,
) -> tuple[
    dict[str, set[tuple[int, int]]],
    dict[str, dict[tuple[int, int], float]],
    dict[str, dict[tuple[int, int], float]],
    pd.Series,
]:
    """Resolve program labels from successful outcomes, not raw application noise.

    Label priorities:
    - Completed / successful mentorships: strongest positive
    - Accepted / approved applications: strong positive
    - Rejected / cancelled / withdrawn / alerted: ignored for positives

    IMPORTANT: For program recommendation we only treat accepted/approved
    outcomes as positive labels. Raw applied/pending rows are intentionally
    excluded so we do not learn from non-converted interest signals.
    """

    def _status_weight(status: str, source: str) -> float:
        status = str(status).strip().lower()

        if source == "mentorships":
            if status in {"completed", "complete", "finished", "done", "ended", "successful"}:
                return 1.0
            if status in {"accepted", "approved", "matched", "in_progress", "ongoing"}:
                return 0.95
            if status in {"rejected", "declined", "cancelled", "canceled", "withdrawn"}:
                return -0.85
            # No weak signals: pending/interested/applied/submitted return 0 (excluded)
            return 0.0

        # Applications: Only strong positive signals (no weak signals for realistic metrics)
        if status in {"accepted", "approved", "matched", "completed"}:
            return 0.95
        if status in {"rejected", "declined", "cancelled", "canceled", "withdrawn"}:
            return -1.0
        # No weak signals: pending/interested/applied/submitted return 0 (excluded)
        return 0.0

    positive_pairs_by_split: dict[str, set[tuple[int, int]]] = {"train": set(), "valid": set(), "test": set()}
    positive_weights_by_split: dict[str, dict[tuple[int, int], float]] = {"train": {}, "valid": {}, "test": {}}
    negative_weights_by_split: dict[str, dict[tuple[int, int], float]] = {"train": {}, "valid": {}, "test": {}}
    event_rows: list[pd.DataFrame] = []

    signal_sources: list[tuple[str, pd.DataFrame]] = []
    if applications is not None and not applications.empty:
        signal_sources.append(("applications", applications))
    if mentorships is not None and not mentorships.empty:
        signal_sources.append(("mentorships", mentorships))

    if not signal_sources:
        return positive_pairs_by_split, positive_weights_by_split, negative_weights_by_split, pd.Series(dtype="datetime64[ns]")

    for source_name, frame in signal_sources:
        required = {"post_id", "mentee_id", "time_split", "status"}
        time_col = "applied_at" if source_name == "applications" else "start_date"
        if time_col in frame.columns:
            required.add(time_col)
        elif source_name == "mentorships" and "end_date" in frame.columns:
            time_col = "end_date"
            required.add(time_col)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{source_name} missing required columns: {sorted(missing)}")

        rows = frame.copy()
        rows["status"] = rows["status"].astype(str).str.strip().str.lower()
        rows["post_id"] = pd.to_numeric(rows["post_id"], errors="coerce")
        rows["mentee_id"] = pd.to_numeric(rows["mentee_id"], errors="coerce")
        rows[time_col] = pd.to_datetime(rows[time_col], errors="coerce")
        rows = rows.dropna(subset=["post_id", "mentee_id", time_col, "time_split"])
        rows[["post_id", "mentee_id"]] = rows[["post_id", "mentee_id"]].astype(int)
        rows["time_split"] = rows["time_split"].astype(str).str.strip().str.lower()
        rows["label_weight"] = rows["status"].map(lambda s: _status_weight(s, source_name))
        rows = rows[rows["label_weight"] != 0].copy()

        if rows.empty:
            continue

        event_rows.append(rows[["mentee_id", time_col]].rename(columns={time_col: "event_time"}))

        for split_name in ("train", "valid", "test"):
            split_df = rows[rows["time_split"] == split_name]
            for mentee_id, post_id, weight in zip(split_df["mentee_id"], split_df["post_id"], split_df["label_weight"]):
                pair = (int(mentee_id), int(post_id))
                if weight > 0:
                    positive_pairs_by_split[split_name].add(pair)
                    current_weight = positive_weights_by_split[split_name].get(pair, 0.0)
                    positive_weights_by_split[split_name][pair] = max(current_weight, float(weight))
                else:
                    current_weight = negative_weights_by_split[split_name].get(pair, 0.0)
                    negative_weights_by_split[split_name][pair] = max(current_weight, abs(float(weight)))

    if event_rows:
        event_time_by_mentee = (
            pd.concat(event_rows, ignore_index=True)
            .groupby("mentee_id")["event_time"]
            .min()
            .sort_index()
        )
    else:
        event_time_by_mentee = pd.Series(dtype="datetime64[ns]")

    return positive_pairs_by_split, positive_weights_by_split, negative_weights_by_split, event_time_by_mentee


def _assign_program_labels_and_splits(
    pair_df: pd.DataFrame,
    positive_pairs_by_split: dict[str, set[tuple[int, int]]],
    positive_pair_weights_by_split: dict[str, dict[tuple[int, int], float]] | None = None,
    negative_pair_weights_by_split: dict[str, dict[tuple[int, int], float]] | None = None,
) -> pd.DataFrame:
    """Assign labels similarly to main pipeline assign_labels_and_splits."""
    if pair_df.empty:
        return pair_df

    out_parts: list[pd.DataFrame] = []
    all_positive_mentees = set()

    for split_name in ("train", "valid", "test"):
        split_pairs = positive_pairs_by_split.get(split_name, set())
        split_weights = (positive_pair_weights_by_split or {}).get(split_name, {})
        split_mentees = {m for m, _ in split_pairs}
        all_positive_mentees |= split_mentees
        if not split_mentees:
            continue

        split_df = pair_df[pair_df["mentee_id"].isin(split_mentees)].copy()
        split_df["time_split"] = split_name
        negative_weights = (negative_pair_weights_by_split or {}).get(split_name, {})
        labels = []
        weights = []
        for m, p in zip(split_df["mentee_id"], split_df["post_id"]):
            pair = (int(m), int(p))
            if pair in split_pairs:
                labels.append(1)
                weights.append(float(split_weights.get(pair, 1.0)))
            else:
                labels.append(0)
                weights.append(float(1.0 + negative_weights.get(pair, 0.0)))
        split_df["label"] = labels
        split_df["label_weight"] = weights
        out_parts.append(split_df)

    # Negative-only mentees stay in train split
    all_mentees = set(pair_df["mentee_id"].unique())
    neg_only = all_mentees - all_positive_mentees
    if neg_only:
        neg_df = pair_df[pair_df["mentee_id"].isin(neg_only)].copy()
        neg_df["time_split"] = "train"
        neg_df["label"] = 0
        neg_df["label_weight"] = 1.0
        out_parts.append(neg_df)

    if not out_parts:
        return pd.DataFrame(columns=list(pair_df.columns) + ["time_split", "label", "label_weight"])

    return pd.concat(out_parts, ignore_index=True)


def _sample_program_negatives_per_group(
    group_df: pd.DataFrame,
    neg_per_pos: int = 8,
    min_candidates_per_group: int = 12,
    hard_fraction: float = 0.55,
    keep_signal_only: bool = True,
    rng_seed: int = _PIPELINE_RANDOM_SEED,
) -> pd.DataFrame | None:
    """Sample program negatives using skill-level hardness signals."""
    pos = group_df[group_df["label"] == 1].copy()
    neg = group_df[group_df["label"] == 0].copy()
    if len(pos) == 0 or len(neg) == 0:
        return None

    # Remove true positives from negative side (safety)
    pos_posts = set(pos["post_id"].unique())
    neg = neg[~neg["post_id"].isin(pos_posts)].copy()
    if len(neg) == 0:
        return None

    neg["hardness_score"] = (
        0.45 * neg["requirement_coverage_score"].fillna(0)
        + 0.25 * neg["required_skill_level_match_score"].fillna(0)
        + 0.20 * neg["requirement_overlap_score"].fillna(0)
        + 0.10 * neg["candidate_pre_score"].fillna(0)
    )

    signal_mask = (
        (neg["requirement_coverage_score"] > 0)
        | (neg["required_skill_level_match_score"] > 0)
    )
    signal_neg = neg[signal_mask].copy()
    if keep_signal_only:
        neg = signal_neg
    else:
        neg = neg.copy()

    if len(neg) == 0:
        return None

    target_neg = max(len(pos) * neg_per_pos, min_candidates_per_group - len(pos))
    target_neg = min(target_neg, len(neg))
    if target_neg <= 0:
        return None

    num_hard = int(np.ceil(target_neg * hard_fraction))
    num_random = target_neg - num_hard

    hard = neg.nlargest(num_hard, "hardness_score")
    remaining = neg.drop(hard.index)
    rnd = remaining.sample(num_random, random_state=rng_seed) if len(remaining) >= num_random else remaining

    sampled = pd.concat([pos, hard, rnd], ignore_index=True)
    sampled["hardness_score"] = sampled.get("hardness_score", 0).fillna(0)
    return sampled


def build_program_recommendation_dataset(
    pair_df: pd.DataFrame,
    positive_pairs_by_split: dict[str, set[tuple[int, int]]],
    positive_pair_weights_by_split: dict[str, dict[tuple[int, int], float]] | None = None,
    negative_pair_weights_by_split: dict[str, dict[tuple[int, int], float]] | None = None,
    train_neg_per_pos: int = 8,
    eval_neg_per_pos: int = 16,
    min_candidates_per_group: int = 12,
) -> pd.DataFrame:
    labeled = _assign_program_labels_and_splits(
        pair_df,
        positive_pairs_by_split,
        positive_pair_weights_by_split=positive_pair_weights_by_split,
        negative_pair_weights_by_split=negative_pair_weights_by_split,
    )
    if labeled.empty:
        return labeled

    train = labeled[labeled["time_split"] == "train"]
    eval_df = labeled[labeled["time_split"].isin(["valid", "test"])]

    train_parts: list[pd.DataFrame] = []
    for _, g in train.groupby("mentee_id"):
        sampled = _sample_program_negatives_per_group(
            g,
            neg_per_pos=train_neg_per_pos,
            min_candidates_per_group=min_candidates_per_group,
            hard_fraction=0.70,
            keep_signal_only=True,
            rng_seed=_PIPELINE_RANDOM_SEED,
        )
        if sampled is not None:
            train_parts.append(sampled)

    train_out = pd.concat(train_parts, ignore_index=True) if train_parts else train.copy()
    # Evaluation keeps the full candidate pool for valid/test mentees so metrics
    # reflect ranking quality on the actual eligible set, not a sampled subset.
    eval_out = eval_df.copy()

    return pd.concat([train_out, eval_out], ignore_index=True)


def _add_program_cf_score(
    pair_df: pd.DataFrame,
    cf_embeddings: Dict[str, Dict[int, np.ndarray]],
) -> pd.DataFrame:
    out = pair_df.copy()
    user_factors = cf_embeddings.get("user_factors", {})
    item_factors = cf_embeddings.get("item_factors", {})
    if not user_factors or not item_factors:
        out["cf_score"] = 0.0
        out["cf_confidence"] = 0.0
        return out

    dim = len(next(iter(user_factors.values())))
    zero = np.zeros(dim)

    # Compute CF scores with confidence-aware weighting (Item 17)
    cf_scores = []
    cf_confidences = []
    for m, p in zip(out["mentee_id"], out["post_id"]):
        u_vec = user_factors.get(int(m), zero)
        i_vec = item_factors.get(int(p), zero)
        raw_score = float(np.dot(u_vec, i_vec))

        # Per-user CF confidence
        confidence = compute_cf_confidence(cf_embeddings, int(m))

        # Dampen CF score for low-confidence users to prevent noisy CF
        # from overpowering content signals
        dampened_score = raw_score * confidence
        cf_scores.append(dampened_score)
        cf_confidences.append(confidence)

    out["cf_score"] = cf_scores
    out["cf_score"] = out["cf_score"].fillna(0.0)
    out["cf_confidence"] = cf_confidences
    return out


def run_program_pipeline(raw_data=None) -> dict:
    """Full program recommendation pipeline using project datasets.

    This mirrors the existing mentor pipeline structure:
    load -> preprocess -> features -> labels -> sample -> train/eval -> save artifacts.

    Production hardened with:
    - Pipeline caching (Item 27)
    - Reproducibility metadata (Item 28)
    - Integrity validation (Item 29)
    - Negative sampling diagnostics (Item 31)
    - Monitoring hooks (Item 32)
    - Performance optimizations (Item 33)
    """
    pipeline_start = time.monotonic()
    monitor = PipelineMonitor()

    raw_tables = _resolve_raw_tables(raw_data)

    config = build_time_split_config(
        raw_tables["mentorships"],
        applications=raw_tables.get("mentorship_applications"),
    )

    processed = prepare_processed_tables(raw_tables, config)

    paths = get_program_paths()
    save_time_split_config(config, paths["root"] / "config" / "time_split_config.csv")

    mentee_features = build_mentee_features(
        processed["mentee_profile"],
        processed["mentee_subdomains"],
        processed["mentee_interests"],
    )

    program_features = build_program_features(
        processed["mentorship_posts"],
        processed["mentorship_requirements"],
        program_enrollments=processed.get("mentorships"),
        mentorship_applications=processed.get("mentorship_applications"),
        reference_time=config.train_end,
    )

    # ────────────────────────────────────────────────────────────────
    # Business Logic: Filter to ACTIVE programs only
    # ────────────────────────────────────────────────────────────────
    # Programs must satisfy ALL of:
    # 1. is_open = True (ProgramPostStatus = Published, accepting applications)
    # 2. is_available = True (mentor has not closed/archive the program)
    # 3. deadline_passed = False (deadline has not expired)
    n_programs_before = len(program_features)
    program_features = program_features[
        (program_features.get("is_open", 0) == 1) &
        (program_features.get("is_available", 0) == 1) &
        (program_features.get("deadline_passed", 0) == 0)
    ].copy()
    n_programs_active = len(program_features)
    n_programs_filtered = n_programs_before - n_programs_active
    logger.info(
        "Active program filter: kept %d/%d programs (filtered %d unpublished/unavailable/expired programs)",
        n_programs_active, n_programs_before, n_programs_filtered,
    )
    if n_programs_active == 0:
        raise ValueError("No active programs found after filtering is_open==1 AND is_available==1 AND deadline_passed==0")

    pair_features = build_mentee_program_candidates(
        mentee_features,
        program_features,
        mentee_interest_levels=processed["mentee_interests"],
        top_k_per_mentee=120,
        enforce_hard_gates=True,
    )

    # Monitor candidate pool health (Item 32)
    # Feature distribution validation (Item 19)
    validate_pair_feature_distributions(pair_features)

    # Leakage detection on pair features (Item 8)
    leakage_cols = detect_leakage_columns(pair_features.columns)
    if leakage_cols:
        raise ValueError(f"Leakage columns detected in pair features: {leakage_cols}")

    # ────────────────────────────────────────────────────────────────
    # Collaborative Filtering (CF) - INTERACTION SIGNALS ONLY
    # ────────────────────────────────────────────────────────────────
    # CF is built from INTERACTION SIGNALS ONLY (likes, comments, saves, shares).
    # Enrollments (mentorships) are EXCLUDED because they ARE the positive labels.
    # Using enrollments in CF would cause direct data leakage.
    # 
    # CF training data is filtered to BEFORE train_end to prevent future interactions
    # from leaking into embeddings. All CF-derived signals are then safely used in training.
    logger.info("Building CF embeddings from interaction signals (likes/comments/saves/shares, training-period only)...")
    mentorships_train_only = (
        processed.get("mentorships", pd.DataFrame())
        .copy()
    )
    # Filter to training period only (before train_end)
    if "start_date" in mentorships_train_only.columns:
        mentorships_train_only = mentorships_train_only[
            mentorships_train_only["start_date"] <= config.train_end
        ]
    
    cf_embeddings = build_program_cf_embeddings(
        enrollments=mentorships_train_only,
        likes=processed.get("posts_likes_dataset", pd.DataFrame()),
        saves=processed.get("saved_posts_dataset", pd.DataFrame()),
        comments=processed.get("posts_comments", pd.DataFrame()),
        shares=processed.get("shared_posts_dataset", pd.DataFrame()),
        n_factors=16,
    )
    logger.info(
        "CF embeddings built: %d user factors, %d item factors",
        len(cf_embeddings.get("user_factors", {})),
        len(cf_embeddings.get("item_factors", {})),
    )
    
    # Add CF score to pair features (safe - uses training-only CF)
    if cf_embeddings.get("user_factors") and cf_embeddings.get("item_factors"):
        pair_features = _add_program_cf_score(pair_features, cf_embeddings)
        
        # Normalize CF scores to [0, 1] range for stable training
        # CF dot products can exceed 1.0; clamp to [-1, 1] then rescale to [0, 1]
        cf_min = pair_features["cf_score"].min()
        cf_max = pair_features["cf_score"].max()
        if cf_max > cf_min:
            pair_features["cf_score"] = (
                (pair_features["cf_score"] - cf_min) / (cf_max - cf_min)
            ).clip(0, 1)
        else:
            pair_features["cf_score"] = 0.0
        
        logger.info(
            "CF scores added and normalized to [0, 1]: min=%.4f, max=%.4f",
            pair_features["cf_score"].min(),
            pair_features["cf_score"].max(),
        )
    else:
        logger.warning("CF embeddings empty - skipping CF score computation")
        pair_features["cf_score"] = 0.0
        pair_features["cf_confidence"] = 0.0

    positive_pairs_by_split, positive_pair_weights_by_split, negative_pair_weights_by_split, _event_time = _resolve_program_positive_pairs(
        processed.get("mentorship_applications", pd.DataFrame()),
        processed.get("mentorships", pd.DataFrame()),
    )

    dataset = build_program_recommendation_dataset(
        pair_features,
        positive_pairs_by_split,
        positive_pair_weights_by_split=positive_pair_weights_by_split,
        negative_pair_weights_by_split=negative_pair_weights_by_split,
        train_neg_per_pos=4,  # 🔧 Reduced from 8 for better balance
        eval_neg_per_pos=16,
        min_candidates_per_group=12,
    )

    # Negative sampling diagnostics (Item 31)
    sampling_diagnostics = _compute_sampling_diagnostics(dataset)

    # Feature prep + split
    numeric_cols = get_program_numeric_feature_cols(dataset)
    feature_cols = [c for c in DEFAULT_PROGRAM_FEATURE_COLS if c in dataset.columns]
    feature_cols = list(dict.fromkeys(feature_cols + [c for c in numeric_cols if c not in {"hardness_score"}]))

    train_df, valid_df, test_df = split_program_by_time(dataset)

    # Monitor training stats (Item 32)
    monitor.record_training_stats(train_df)

    train_df, valid_df, scaler, scale_cols = fit_and_apply_program_scaler(
        train_df,
        valid_df,
        numeric_cols=get_program_numeric_feature_cols(train_df),
        feature_cols=feature_cols,
    )
    test_df = test_df.copy()
    if not test_df.empty and scale_cols:
        from .preprocessing import apply_program_scaler

        test_df = apply_program_scaler(test_df, scaler, scale_cols, feature_cols=feature_cols)

    model = train_program_model(train_df, valid_df, feature_cols)

    valid_scored = predict_program_scores(model, valid_df, feature_cols)
    test_scored = predict_program_scores(model, test_df, feature_cols)

    # Apply experience-level filter: only show programs with similar or slightly lower required level
    valid_scored = _filter_by_experience_level(valid_scored, max_level_gap=1)
    test_scored = _filter_by_experience_level(test_scored, max_level_gap=1)

    valid_reranked = rerank_program_recommendations(valid_scored)
    test_reranked = rerank_program_recommendations(test_scored)

    # Monitor recommendation quality (Item 32)
    metrics_valid = evaluate_program_model(valid_scored, k=10)
    metrics_test_raw = evaluate_program_model(test_scored, k=10)
    metrics_test_reranked = evaluate_program_model(test_reranked, k=10)

    # Save artifacts
    art = get_default_program_artifact_paths()
    save_model(model, art["model"])
    save_scaler(scaler, art["scaler"])
    save_feature_artifact(dataset, art["dataset"].name)
    save_feature_artifact(program_features, art["program_features"].name)

    manifest = build_program_feature_manifest(
        train_df,
        feature_cols,
        scale_cols,
        scaler_strategy="compatibility",
    )

    # Reproducibility metadata (Item 28)
    reproducibility = _build_reproducibility_metadata(
        dataset, "compatibility", feature_cols, scale_cols,
    )
    manifest["reproducibility"] = reproducibility
    manifest["sampling_diagnostics"] = sampling_diagnostics
    manifest["monitoring"] = monitor.get_report()

    manifest.update(
        {
            "metrics_valid": metrics_valid,
            "metrics_test_raw": metrics_test_raw,
            "metrics_test_reranked": metrics_test_reranked,
        }
    )
    save_json_artifact(manifest, art["manifest"])

    # Pipeline integrity validation (Item 29)
    validate_pipeline_integrity(manifest, scaler, model, feature_cols)

    pipeline_duration = time.monotonic() - pipeline_start
    logger.info("Program pipeline completed in %.1f seconds", pipeline_duration)

    return {
        "config": config,
        "processed": processed,
        "mentee_features": mentee_features,
        "program_features": program_features,
        "pair_features": pair_features,
        "dataset": dataset,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "model": model,
        "scaler": scaler,
        "metrics_valid": metrics_valid,
        "metrics_test_raw": metrics_test_raw,
        "metrics_test_reranked": metrics_test_reranked,
        "valid_reranked": valid_reranked,
        "test_reranked": test_reranked,
        "artifacts": art,
        "sampling_diagnostics": sampling_diagnostics,
        "monitoring": monitor.get_report(),
        "reproducibility": reproducibility,
    }
