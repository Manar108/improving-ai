"""Program recommender preprocessing — production-hardened.

Version: 2026.05.12-hardened
Architecture:
    features.PAIR_FEATURE_COLS → dynamic schema → validation → scaling → manifest

Key design decisions:
    - Feature lists are dynamically generated from PAIR_FEATURE_COLS + optional groups
      rather than maintained as static duplicates (prevents silent schema drift).
    - Dataset fingerprinting uses schema + row-count + column-stats hash (not row data)
      for deterministic, fast artifact staleness detection.
    - All diagnostics use structured logging (no warnings.warn) for production
      log aggregation compatibility.
    - Training mode enforces strict column validation (allow_extra_columns=False)
      to prevent silent leakage from stale or unexpected columns.
    - Scaler assignment uses .values to prevent index misalignment on filtered frames.
"""
from __future__ import annotations

import logging
import hashlib
import json
import re as _re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from .features import PAIR_FEATURE_COLS
from .io import save_json_artifact


logger = logging.getLogger(__name__)

PROGRAM_PREPROCESSING_VERSION = "2026.05.12"
PROGRAM_MANIFEST_ARTIFACT_VERSION = 2

# ── Diagnostic thresholds (extracted for tunability) ──
_NAN_SPIKE_RELATIVE_THRESHOLD = 2.0
_NAN_SPIKE_ABSOLUTE_THRESHOLD = 0.25
_NAN_SPIKE_FLOOR = 0.10
_MEDIAN_DRIFT_THRESHOLD = 2.0
_OUTLIER_IQR_MULTIPLIER = 6
PROGRAM_OPTIONAL_ALIGNMENT_COLUMNS = {
    # "cf_score" now re-enabled with training-only CF (safe)
    "program_popularity_log",
    "program_difficulty_score",
}
PROGRAM_FEATURE_GROUPS = {
    "binary": set(),
    "eligibility": {
        "target_level_pass",
        "education_level_pass",
        # "availability_pass" removed (no variance)
        "eligibility_pass",
        "target_level_exact_match",
        "education_level_exact_match",
        "minimum_requirement_exact_match",
        "minimum_requirement_above_minimum",
    },
    # NEW (May 2026): Soft compatibility scoring
    "soft_compatibility": {
        "target_level_distance",
        "target_level_softness",
        "education_distance",
        "education_softness",
        "skill_level_compatibility",
        "overall_eligibility_softness",
    },
    "confidence_band": {
        "compatibility_confidence_band",  # String-valued: exact_fit|near_fit|stretch_fit|weak_fit
    },
    "reranking_only": {
        "compatibility_confidence_band",
    },
    "cf": set(),  # Removed: cf_score causes data leakage
    "popularity": {
        "program_popularity_log",
        "program_difficulty_score",
        "hardness_score",
    },
    "ml_only": set(),
    "business_rule": {
        "target_level_pass",
        "education_level_pass",
        # "availability_pass" removed (no variance)
        "eligibility_pass",
        "target_level_exact_match",
        "education_level_exact_match",
        "minimum_requirement_exact_match",
        "minimum_requirement_above_minimum",
    },
}
PROGRAM_RATIO_FEATURES = {
    "requirement_coverage_score",
    "requirement_overlap_score",
    "required_skill_level_match_score",
    "candidate_pre_score",
    "cf_score",                            # CF score (normalized similarity)
    "cf_confidence",                       # CF confidence (0-1)
    # NEW (May 2026): Soft compatibility scores are also ratios [0,1]
    "target_level_softness",
    "education_softness",
    "skill_level_compatibility",
    "overall_eligibility_softness",
}
PROGRAM_COUNT_FEATURES = {
    "matched_required_skill_count",
    "missing_required_skill_count",
    "program_enrollment_count",
    "spots_left",
    "capacity",
    # NEW (May 2026): Level distances are count-like integers
    "target_level_distance",
    "education_distance",
    # NEW (May 2026): Deadline features
    "days_until_deadline",
}
PROGRAM_POPULARITY_FEATURES = {
    "program_popularity_log",
    "program_difficulty_score",
}
# CF interactions from training-only mentorships (SAFE - no future leakage)
PROGRAM_INTERACTION_FEATURES = {"cf_score"}  # Re-enabled with training-only data

PROGRAM_BINARY_COLS = {
    "is_open",
    "target_level_pass",
    "education_level_pass",
    "target_level_exact_match",
    "education_level_exact_match",
    "minimum_requirement_exact_match",
    "minimum_requirement_above_minimum",
    # "availability_pass" removed (no variance)
    "eligibility_pass",
    # NEW (May 2026): Deadline status
    "deadline_passed",
}

PROGRAM_ID_COLS = {
    "mentee_id",
    "post_id",
}

PROGRAM_LABEL_COLS = {
    "label",
    "time_split",
}

PROGRAM_LEAKAGE_BLACKLIST = {
    "future_clicks",
    "future_applications",
    "accepted_after_split",
    "future_popularity",
    "future_interactions",
    "post_split_statistics",
    "post_split_stats",
    "evaluation_only_column",
}

# Regex patterns for detecting potential leakage columns at runtime.
# Columns matching any pattern are blocked from training, scaling, and reranking.
_LEAKAGE_REGEX_PATTERNS = [
    _re.compile(r"^future_"),
    _re.compile(r"^post_split_"),
    _re.compile(r"^after_split"),
    _re.compile(r"^eval_(?!uation)"),
    _re.compile(r"leakage"),
    _re.compile(r"^label_future"),
    _re.compile(r"_after_split$"),
]


def detect_leakage_columns(columns: Iterable[str]) -> list[str]:
    """Detect potential leakage columns using static blacklist + regex heuristics.

    Returns column names that match either the static PROGRAM_LEAKAGE_BLACKLIST
    or any of the _LEAKAGE_REGEX_PATTERNS.  Used by schema validation and
    scaler fitting to prevent accidental leakage into training/scaling/reranking.
    """
    detected = []
    for col in columns:
        lowered = col.lower()
        if col in PROGRAM_LEAKAGE_BLACKLIST:
            detected.append(col)
            continue
        for pattern in _LEAKAGE_REGEX_PATTERNS:
            if pattern.search(lowered):
                detected.append(col)
                break
    return detected

PROGRAM_INTERNAL_EXCLUDED_COLS = {
    "hardness_score",
    "compatibility_confidence_band",
}

# PROGRAM_OPTIONAL_EXTRA_COLS removed (cf_score causes data leakage)

# ── Dynamic schema generation ──
# These functions derive expected/required features from PAIR_FEATURE_COLS
# at call time, so any future changes to features.py propagate automatically.

def build_required_features() -> tuple[str, ...]:
    """Dynamically build required feature tuple from current PAIR_FEATURE_COLS."""
    return tuple(dict.fromkeys([*PROGRAM_ID_COLS, *PAIR_FEATURE_COLS, *PROGRAM_LABEL_COLS]))


def build_expected_features() -> tuple[str, ...]:
    """Dynamically build expected feature tuple (required + optional + internal)."""
    required = build_required_features()
    # PROGRAM_OPTIONAL_EXTRA_COLS removed (cf_score causes data leakage)
    return tuple(dict.fromkeys([*required, *PROGRAM_INTERNAL_EXCLUDED_COLS]))


def build_feature_dtypes() -> Dict[str, str]:
    """Dynamically build feature dtype map from current PAIR_FEATURE_COLS."""
    dtypes: Dict[str, str] = {
        "mentee_id": "numeric",
        "post_id": "numeric",
        "label": "numeric",
        "time_split": "string",
        "hardness_score": "numeric",
        "cf_score": "numeric",  # Re-enabled: built from training-only mentorships (safe)
        "compatibility_confidence_band": "string",
    }
    for col in PAIR_FEATURE_COLS:
        if col not in dtypes:
            dtypes[col] = "numeric"
    return dtypes


# Legacy aliases — kept for backward compatibility with existing callers.
# These are evaluated once at import time; callers needing live schema
# should call build_required_features() / build_expected_features() directly.
REQUIRED_FEATURES = build_required_features()
EXPECTED_PROGRAM_FEATURES = build_expected_features()
FEATURE_DTYPES = build_feature_dtypes()

# Boolean normalization — ONLY pure boolean semantics.
# Enum states (publication, availability, activity) are handled separately
# to prevent semantic corruption (e.g. "published" → 1, "draft" → 0).
BOOL_TRUE_STRINGS = {
    "1",
    "t",
    "true",
    "yes",
    "y",
    "on",
}
BOOL_FALSE_STRINGS = {
    "0",
    "f",
    "false",
    "no",
    "n",
    "off",
}

# Enum state mappings — separate business concepts, NOT boolean values.
# Publication state: ProgramPostStatus ("draft" vs "published")
# Availability state: Availability ("open"/"opened" vs "closed")
# Activity state: general active/inactive semantics
PUBLICATION_STATE_MAP: Dict[str, bool] = {
    "published": True,
    "draft": False,
    "archived": False,
}
AVAILABILITY_STATE_MAP: Dict[str, bool] = {
    "open": True,
    "opened": True,
    "available": True,
    "closed": False,
    "inactive": False,
    "unavailable": False,
}
ACTIVITY_STATE_MAP: Dict[str, bool] = {
    "active": True,
    "inactive": False,
}

COUNT_LIKE_SUFFIXES = (
    "_count",
    "_counts",
)

GROUPED_STRATEGY = "grouped"
LEGACY_STRATEGY = "compatibility"
STRICT_ALIGNMENT = "strict"
SOFT_ALIGNMENT = "soft"


@dataclass
class ProgramFeatureScaler:
    """A lightweight scaler wrapper for grouped preprocessing strategies.

    Persists exact column ordering, strategy metadata, and group assignment
    to prevent silent scaling corruption from column reordering or drift.
    """

    strategy: str
    bounded_cols: list[str]
    robust_cols: list[str]
    bounded_scaler: MinMaxScaler | None = None
    robust_scaler: RobustScaler | None = None
    _fitted_column_order: list[str] | None = None
    _fit_timestamp: str | None = None

    def fit(self, df: pd.DataFrame) -> ProgramFeatureScaler:
        self._fitted_column_order = list(self.bounded_cols) + list(self.robust_cols)
        self._fit_timestamp = datetime.now(timezone.utc).isoformat()
        if self.bounded_cols:
            self.bounded_scaler = MinMaxScaler()
            self.bounded_scaler.fit(df[self.bounded_cols].astype(float))
        if self.robust_cols:
            self.robust_scaler = RobustScaler()
            self.robust_scaler.fit(df[self.robust_cols].astype(float))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.bounded_cols and self.bounded_scaler is not None:
            scaled = self.bounded_scaler.transform(out[self.bounded_cols].astype(float))
            out[self.bounded_cols] = pd.DataFrame(
                scaled, columns=self.bounded_cols, index=out.index,
            ).values
        if self.robust_cols and self.robust_scaler is not None:
            scaled = self.robust_scaler.transform(out[self.robust_cols].astype(float))
            out[self.robust_cols] = pd.DataFrame(
                scaled, columns=self.robust_cols, index=out.index,
            ).values
        return out

    @property
    def feature_names_in_(self) -> np.ndarray:
        if self._fitted_column_order is not None:
            return np.asarray(self._fitted_column_order, dtype=object)
        return np.asarray([*self.bounded_cols, *self.robust_cols], dtype=object)

    def get_scaler_metadata(self) -> Dict[str, Any]:
        """Return serializable metadata for artifact persistence."""
        return {
            "strategy": self.strategy,
            "bounded_cols": list(self.bounded_cols),
            "robust_cols": list(self.robust_cols),
            "fitted_column_order": list(self._fitted_column_order or []),
            "fit_timestamp": self._fit_timestamp,
        }

    def validate_column_order(self, expected_cols: Sequence[str]) -> None:
        """Validate that expected columns match the fitted column order."""
        if self._fitted_column_order is None:
            return
        expected = list(expected_cols)
        if self._fitted_column_order != expected:
            raise ValueError(
                f"Scaler column order mismatch: fitted={self._fitted_column_order}, "
                f"expected={expected}"
            )


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_fingerprint(df: pd.DataFrame, *, full_checksum: bool = False) -> str:
    """Compute a robust, deterministic fingerprint from schema + statistics.

    Uses schema shape, dtypes, quantiles, and per-column statistics rather than
    raw row data so the fingerprint is stable across row ordering and fast on
    large frames.  Detects stale datasets, reordered data, partial corruption,
    and distribution-preserving drift.

    Args:
        full_checksum: If True, include deterministic sampled row hashes for
            stronger corruption detection (slower but more thorough).
    """
    col_stats = {}
    for col in df.columns:
        series = df[col]
        stats: Dict[str, Any] = {"dtype": str(series.dtype), "null_count": int(series.isna().sum())}
        if pd.api.types.is_numeric_dtype(series):
            finite = series.replace([np.inf, -np.inf], np.nan).dropna()
            if not finite.empty:
                stats["min"] = float(finite.min())
                stats["max"] = float(finite.max())
                stats["mean"] = round(float(finite.mean()), 6)
                stats["std"] = round(float(finite.std()), 6) if len(finite) > 1 else 0.0
                # Quantiles for distribution drift detection
                for q in (0.1, 0.25, 0.5, 0.75, 0.9):
                    stats[f"q{int(q * 100)}"] = round(float(finite.quantile(q)), 6)
        col_stats[col] = stats

    # Schema checksum (column names + dtypes) — detects reordered/renamed columns
    schema_sig = _sha256_payload({
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    })

    summary_payload = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "col_stats": col_stats,
        "schema_checksum": schema_sig,
    }

    # Deterministic sampled row hashes for stronger corruption detection
    if full_checksum and len(df) > 0:
        sample_size = min(200, len(df))
        rng = np.random.RandomState(42)
        sample_indices = sorted(rng.choice(len(df), size=sample_size, replace=False))
        sample_rows = df.iloc[sample_indices].reset_index(drop=True)
        row_hash = _sha256_payload(sample_rows.to_dict(orient="list"))
        summary_payload["sampled_row_hash"] = row_hash

    return _sha256_payload(summary_payload)


def _manifest_schema_signature(feature_cols: Sequence[str], scale_cols: Sequence[str]) -> str:
    return _sha256_payload({"feature_cols": list(feature_cols), "scale_cols": list(scale_cols)})


def _is_duplicate_columns(df: pd.DataFrame) -> list[str]:
    return df.columns[df.columns.duplicated()].tolist()


def _normalize_string_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _normalize_enum_label(series: pd.Series) -> pd.Series:
    normalized = _normalize_string_series(series)
    normalized = normalized.replace({"nan": "", "none": "", "null": ""})
    return normalized


def _coerce_bool_like_value(value: Any) -> float | int | np.floating | np.integer | None:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) == 1:
            return 1
        if int(value) == 0:
            return 0
        return np.nan
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        if value == 1.0:
            return 1
        if value == 0.0:
            return 0
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan
    if text in BOOL_TRUE_STRINGS:
        return 1
    if text in BOOL_FALSE_STRINGS:
        return 0
    return np.nan


def _coerce_binary_series(series: pd.Series, column: str) -> pd.Series:
    normalized = series.map(_coerce_bool_like_value)
    invalid_mask = normalized.isna() & series.notna()
    if invalid_mask.any():
        logger.warning(
            "Program binary column '%s' contained unexpected values; coerced to 0 for %d rows.",
            column, int(invalid_mask.sum()),
        )
    return normalized.fillna(0).astype(int)


def _normalize_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in PROGRAM_BINARY_COLS.intersection(out.columns):
        out[column] = _coerce_binary_series(out[column], column)
    return out


def _normalize_enum_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    enum_columns = {
        "availability",
        "status",
        "target_level",
        "education_level",
        "experience_level",
    }
    for column in enum_columns.intersection(out.columns):
        out[column] = _normalize_enum_label(out[column])
    return out


def _normalize_program_enum_columns(df: pd.DataFrame) -> pd.DataFrame:
    return _normalize_enum_columns(df)


def _numeric_like_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column in FEATURE_DTYPES and FEATURE_DTYPES[column] == "numeric"
    ]


def get_program_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric model columns for the program pipeline.

    Excludes ids, labels, leakage columns, and binary columns.
    """
    numeric_cols = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    excluded = PROGRAM_BINARY_COLS | PROGRAM_ID_COLS | PROGRAM_LABEL_COLS | PROGRAM_LEAKAGE_BLACKLIST | PROGRAM_INTERNAL_EXCLUDED_COLS
    return [column for column in numeric_cols if column not in excluded]


def infer_program_feature_groups(feature_cols: Sequence[str]) -> Dict[str, list[str]]:
    feature_list = list(feature_cols)
    return {
        "binary": [column for column in feature_list if column in PROGRAM_BINARY_COLS],
        "eligibility": [column for column in feature_list if column in PROGRAM_FEATURE_GROUPS["eligibility"]],
        "cf": [column for column in feature_list if column in PROGRAM_INTERACTION_FEATURES],
        "popularity": [column for column in feature_list if column in PROGRAM_POPULARITY_FEATURES],
        "business_rule": [column for column in feature_list if column in PROGRAM_FEATURE_GROUPS["business_rule"]],
        "reranking_only": [column for column in feature_list if column in PROGRAM_FEATURE_GROUPS["reranking_only"]],
        "ml_only": [column for column in feature_list if column not in PROGRAM_BINARY_COLS and column not in PROGRAM_FEATURE_GROUPS["business_rule"] and column not in PROGRAM_INTERACTION_FEATURES and column not in PROGRAM_POPULARITY_FEATURES and column not in PROGRAM_FEATURE_GROUPS["reranking_only"]],
    }


def _split_optional_and_required_feature_cols(feature_cols: Sequence[str]) -> tuple[list[str], list[str]]:
    required = [column for column in feature_cols if column not in PROGRAM_OPTIONAL_ALIGNMENT_COLUMNS]
    optional = [column for column in feature_cols if column in PROGRAM_OPTIONAL_ALIGNMENT_COLUMNS]
    return required, optional


def _feature_manifest_signature(feature_cols: Sequence[str], scale_cols: Sequence[str]) -> str:
    return _manifest_schema_signature(feature_cols, scale_cols)


def validate_scaler_feature_order(scaler: Any, numeric_cols: Sequence[str]) -> None:
    scaler_cols = list(getattr(scaler, "feature_names_in_", []))
    numeric_list = list(numeric_cols)
    if scaler_cols and scaler_cols != numeric_list:
        raise ValueError(
            "Scaler feature order mismatch: "
            f"scaler={scaler_cols}, numeric_cols={numeric_list}"
        )


def align_program_feature_frame(
    df: pd.DataFrame,
    manifest: Mapping[str, Any] | None,
    *,
    mode: str = STRICT_ALIGNMENT,
) -> pd.DataFrame:
    if df is None:
        raise ValueError("Program feature frame is None")

    out = df.copy()
    manifest = dict(manifest or {})
    expected_cols = list(manifest.get("feature_cols") or manifest.get("feature_order") or [])
    required_cols = list(manifest.get("required_feature_cols") or manifest.get("required_features") or [])
    optional_cols = list(manifest.get("optional_feature_cols") or [])
    feature_cols = list(expected_cols or required_cols)

    if not feature_cols and mode == STRICT_ALIGNMENT:
        feature_cols = [column for column in out.columns if column not in PROGRAM_LEAKAGE_BLACKLIST]

    missing_required = [column for column in required_cols if column not in out.columns]
    if missing_required:
        raise ValueError(f"Program inference frame missing required columns: {missing_required}")

    if mode == STRICT_ALIGNMENT:
        assert_program_feature_alignment(feature_cols, [column for column in out.columns if column in feature_cols])
        unexpected = [column for column in out.columns if column not in set(feature_cols) | set(PROGRAM_ID_COLS) | set(PROGRAM_LABEL_COLS)]
        if unexpected:
            raise ValueError(f"Program inference frame contains unexpected columns: {unexpected}")
        return out

    fill_values = dict(manifest.get("imputation_values") or {})
    for column in feature_cols:
        if column not in out.columns:
            if column in optional_cols or column in PROGRAM_OPTIONAL_ALIGNMENT_COLUMNS:
                out[column] = fill_values.get(column, 0.0)
                logger.warning("Filled missing optional program feature '%s' with default value", column)
            else:
                raise ValueError(f"Program inference frame missing required feature: {column}")

    unexpected = [column for column in out.columns if column not in set(feature_cols) | set(PROGRAM_ID_COLS) | set(PROGRAM_LABEL_COLS) | set(PROGRAM_LEAKAGE_BLACKLIST)]
    if unexpected:
        logger.warning("Ignoring unexpected program inference columns: %s", sorted(unexpected))

    aligned = out[[column for column in feature_cols if column in out.columns] + [column for column in out.columns if column not in feature_cols]]
    return aligned


def validate_program_artifact_compatibility(
    manifest: Mapping[str, Any],
    scaler: Any | None,
    model: Any | None = None,
) -> None:
    if not manifest:
        raise ValueError("Program manifest is missing or empty")

    manifest_version = manifest.get("preprocessing_version")
    if manifest_version != PROGRAM_PREPROCESSING_VERSION:
        raise ValueError(
            f"Program preprocessing version mismatch: manifest={manifest_version}, code={PROGRAM_PREPROCESSING_VERSION}"
        )

    if scaler is not None:
        manifest_scale_cols = list(manifest.get("scaler_feature_cols") or manifest.get("numeric_cols") or [])
        validate_scaler_feature_order(scaler, manifest_scale_cols)

    if model is not None:
        model_features = list(getattr(model, "feature_name_", [])) or list(getattr(model, "trained_feature_names_", []))
        manifest_features = list(manifest.get("feature_cols") or manifest.get("feature_order") or [])
        if model_features and manifest_features and model_features != manifest_features:
            raise ValueError(
                "Program model feature mismatch: "
                f"model={model_features}, manifest={manifest_features}"
            )


def _detect_undeclared_binary_features(df: pd.DataFrame, declared_binary_cols: Iterable[str]) -> list[str]:
    declared = set(declared_binary_cols)
    candidates: list[str] = []
    for column in df.columns:
        if column in declared or column in PROGRAM_ID_COLS or column in PROGRAM_LABEL_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        values = set(pd.Series(df[column]).dropna().unique().tolist())
        if values and values.issubset({0, 1, 0.0, 1.0}):
            candidates.append(column)
    if candidates:
        logger.warning(
            "Detected undeclared binary-like columns that remain unscaled: %s",
            sorted(candidates),
        )
    return candidates


def _detect_count_like_column(column: str) -> bool:
    """Detect count-like columns using suffix matching + explicit registry.

    Uses suffix-only detection (_count, _counts) plus the explicit
    PROGRAM_COUNT_FEATURES registry.  Avoids unsafe substring matching
    (e.g. 'account_name' should not be treated as a count column).
    """
    lowered = column.lower()
    return any(lowered.endswith(suffix) for suffix in COUNT_LIKE_SUFFIXES) or column in PROGRAM_COUNT_FEATURES


def _detect_ratio_like_column(column: str) -> bool:
    return column in PROGRAM_RATIO_FEATURES


def _validate_column_family(column: str, series: pd.Series, expected_family: str) -> None:
    if expected_family == "string":
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            raise ValueError(f"Column '{column}' expected string-like dtype, found {series.dtype}")
        return
    if expected_family == "numeric" and not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{column}' expected numeric dtype, found {series.dtype}")


def validate_program_schema(
    df: pd.DataFrame,
    *,
    expected_columns: Sequence[str] | None = None,
    required_columns: Sequence[str] | None = None,
    feature_dtypes: Mapping[str, str] | None = None,
    allow_extra_columns: bool = False,
) -> Dict[str, list[str]]:
    """Validate schema drift loudly.

    Returns a report dict to aid debugging.
    """
    if df is None:
        raise ValueError("Program feature frame is None")

    duplicates = _is_duplicate_columns(df)
    if duplicates:
        raise ValueError(f"Program feature frame contains duplicated columns: {duplicates}")

    columns = list(df.columns)
    expected = list(expected_columns) if expected_columns is not None else list(EXPECTED_PROGRAM_FEATURES)
    required = list(required_columns) if required_columns is not None else list(REQUIRED_FEATURES)
    expected_set = set(expected)
    column_set = set(columns)

    missing_required = [column for column in required if column not in column_set]
    if missing_required:
        raise ValueError(f"Program feature frame missing required columns: {missing_required}")

    missing_expected = [column for column in expected if column not in column_set]
    unexpected = [column for column in columns if column not in expected_set]
    if not allow_extra_columns and unexpected:
        raise ValueError(f"Program feature frame contains unexpected columns: {unexpected}")

    dtype_mismatches: list[str] = []
    if feature_dtypes:
        for column, family in feature_dtypes.items():
            if column not in df.columns:
                continue
            try:
                _validate_column_family(column, df[column], family)
            except ValueError as exc:
                dtype_mismatches.append(str(exc))
        if dtype_mismatches:
            raise ValueError("; ".join(dtype_mismatches))

    leakage_present = [column for column in PROGRAM_LEAKAGE_BLACKLIST if column in column_set]
    # Regex-based leakage detection for runtime columns not in static blacklist
    regex_leakage = detect_leakage_columns([c for c in columns if c not in PROGRAM_LEAKAGE_BLACKLIST])
    all_leakage = sorted(set(leakage_present + regex_leakage))
    if all_leakage:
        raise ValueError(f"Program feature frame contains leakage columns: {all_leakage}")

    return {
        "missing_required": missing_required,
        "missing_expected": missing_expected,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "dtype_mismatches": dtype_mismatches,
        "leakage_present": leakage_present,
    }


def assert_program_feature_alignment(train_cols: Sequence[str], inference_cols: Sequence[str]) -> None:
    """Fail loudly when feature order or membership drifts."""
    train_list = list(train_cols)
    inference_list = list(inference_cols)
    if train_list != inference_list:
        raise ValueError(
            "Program train/inference feature alignment mismatch: "
            f"train={train_list}, inference={inference_list}"
        )


def _compute_imputation_values(df: pd.DataFrame, numeric_cols: Sequence[str]) -> Dict[str, float]:
    imputation_values: Dict[str, float] = {}
    for column in numeric_cols:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        if column in PROGRAM_BINARY_COLS:
            imputation_values[column] = 0.0
            continue
        if _detect_count_like_column(column):
            imputation_values[column] = 0.0
            continue
        median = float(series.replace([np.inf, -np.inf], np.nan).median())
        if np.isnan(median):
            median = 0.0
        imputation_values[column] = median
    return imputation_values


def _apply_imputation_values(df: pd.DataFrame, imputation_values: Mapping[str, float]) -> pd.DataFrame:
    out = df.copy()
    for column, fill_value in imputation_values.items():
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return out


def _validate_numeric_ranges(df: pd.DataFrame, numeric_cols: Sequence[str], *, clip_numeric: bool = False) -> pd.DataFrame:
    out = df.copy()
    for column in numeric_cols:
        if column not in out.columns:
            continue
        series = pd.to_numeric(out[column], errors="coerce")
        if series.isna().all():
            logger.warning("Program numeric column '%s' is entirely missing after coercion.", column)
            continue
        if _detect_count_like_column(column):
            negative_mask = series < 0
            if negative_mask.any():
                logger.warning(
                    "Program count-like column '%s' has %d negative values.",
                    column, int(negative_mask.sum()),
                )
                if clip_numeric:
                    series = series.clip(lower=0)
        elif _detect_ratio_like_column(column):
            out_of_bounds = (~series.isna()) & ((series < 0) | (series > 1))
            if out_of_bounds.any():
                logger.warning(
                    "Program ratio-like column '%s' has %d out-of-range values.",
                    column, int(out_of_bounds.sum()),
                )
                if clip_numeric:
                    series = series.clip(lower=0, upper=1)
        finite = series.replace([np.inf, -np.inf], np.nan)
        q1 = finite.quantile(0.25)
        q3 = finite.quantile(0.75)
        if pd.notna(q1) and pd.notna(q3):
            iqr = q3 - q1
            if iqr > 0:
                upper = q3 + _OUTLIER_IQR_MULTIPLIER * iqr
                lower = q1 - _OUTLIER_IQR_MULTIPLIER * iqr
                outlier_mask = (finite < lower) | (finite > upper)
                if outlier_mask.any():
                    logger.warning(
                        "Program numeric column '%s' has %d extreme outliers.",
                        column, int(outlier_mask.sum()),
                    )
        out[column] = series
    return out


def _run_program_diagnostics(train_df: pd.DataFrame, eval_df: pd.DataFrame | None, columns: Sequence[str]) -> None:
    for column in columns:
        if column not in train_df.columns:
            continue
        train_series = pd.to_numeric(train_df[column], errors="coerce")
        if train_series.isna().all():
            logger.warning("Program training column '%s' is empty after numeric coercion.", column)
            continue
        if train_series.nunique(dropna=True) <= 1:
            logger.warning("Program training column '%s' has zero variance.", column)
        if eval_df is None or column not in eval_df.columns:
            continue
        eval_series = pd.to_numeric(eval_df[column], errors="coerce")
        train_nan = float(train_series.isna().mean())
        eval_nan = float(eval_series.isna().mean())
        if eval_nan > max(train_nan * _NAN_SPIKE_RELATIVE_THRESHOLD, train_nan + _NAN_SPIKE_ABSOLUTE_THRESHOLD) and eval_nan > _NAN_SPIKE_FLOOR:
            logger.warning(
                "Program column '%s' has a noticeable NaN spike in evaluation data: train=%.3f, eval=%.3f.",
                column, train_nan, eval_nan,
            )
        train_median = float(train_series.median())
        eval_median = float(eval_series.median()) if not eval_series.isna().all() else np.nan
        if pd.notna(train_median) and pd.notna(eval_median):
            denom = abs(train_median) + 1e-9
            shift = abs(eval_median - train_median) / denom
            if shift > _MEDIAN_DRIFT_THRESHOLD:
                logger.warning(
                    "Program column '%s' median drift looks large: train=%.4f, eval=%.4f.",
                    column, train_median, eval_median,
                )


def _normalize_program_numeric_frame(df: pd.DataFrame, *, clip_numeric: bool = False) -> pd.DataFrame:
    out = df.copy()
    out = _normalize_enum_columns(out)
    out = _normalize_boolean_columns(out)

    if "label" in out.columns:
        out["label"] = pd.to_numeric(out["label"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

    numeric_cols = _numeric_like_columns(out)
    for column in numeric_cols:
        if column in PROGRAM_BINARY_COLS or column == "label":
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    out = _validate_numeric_ranges(out, numeric_cols, clip_numeric=clip_numeric)
    _detect_undeclared_binary_features(out, PROGRAM_BINARY_COLS)
    return out


def prepare_program_feature_frame(
    df: pd.DataFrame,
    *,
    clip_numeric: bool = False,
) -> pd.DataFrame:
    """Normalize dtypes and fill missing values for program features.

    This is intentionally compatible with the existing pipeline, but now performs
    stricter and more explicit coercion before imputation.
    """
    if df is None:
        raise ValueError("Program feature frame is None")

    out = _normalize_program_numeric_frame(df, clip_numeric=clip_numeric)
    numeric_cols = _numeric_like_columns(out)
    imputation_values = _compute_imputation_values(out, numeric_cols)
    out = _apply_imputation_values(out, imputation_values)
    return out


def _split_scaler_groups(numeric_cols: Sequence[str]) -> tuple[list[str], list[str]]:
    bounded_cols = [column for column in numeric_cols if column in PROGRAM_RATIO_FEATURES]
    robust_cols = [column for column in numeric_cols if column not in PROGRAM_RATIO_FEATURES]
    return bounded_cols, robust_cols


def fit_program_scaler(
    df: pd.DataFrame,
    numeric_cols: list[str],
    out_path: str | Path | None = None,
    *,
    strategy: str = LEGACY_STRATEGY,
):
    if df is None:
        raise ValueError("Program scaler fit frame is None")
    if not numeric_cols:
        raise ValueError("numeric_cols must not be empty")

    # Enforce leakage blacklist — these columns must never be scaled
    leakage_in_cols = [c for c in numeric_cols if c in PROGRAM_LEAKAGE_BLACKLIST]
    if leakage_in_cols:
        raise ValueError(f"Leakage columns passed to scaler fit: {leakage_in_cols}")

    missing = [column for column in numeric_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Program scaler fit frame missing numeric columns: {missing}")

    data = df[numeric_cols].fillna(0).astype(float)
    if strategy == GROUPED_STRATEGY:
        bounded_cols, robust_cols = _split_scaler_groups(numeric_cols)
        scaler: Any = ProgramFeatureScaler(strategy=strategy, bounded_cols=bounded_cols, robust_cols=robust_cols)
        scaler.fit(data)
    else:
        scaler = MinMaxScaler()
        scaler.fit(data)

    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, path)
    return scaler


def apply_program_scaler(
    df: pd.DataFrame | None,
    scaler: Any,
    numeric_cols: list[str],
    *,
    feature_cols: Sequence[str] | None = None,
    allow_extra_columns: bool = True,
) -> pd.DataFrame | None:
    if df is None:
        return None

    out = df.copy()
    missing = [column for column in numeric_cols if column not in out.columns]
    if missing:
        raise ValueError(f"Program scaler apply frame missing numeric columns: {missing}")

    if feature_cols is not None:
        feature_cols = list(feature_cols)
        missing_features = [column for column in feature_cols if column not in out.columns]
        if missing_features:
            raise ValueError(f"Program scaler apply frame missing feature columns: {missing_features}")
        if not allow_extra_columns:
            unexpected = [column for column in out.columns if column not in set(feature_cols) | set(numeric_cols)]
            if unexpected:
                raise ValueError(f"Program scaler apply frame contains unexpected columns: {unexpected}")

    # Enforce leakage blacklist at apply time too
    leakage_in_cols = [c for c in numeric_cols if c in PROGRAM_LEAKAGE_BLACKLIST]
    if leakage_in_cols:
        raise ValueError(f"Leakage columns passed to scaler apply: {leakage_in_cols}")

    data = out[numeric_cols].fillna(0).astype(float)
    if hasattr(scaler, "transform"):
        scaled = scaler.transform(data)
    else:
        raise TypeError(f"Scaler object does not implement transform(): {type(scaler)!r}")

    # Shape validation — catch silent dimension mismatches from stale scalers
    if scaled.shape != data.shape:
        raise ValueError(
            f"Scaler output shape {scaled.shape} != input shape {data.shape}; "
            "scaler may be stale or trained on different columns"
        )

    # Explicit DataFrame reconstruction — preserves index, ordering, dtypes,
    # and alignment without relying on .loc assignment which can fail silently
    # on filtered or reindexed frames.
    if isinstance(scaled, pd.DataFrame):
        scaled_values = scaled.values
    else:
        scaled_values = scaled
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_cols, index=out.index)
    for col in numeric_cols:
        out[col] = scaled_df[col].values
    return out


def build_program_feature_manifest(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    scale_cols: Sequence[str],
    *,
    scaler_strategy: str = LEGACY_STRATEGY,
    imputation_values: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Build the program feature manifest.

    ``feature_cols`` is the CANONICAL source of truth for the full model feature
    space.  It MUST include all features the model uses: numeric, binary,
    eligibility, CF, popularity, and engineered categoricals.

    ``scale_cols`` contains ONLY the numeric columns that are scaled.
    These are stored separately as ``scaler_feature_cols`` and must not be
    confused with the full feature set.

    Legacy aliases (feature_names, feature_order) are preserved for backward
    compatibility but point to the same canonical feature_cols list.

    Feature groups (eligibility, business_rule) are metadata-only tags for
    explainability and auditing — they do NOT affect preprocessing behavior.
    """
    feature_cols = list(feature_cols)
    scale_cols = list(scale_cols)

    # Validate that feature_cols is a superset of scale_cols
    # (manifest must represent the FULL model feature space, not just scaled cols)
    missing_scale = [c for c in scale_cols if c not in feature_cols]
    if missing_scale:
        logger.warning(
            "Scale columns not in feature_cols (will be added): %s", missing_scale,
        )
        feature_cols = list(dict.fromkeys([*feature_cols, *missing_scale]))

    # Validate that binary features present in training are included
    binary_in_train = [c for c in PROGRAM_BINARY_COLS if c in train_df.columns]
    missing_binary = [c for c in binary_in_train if c not in feature_cols]
    if missing_binary:
        logger.warning(
            "Binary features missing from feature_cols (will be added): %s", missing_binary,
        )
        feature_cols = list(dict.fromkeys([*feature_cols, *missing_binary]))

    required_feature_cols, optional_feature_cols = _split_optional_and_required_feature_cols(feature_cols)
    feature_groups = infer_program_feature_groups(feature_cols)
    feature_dtypes = {
        column: str(train_df[column].dtype)
        for column in feature_cols
        if column in train_df.columns
    }
    timestamp = datetime.now(timezone.utc).isoformat()

    # Feature lineage metadata (Item 18) — tags for explainability/auditing.
    # Eligibility and business_rule are metadata-only concepts (Item 7);
    # they do NOT drive preprocessing logic.
    feature_lineage = {
        col: _classify_feature_lineage(col) for col in feature_cols
    }

    manifest: Dict[str, Any] = {
        "artifact_version": PROGRAM_MANIFEST_ARTIFACT_VERSION,
        "preprocessing_version": PROGRAM_PREPROCESSING_VERSION,
        "preprocessing_timestamp": timestamp,
        "dataset_fingerprint": _dataset_fingerprint(train_df),
        "training_rows": int(len(train_df)),
        "training_columns": int(len(train_df.columns)),
        # Canonical feature source — FULL model feature space
        "feature_cols": feature_cols,
        # Deprecated aliases (backward compat) — point to same canonical list
        "feature_names": feature_cols,
        "feature_order": feature_cols,
        "feature_groups": feature_groups,
        "feature_lineage": feature_lineage,
        "feature_traceability": {
            "business_rule_features": feature_groups["business_rule"],
            "ml_only_features": feature_groups["ml_only"],
            "cf_features": feature_groups["cf"],
            "popularity_features": feature_groups["popularity"],
            "eligibility_features": feature_groups["eligibility"],
            "reranking_only_features": feature_groups.get("reranking_only", []),
        },
        "feature_dtypes": feature_dtypes,
        "binary_cols": sorted(binary_in_train),
        # Scaler columns — separate from full feature space
        "scaler_feature_cols": scale_cols,
        "scaler_strategy": scaler_strategy,
        "required_feature_cols": required_feature_cols,
        "optional_feature_cols": optional_feature_cols,
        "excluded_columns": sorted(
            {
                *PROGRAM_ID_COLS,
                *PROGRAM_LABEL_COLS,
                *PROGRAM_INTERNAL_EXCLUDED_COLS,
                *PROGRAM_LEAKAGE_BLACKLIST,
            }
        ),
        "required_features": list(REQUIRED_FEATURES),
        "expected_features": list(EXPECTED_PROGRAM_FEATURES),
        "train_columns": list(train_df.columns),
        "training_schema_signature": _feature_manifest_signature(feature_cols, scale_cols),
        "imputation_values": dict(imputation_values or {}),
        # Deprecated aliases — use scaler_feature_cols instead
        "numeric_cols": scale_cols,
        "numeric_columns": scale_cols,
        "binary_columns": sorted(binary_in_train),
    }
    return manifest


def _classify_feature_lineage(column: str) -> str:
    """Classify a feature column by its lineage for explainability metadata.

    Returns one of: 'deterministic_eligibility', 'business_rule', 'cf_derived',
    'popularity_derived', 'ml_only', 'reranking_only'.
    """
    if column in PROGRAM_FEATURE_GROUPS.get("eligibility", set()):
        return "deterministic_eligibility"
    if column in PROGRAM_FEATURE_GROUPS.get("business_rule", set()):
        return "business_rule"
    if column in PROGRAM_INTERACTION_FEATURES:
        return "cf_derived"
    if column in PROGRAM_POPULARITY_FEATURES:
        return "popularity_derived"
    if column in PROGRAM_INTERNAL_EXCLUDED_COLS:
        return "reranking_only"
    return "ml_only"


def fit_and_apply_program_scaler(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
    numeric_cols: list[str] | None = None,
    out_path: str | None = None,
    *,
    feature_cols: Sequence[str] | None = None,
    scaler_strategy: str = LEGACY_STRATEGY,
    clip_numeric: bool = False,
    alignment_mode: str = STRICT_ALIGNMENT,
) -> tuple[pd.DataFrame, pd.DataFrame | None, Any, list[str]]:
    """Fit scaler on train and apply to train/eval using aligned columns."""
    train_normalized = _normalize_program_numeric_frame(train_df, clip_numeric=clip_numeric)
    eval_normalized = _normalize_program_numeric_frame(eval_df, clip_numeric=clip_numeric) if eval_df is not None else None

    if eval_normalized is not None:
        assert_program_feature_alignment(train_normalized.columns.tolist(), eval_normalized.columns.tolist())

    manifest_feature_cols = list(feature_cols) if feature_cols is not None else get_program_numeric_feature_cols(train_normalized)
    # Include internal excluded cols (e.g. hardness_score) in expected set —
    # they are legitimately present in training data but excluded from model features.
    training_expected_cols = _ordered_unique([
        *PROGRAM_ID_COLS, *PROGRAM_LABEL_COLS, *manifest_feature_cols,
        *[c for c in PROGRAM_INTERNAL_EXCLUDED_COLS if c in train_normalized.columns],
    ])
    training_required_cols = _ordered_unique([*PROGRAM_ID_COLS, *PROGRAM_LABEL_COLS, *manifest_feature_cols])
    training_dtypes = {
        column: FEATURE_DTYPES.get(column, str(train_normalized[column].dtype))
        for column in manifest_feature_cols
        if column in train_normalized.columns
    }
    training_dtypes.update({column: FEATURE_DTYPES.get(column, "string") for column in PROGRAM_LABEL_COLS if column in train_normalized.columns})
    validate_program_schema(
        train_normalized,
        expected_columns=training_expected_cols,
        required_columns=training_required_cols,
        feature_dtypes=training_dtypes,
        allow_extra_columns=False,
    )
    if eval_normalized is not None:
        validate_program_schema(
            eval_normalized,
            expected_columns=training_expected_cols,
            required_columns=training_required_cols,
            feature_dtypes=training_dtypes,
            allow_extra_columns=False,
        )

    cols = list(numeric_cols) if numeric_cols is not None else get_program_numeric_feature_cols(train_normalized)
    cols = [
        column
        for column in cols
        if column not in PROGRAM_BINARY_COLS
        and column not in PROGRAM_ID_COLS
        and column not in PROGRAM_LABEL_COLS
        and column not in PROGRAM_LEAKAGE_BLACKLIST
        and column not in PROGRAM_INTERNAL_EXCLUDED_COLS
    ]
    if not cols:
        raise ValueError("Program numeric feature column list is empty")

    missing_numeric = [column for column in cols if column not in train_normalized.columns]
    if missing_numeric:
        raise ValueError(f"Training frame missing numeric columns: {missing_numeric}")

    if feature_cols is not None:
        feature_cols = list(feature_cols)
        missing_features = [column for column in feature_cols if column not in train_normalized.columns]
        if missing_features:
            raise ValueError(f"Training frame missing feature columns: {missing_features}")
        if eval_normalized is not None:
            eval_missing = [column for column in feature_cols if column not in eval_normalized.columns]
            if eval_missing:
                raise ValueError(f"Evaluation frame missing feature columns: {eval_missing}")
        # Real train-vs-eval feature alignment validation (replaces self-referential check)
        if eval_normalized is not None:
            train_feature_cols = [c for c in feature_cols if c in train_normalized.columns]
            eval_feature_cols = [c for c in feature_cols if c in eval_normalized.columns]
            assert_program_feature_alignment(train_feature_cols, eval_feature_cols)

    imputation_values = _compute_imputation_values(train_normalized, cols)
    train_imputed = _apply_imputation_values(train_normalized, imputation_values)
    eval_imputed = _apply_imputation_values(eval_normalized, imputation_values) if eval_normalized is not None else None

    _run_program_diagnostics(train_imputed, eval_imputed, cols)

    scaler = fit_program_scaler(train_imputed, cols, out_path=out_path, strategy=scaler_strategy)
    validate_scaler_feature_order(scaler, cols)
    train_scaled = apply_program_scaler(train_imputed, scaler, cols, feature_cols=feature_cols)
    eval_scaled = apply_program_scaler(eval_imputed, scaler, cols, feature_cols=feature_cols) if eval_imputed is not None else None

    train_binary_cols = [column for column in PROGRAM_BINARY_COLS if column in train_scaled.columns]
    for column in train_binary_cols:
        unique_values = set(pd.Series(train_scaled[column]).dropna().unique().tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(f"Binary column '{column}' was altered during preprocessing: {sorted(unique_values)}")

    if feature_cols is not None and eval_scaled is not None:
        assert_program_feature_alignment(
            [column for column in feature_cols if column in train_scaled.columns],
            [column for column in feature_cols if column in eval_scaled.columns],
        )

    # Build full model feature space for manifest (Item 6 fix)
    # When feature_cols is None, build from train frame to include ALL feature types
    # (binary, eligibility, engineered categorical) — not just numeric/scaled cols.
    if feature_cols is not None:
        full_feature_cols = list(feature_cols)
    else:
        excluded = PROGRAM_ID_COLS | PROGRAM_LABEL_COLS | PROGRAM_LEAKAGE_BLACKLIST | PROGRAM_INTERNAL_EXCLUDED_COLS
        full_feature_cols = [c for c in train_scaled.columns if c not in excluded]

    manifest = build_program_feature_manifest(
        train_scaled,
        full_feature_cols,
        cols,
        scaler_strategy=scaler_strategy,
        imputation_values=imputation_values,
    )

    if out_path:
        manifest_path = Path(out_path).with_name("program_feature_manifest.json")
        save_json_artifact(manifest, manifest_path)

    logger.info(
        "Program preprocessing summary: train_rows=%d eval_rows=%d feature_cols=%d binary_cols=%d numeric_cols=%d excluded_cols=%d",
        len(train_scaled),
        0 if eval_scaled is None else len(eval_scaled),
        len(manifest["feature_cols"]),
        len(manifest["binary_cols"]),
        len(cols),
        len(manifest["excluded_columns"]),
    )
    logger.info("Program scaler feature groups: strategy=%s scale_cols=%s", scaler_strategy, cols)
    logger.info("Program leakage exclusions: %s", manifest["excluded_columns"])
    logger.info("Program binary columns: %s", manifest["binary_cols"])

    return train_scaled, eval_scaled, scaler, cols
