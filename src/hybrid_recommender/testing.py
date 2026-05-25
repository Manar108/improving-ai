from __future__ import annotations

from typing import Dict

import pandas as pd


def run_dataset_checks(df: pd.DataFrame) -> Dict[str, object]:
    """Run quick sanity checks on a recommendation dataset.

    Returns a report dict with shape, duplicate pairs, missing values,
    mixed-label pairs, coverage, class balance, and distribution statistics.
    """
    report: Dict[str, object] = {}
    report["shape"] = df.shape
    report["n_unique_mentees"] = int(df["mentee_id"].nunique()) if "mentee_id" in df.columns else None
    report["n_unique_mentors"] = int(df["mentor_id"].nunique()) if "mentor_id" in df.columns else None
    report["duplicate_pairs"] = int(df.duplicated(["mentee_id", "mentor_id"]).sum()) if {"mentee_id", "mentor_id"}.issubset(df.columns) else None

    # Missing values (excluding optional columns)
    optional = {"start_date", "event_time"}
    check_cols = [c for c in df.columns if c not in optional]
    report["missing_values_required"] = int(df[check_cols].isna().sum().sum())

    if {"mentee_id", "mentor_id", "label"}.issubset(df.columns):
        label_sets = df.groupby(["mentee_id", "mentor_id"])['label'].apply(lambda s: set(s.dropna().astype(int)))
        report["mixed_pairs"] = int((label_sets.apply(len) > 1).sum())
        report["label_distribution"] = df["label"].value_counts().to_dict()
        report["positive_ratio"] = round(df["label"].sum() / len(df), 4) if len(df) > 0 else 0.0
    else:
        report["mixed_pairs"] = None
        report["label_distribution"] = {}
        report["positive_ratio"] = 0.0

    report["time_split_distribution"] = df["time_split"].value_counts().to_dict() if "time_split" in df.columns else {}
    return report


def validate_pipeline_output(df: pd.DataFrame) -> tuple[bool, Dict[str, object]]:
    """Validate that the final dataset is suitable for training and evaluation.

    Checks for duplicate pairs, missing values, mixed labels, class balance,
    coverage, and empty splits.

    Returns:
        Tuple of (is_valid, report_dict) with detailed error messages.
    """
    report = run_dataset_checks(df)
    errors: list[str] = []

    if report["duplicate_pairs"] not in (0, None):
        errors.append(f"Found {report['duplicate_pairs']} duplicate (mentee, mentor) pairs")

    if report["missing_values_required"] > 0:
        errors.append(f"Found {report['missing_values_required']} missing values in required columns")

    if report["mixed_pairs"] not in (0, None):
        errors.append(f"Found {report['mixed_pairs']} pairs with conflicting labels")

    if report.get("positive_ratio", 0) == 0:
        errors.append("No positive labels (label=1) found")

    ts = report.get("time_split_distribution", {})
    if ts.get("valid", 0) == 0:
        errors.append("Empty validation split")
    if ts.get("test", 0) == 0:
        errors.append("Empty test split")

    ok = len(errors) == 0
    report["validation_errors"] = errors
    return ok, report
