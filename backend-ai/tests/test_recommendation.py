"""Simple local recommendation test — evaluate ranking quality.

Usage:
    cd mentorship-ai-assistant-mvp/backend-ai
    python test_recommendation.py                     # pipeline eval
    python test_recommendation.py <user_id>           # single-user test
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_recommendation")

# Add project root to path so we can import the ML pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Single-user test (online DB-based scoring)
# ---------------------------------------------------------------------------

async def run_recommendation_test(user_id: str) -> None:
    """Load recommendations for a user and print Top 10 with score breakdown."""
    from services.recommendation_service import recommendation_service

    print(f"\n{'=' * 60}")
    print(f"  RECOMMENDATION TEST — user_id: {user_id}")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    recs = await recommendation_service.get_recommendations(user_id=user_id)
    elapsed = time.perf_counter() - t0

    if not recs:
        print("  No recommendations returned.")
        print(f"  Time: {elapsed:.2f} sec")
        return

    print(f"\n  Top {len(recs)} Mentors (response time: {elapsed:.2f} sec):\n")
    print(f"  {'#':<4} {'Mentor':<25} {'Domain':<15} {'Match%':<8} {'Score':<8} {'Reason'}")
    print(f"  {'-'*4} {'-'*25} {'-'*15} {'-'*8} {'-'*8} {'-'*40}")
    for i, rec in enumerate(recs, 1):
        name = str(rec.get("mentor_name", "?"))[:24]
        domain = str(rec.get("domain", "?"))[:14]
        pct = int(rec.get("match_percentage", 75))
        score = float(rec.get("score", 0))
        reason = str(rec.get("reason", ""))[:45]
        print(f"  {i:<4} {name:<25} {domain:<15} {pct:<8}% {score:<8.3f} {reason}")

    print(f"\n{'=' * 60}")


# ---------------------------------------------------------------------------
# Pipeline evaluation (offline ML metrics)
# ---------------------------------------------------------------------------


def _precision_at_k(labels: np.ndarray, k: int) -> float:
    """Precision@K: fraction of top-K items that are relevant."""
    return float(labels[:k].sum()) / k if k > 0 else 0.0


def _average_precision_at_k(labels: np.ndarray, k: int) -> float:
    """Average Precision@K for a single ranked list."""
    hits = 0.0
    score = 0.0
    for i in range(min(k, len(labels))):
        if labels[i] > 0:
            hits += 1
            score += hits / (i + 1)
    return score / min(k, int(labels.sum())) if labels.sum() > 0 else 0.0


def _recall_at_k(labels: np.ndarray, k: int) -> float:
    """Recall@K: fraction of all positives found in top-K."""
    total_pos = labels.sum()
    if total_pos == 0:
        return 0.0
    return float(labels[:k].sum()) / total_pos


def run_pipeline_eval() -> None:
    """Load pre-trained artifacts and evaluate full recommendation metrics."""
    from sklearn.metrics import ndcg_score

    artifacts_dir = PROJECT_ROOT / "data" / "artifacts"
    features_dir = PROJECT_ROOT / "data" / "features"

    print(f"\n{'=' * 60}")
    print("  RECOMMENDATION PIPELINE EVALUATION")
    print(f"{'=' * 60}")

    # --- Load artifacts ---
    model_path = artifacts_dir / "model.joblib"
    if not model_path.exists():
        print(f"  ERROR: Model not found at {model_path}")
        print("  Run the training pipeline first: python run_eval.py")
        return

    try:
        from src.hybrid_recommender.pipeline import load_inference_artifacts
        bundle = load_inference_artifacts(artifacts_dir)
    except Exception as exc:
        print(f"  ERROR loading artifacts: {exc}")
        return

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    rec_features = bundle["recommendation_features"]

    print(f"  Model loaded: {model_path}")
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Total rows: {len(rec_features):,}")

    # --- Check for time_split column ---
    if "time_split" not in rec_features.columns:
        print("  WARNING: No time_split column — evaluating on all data (may be inflated)")
        eval_df = rec_features.copy()
    else:
        test_df = rec_features[rec_features["time_split"] == "test"].copy()
        if test_df.empty:
            print("  WARNING: No test split data — using valid split")
            test_df = rec_features[rec_features["time_split"] == "valid"].copy()
        eval_df = test_df
        print(f"  Evaluation set: {len(eval_df):,} rows (test split)")

    if "label" not in eval_df.columns:
        print("  ERROR: No 'label' column in features — cannot evaluate")
        return

    # --- Score ---
    eval_df = eval_df.copy()
    eval_df["pred_score"] = model.predict(eval_df[feature_cols])

    # --- Compute metrics per group ---
    K = 10
    ndcg_vals = []
    hitrate_vals = []
    precision_vals = []
    map_vals = []
    recall_vals = []
    evaluated = 0
    skipped = 0

    for _, grp in eval_df.groupby("mentee_id"):
        if grp["label"].sum() == 0 or len(grp) < 5:
            skipped += 1
            continue
        evaluated += 1

        grp = grp.sort_values("pred_score", ascending=False)
        labels = grp["label"].to_numpy()
        preds = grp["pred_score"].to_numpy()

        k = min(K, len(grp))
        ndcg_vals.append(ndcg_score(labels.reshape(1, -1), preds.reshape(1, -1), k=k))
        hitrate_vals.append(int(labels[:k].sum() > 0))
        precision_vals.append(_precision_at_k(labels, k))
        map_vals.append(_average_precision_at_k(labels, k))
        recall_vals.append(_recall_at_k(labels, k))

    # --- Print results ---
    print(f"\n  Evaluated groups: {evaluated} (skipped: {skipped})")
    print(f"\n  {'Metric':<20} {'Value':<10}")
    print(f"  {'-'*20} {'-'*10}")

    metrics = {
        f"NDCG@{K}": np.mean(ndcg_vals) if ndcg_vals else float("nan"),
        f"HitRate@{K}": np.mean(hitrate_vals) if hitrate_vals else float("nan"),
        f"Precision@{K}": np.mean(precision_vals) if precision_vals else float("nan"),
        f"MAP@{K}": np.mean(map_vals) if map_vals else float("nan"),
        f"Recall@{K}": np.mean(recall_vals) if recall_vals else float("nan"),
    }

    for name, val in metrics.items():
        print(f"  {name:<20} {val:<10.4f}")

    # --- Feature importance ---
    if hasattr(model, "feature_importance_df_"):
        print(f"\n  Top 10 Feature Importance (gain):")
        for _, row in model.feature_importance_df_.head(10).iterrows():
            print(f"    {row['feature']:<30} {row['importance_gain']:>10.1f}")

    print(f"\n{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        uid = sys.argv[1]
        asyncio.run(run_recommendation_test(uid))
    else:
        run_pipeline_eval()
