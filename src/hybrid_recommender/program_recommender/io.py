from __future__ import annotations

"""Program recommender I/O utilities.

Built from the same patterns used in src/hybrid_recommender/io.py,
with dedicated artifact files for program recommendation.
"""

from pathlib import Path
from typing import Any, Dict

import json
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_FEATURES = ROOT / "data" / "features"
DATA_ARTIFACTS = ROOT / "data" / "artifacts"
PROGRAM_ARTIFACTS = DATA_ARTIFACTS / "program_recommender"


def get_program_paths() -> Dict[str, Path]:
    return {
        "root": ROOT,
        "raw": DATA_RAW,
        "processed": DATA_PROCESSED,
        "features": DATA_FEATURES,
        "artifacts": DATA_ARTIFACTS,
        "program_artifacts": PROGRAM_ARTIFACTS,
    }


def load_csv(name: str, folder: Path | None = None) -> pd.DataFrame:
    base_folder = folder or DATA_RAW
    path = base_folder / name
    if not path.exists():
        raise FileNotFoundError(f"{name} not found in {base_folder}")
    return pd.read_csv(path)


def load_program_raw_tables() -> Dict[str, pd.DataFrame]:
    """Load program recommender source tables from data/raw.

    Uses real table names from preprocessing DB_TABLE_MAP normalized outputs.
    """
    return {
        "mentee_profile": load_csv("mentee_profile.csv"),
        "mentee_subdomains": load_csv("MenteeSubDomains.csv"),
        "mentee_interests": load_csv("mentee_interests.csv"),
        "mentorship_posts": load_csv("programs.csv"),
        "mentorship_requirements": load_csv("mentorship_requirements.csv"),
        "mentorships": load_csv("mentorships.csv"),
        "mentorship_applications": load_csv("applications.csv"),
        "posts_likes_dataset": load_csv("mentorship_post_likes.csv"),
        "posts_comments": load_csv("mentorship_post_comments.csv"),
        "saved_posts_dataset": load_csv("saved_posts.csv"),
        "shared_posts_dataset": load_csv("shared_posts.csv"),
    }


def save_model(model: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def save_scaler(scaler: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return path


def load_scaler(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {path}")
    return joblib.load(path)


def save_feature_artifact(df: pd.DataFrame, name: str) -> Path:
    PROGRAM_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = PROGRAM_ARTIFACTS / name
    df.to_csv(path, index=False)
    return path


def load_feature_artifact(name: str) -> pd.DataFrame:
    path = PROGRAM_ARTIFACTS / name
    if not path.exists():
        raise FileNotFoundError(f"Program feature artifact not found: {path}")
    return pd.read_csv(path)


def save_json_artifact(payload: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json_artifact(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_default_program_artifact_paths() -> Dict[str, Path]:
    PROGRAM_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return {
        "model": PROGRAM_ARTIFACTS / "program_model.joblib",
        "scaler": PROGRAM_ARTIFACTS / "program_scaler.joblib",
        "dataset": PROGRAM_ARTIFACTS / "program_recommendation_features.csv",
        "program_features": PROGRAM_ARTIFACTS / "program_features.csv",
        "manifest": PROGRAM_ARTIFACTS / "program_feature_manifest.json",
    }
