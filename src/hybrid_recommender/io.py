"""I/O utilities for loading/saving data, models, and artifacts."""

from pathlib import Path
from typing import Any, Dict

import json
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DB = ROOT / "data" / "database_import_ready"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_FEATURES = ROOT / "data" / "features"
DATA_ARTIFACTS = ROOT / "data" / "artifacts"


def get_project_paths() -> Dict[str, Path]:
    """Return canonical project paths used across the pipeline."""
    return {
        "root": ROOT,
        "db": DATA_DB,
        "raw": DATA_RAW,
        "processed": DATA_PROCESSED,
        "features": DATA_FEATURES,
        "artifacts": DATA_ARTIFACTS,
    }


def load_csv(name: str, folder: Path | None = None) -> pd.DataFrame:
    """Load a CSV file from a project directory (default: data/raw)."""
    base_folder = folder or DATA_RAW
    path = base_folder / name
    if not path.exists():
        raise FileNotFoundError(f"{name} not found in {base_folder}")
    return pd.read_csv(path)


def load_feature_csv(name: str) -> pd.DataFrame:
    """Load a CSV file from data/features."""
    return load_csv(name, DATA_FEATURES)


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame to data/processed."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / name
    df.to_csv(path, index=False)
    return path


def save_features(df: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame to data/features."""
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    path = DATA_FEATURES / name
    df.to_csv(path, index=False)
    return path


def save_model(model: Any, path: Path) -> Path:
    """Save a trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> Any:
    """Load a trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def save_scaler(scaler: Any, path: Path) -> Path:
    """Save a fitted scaler to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return path


def load_scaler(path: Path) -> Any:
    """Load a fitted scaler from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {path}")
    return joblib.load(path)


def save_feature_artifact(df: pd.DataFrame, name: str) -> Path:
    """Save a feature DataFrame to data/artifacts."""
    DATA_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = DATA_ARTIFACTS / name
    df.to_csv(path, index=False)
    return path


def load_features(name: str) -> pd.DataFrame:
    """Load a feature artifact from data/artifacts."""
    path = DATA_ARTIFACTS / name
    if not path.exists():
        raise FileNotFoundError(f"Feature artifact not found: {path}")
    return pd.read_csv(path)


def save_json_artifact(payload: Dict[str, Any], path: Path) -> Path:
    """Save a JSON metadata artifact to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json_artifact(path: Path) -> Dict[str, Any]:
    """Load a JSON metadata artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
