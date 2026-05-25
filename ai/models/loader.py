"""Simple model/scaler loader with safe fallbacks for tests and CI.

This loader looks for artifacts in the following order:
1. Provided `bundle_path` (preferred)
2. The `ai/models/` folder next to this file

If artifacts are missing, a lightweight DummyModel is returned so tests can run.
"""
from pathlib import Path
from typing import Tuple, Dict, Any
import joblib
import json

MODEL_NAMES = ["model.joblib", "model.pkl", "model.safetensors"]
SCALER_NAMES = ["scaler.joblib", "scaler.pkl"]
METADATA_NAMES = ["metadata.json", "model_config.json", "config.json"]


class DummyModel:
    def predict(self, X):
        # return zeros with appropriate length
        try:
            return [0] * (len(X) if hasattr(X, '__len__') else 1)
        except Exception:
            return [0]

    def __call__(self, X):
        return self.predict(X)


def _find_file(dir_path: Path, names):
    for n in names:
        p = dir_path / n
        if p.exists():
            return p
    return None


def load_model(bundle_path: Path = None) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load model, scaler and metadata.

    Returns: (model, scaler_or_none, metadata_dict)
    If files not found, returns DummyModel(), None, {'dummy': True}
    """
    base = Path(bundle_path) if bundle_path else Path(__file__).parent
    # allow directing to data/artifacts
    candidates = [base, Path.cwd() / 'data' / 'artifacts', Path(__file__).parent]

    model_path = None
    scaler_path = None
    metadata_path = None

    for c in candidates:
        if not c:
            continue
        mp = _find_file(c, MODEL_NAMES)
        if mp:
            model_path = mp
        sp = _find_file(c, SCALER_NAMES)
        if sp:
            scaler_path = sp
        md = _find_file(c, METADATA_NAMES)
        if md:
            metadata_path = md
        if model_path or scaler_path or metadata_path:
            # prefer the first folder that has anything
            break

    model = None
    scaler = None
    metadata = {}

    if model_path:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Warning: failed to load model from {model_path}: {e}")
            model = None

    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Warning: failed to load scaler from {scaler_path}: {e}")
            scaler = None

    if metadata_path:
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load metadata from {metadata_path}: {e}")
            metadata = {}

    if model is None:
        print("No model artifact found — returning DummyModel.")
        model = DummyModel()
        metadata['dummy'] = True

    return model, scaler, metadata


def load_scaler(bundle_path: Path = None) -> Any:
    """Load only the scaler if present; otherwise return None."""
    base = Path(bundle_path) if bundle_path else Path(__file__).parent
    candidates = [base, Path.cwd() / 'data' / 'artifacts', Path(__file__).parent]
    scaler_path = None
    for c in candidates:
        sp = _find_file(c, SCALER_NAMES)
        if sp:
            scaler_path = sp
            break
    if scaler_path:
        try:
            return joblib.load(scaler_path)
        except Exception as e:
            print(f"Warning: failed to load scaler from {scaler_path}: {e}")
            return None
    return None
