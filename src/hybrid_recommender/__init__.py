"""Hybrid Mentorship Recommendation System.

Public API:
    run_full_pipeline(raw_data)     — Train end-to-end from raw data
    load_inference_artifacts(path)  — Load pre-trained artifacts
    predict_for_user(user_id, data) — Generate recommendations for a user
    tune_pipeline(raw_data)         — Auto-tune hyperparameters
"""

from .io import get_project_paths, load_csv, load_feature_csv, save_features, save_processed
from .pipeline import load_inference_artifacts, predict_for_user, run_full_pipeline, tune_pipeline

