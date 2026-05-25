"""Program-level recommender module.

This package contains IO, feature builders, preprocessing, pipeline and
ranking helpers for mentee→program recommendations. It is intentionally
lightweight and reuses utilities from the mentor recommender to keep
semantics consistent.
"""

from .io import get_program_paths, get_default_program_artifact_paths
from .features import (
    build_program_features,
    build_mentee_program_candidates,
    build_program_cf_embeddings,
    compute_cf_confidence,
    validate_pair_feature_distributions,
)
from .pipeline import (
    build_program_recommendation_dataset,
    run_program_pipeline,
    load_program_inference_artifacts,
    validate_pipeline_integrity,
)
from .ranking import (
    train_program_model,
    generate_program_recommendations,
    evaluate_program_model,
)

__all__ = [
    "get_program_paths",
    "get_default_program_artifact_paths",
    "build_program_features",
    "build_mentee_program_candidates",
    "build_program_cf_embeddings",
    "compute_cf_confidence",
    "validate_pair_feature_distributions",
    "build_program_recommendation_dataset",
    "run_program_pipeline",
    "load_program_inference_artifacts",
    "validate_pipeline_integrity",
    "train_program_model",
    "generate_program_recommendations",
    "evaluate_program_model",
]
