"""Quality checks for mentor recommendations (mentee → mentor).

Validates:
  - ``match_percentage`` is a user-facing 60–100% scale, not a raw model score.
  - Ranking vs percentage consistency when coverage is held fixed.
  - ``generate_reason`` lines up with skills, subdomain/domain, and program signals.
  - Collaborative filtering: embeddings build from engagement (follows alone do not).
  - ``cf_score`` stays in a bounded range after L2-normalized dot product + scale.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# backend-ai (services.*)
_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# repo root — load features.py directly so we never import hybrid_recommender.__init__
# (that pulls lightgbm/ranking and slows or breaks lightweight test envs).
_ROOT = Path(__file__).resolve().parents[2]
_FEATURES_PATH = _ROOT / "src" / "hybrid_recommender" / "features.py"
_spec = importlib.util.spec_from_file_location("_hybrid_features_cf_tests", _FEATURES_PATH)
assert _spec and _spec.loader
_hybrid_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hybrid_features)
build_cf_embeddings = _hybrid_features.build_cf_embeddings

from services.recommendation_service import (  # noqa: E402
    RecommendationService,
    generate_reason,
    normalize_scores,
)


class TestMatchPercentageNotRawScore:
    """UI should show match_percentage in [60, 100], distinct from internal score."""

    def test_bounds_when_scores_spread(self):
        ranked = pd.DataFrame(
            {
                "score": [0.05, 0.12, 0.88],
                "skill_coverage_score": [0.5, 0.5, 0.5],
            }
        )
        out = normalize_scores(ranked)
        assert out["match_percentage"].between(60, 100).all()
        # Internal scores are small decimals; percentages must not mirror them.
        assert not np.allclose(
            out["match_percentage"].astype(float).values,
            out["score"].astype(float).values * 100.0,
        )

    def test_all_equal_scores_returns_mid_band(self):
        ranked = pd.DataFrame({"score": [0.5, 0.5, 0.5]})
        out = normalize_scores(ranked)
        assert (out["match_percentage"] == 75).all()

    def test_higher_internal_score_higher_percentage_when_coverage_flat(self):
        ranked = pd.DataFrame(
            {
                "score": [0.1, 0.4, 0.9],
            }
        )
        out = normalize_scores(ranked)
        pcts = out["match_percentage"].tolist()
        assert pcts[0] < pcts[1] < pcts[2]

    def test_format_output_preserves_normalized_percentage(self):
        svc = RecommendationService()
        rows = [
            {
                "mentor_id": "a",
                "mentor_name": "Test",
                "domain": "AI",
                "score": 0.7312,
                "pred_score": 0.7312,
                "match_percentage": 91,
                "skill_overlap_score": 0.8,
                "skill_coverage_score": 0.9,
                "subdomain_similarity": 0.5,
                "mentor_domain_match": 1.0,
                "mentor_quality_score": 4.0,
                "mentor_weighted_rating": 4.5,
                "is_following": 0,
                "mentor_covers_all_skills": 1,
            }
        ]
        formatted = svc._format_output(rows)
        assert formatted[0]["match_percentage"] == 91
        assert formatted[0]["score"] == pytest.approx(0.7312, rel=1e-3)

    def test_format_fallback_match_percentage_is_not_raw_score(self):
        """When normalize_scores did not run, _format_output derives % from Layer-A signals only (60–100)."""
        svc = RecommendationService()
        rows = [
            {
                "mentor_id": "b",
                "mentor_name": "M",
                "domain": "AI",
                "pred_score": 0.001,
                "skill_overlap_score": 0.0,
                "skill_coverage_score": 0.0,
                "subdomain_similarity": 0.0,
                "mentor_domain_match": 0.0,
                "mentor_quality_score": 5.0,
                "mentor_weighted_rating": 5.0,
                "is_following": 0,
                "mentor_covers_all_skills": 0,
            }
        ]
        formatted = svc._format_output(rows)
        pct = formatted[0]["match_percentage"]
        assert 60 <= pct <= 100
        assert formatted[0]["score"] == pytest.approx(0.001, rel=1e-6)
        assert abs(pct - formatted[0]["score"] * 100) > 1.0


class TestGenerateReasonConsistency:
    """One-liner reasons must reflect the same signals used for ranking."""

    def test_open_programs_mentioned_when_available(self):
        row = {
            "domain": "Web",
            "average_rating": 4.6,
            "completed_mentorships": 2,
            "open_programs": 2,
            "completed_programs": 0,
            "_domain_match": True,
            "_subdomain_match": False,
            "_skills_overlap": 0.6,
            "_matched_skills": "javascript, react",
            "_covers_all_skills": False,
            "skill_coverage_score": 0.4,
            "_is_followed": False,
        }
        text = generate_reason(row, language="en").lower()
        assert "open program" in text

    def test_completed_programs_when_no_open(self):
        row = {
            "domain": "Data",
            "average_rating": 3.5,
            "completed_mentorships": 1,
            "open_programs": 0,
            "completed_programs": 3,
            "_domain_match": True,
            "_subdomain_match": False,
            "_skills_overlap": 0.0,
            "_matched_skills": "",
            "_covers_all_skills": False,
            "skill_coverage_score": 0.0,
            "_is_followed": False,
        }
        text = generate_reason(row, language="en").lower()
        assert "open program" not in text
        assert "completed program" in text

    def test_covers_all_skills_phrase(self):
        row = {
            "domain": "AI",
            "average_rating": 4.0,
            "completed_mentorships": 0,
            "open_programs": 0,
            "completed_programs": 0,
            "_domain_match": False,
            "_subdomain_match": True,
            "_skills_overlap": 0.2,
            "_matched_skills": "",
            "_covers_all_skills": True,
            "skill_coverage_score": 1.0,
            "_is_followed": False,
        }
        text = generate_reason(row, language="en").lower()
        assert "covers all your skills" in text

    def test_subdomain_and_domain_in_reason(self):
        row = {
            "domain": "Machine Learning",
            "average_rating": 4.2,
            "completed_mentorships": 0,
            "open_programs": 0,
            "completed_programs": 0,
            "_domain_match": True,
            "_subdomain_match": True,
            "_skills_overlap": 0.0,
            "_matched_skills": "",
            "_covers_all_skills": False,
            "skill_coverage_score": 0.0,
            "_is_followed": False,
        }
        text = generate_reason(row, language="en").lower()
        assert "same specialization" in text or "machine learning" in text


def _sklearn_truncated_svd_importable() -> bool:
    """Skip SVD integration checks when sklearn is missing or broken (e.g. corrupt install)."""
    try:
        from sklearn.decomposition import TruncatedSVD  # noqa: F401
    except Exception:
        return False
    return True


class TestCollaborativeFilteringSanity:
    """CF must not activate on follows-only data; SVD path must behave numerically."""

    def test_follows_alone_produce_no_embeddings(self):
        follows = pd.DataFrame(
            {
                "follower_id": [1, 2, 3, 4, 5],
                "following_id": [10, 20, 30, 40, 50],
            }
        )
        emb = build_cf_embeddings(
            interaction_features=pd.DataFrame(),
            follows_hist=follows,
            mentorships_hist=pd.DataFrame(),
            posts_hist=None,
            n_factors=8,
        )
        assert emb["user_factors"] == {}
        assert emb["item_factors"] == {}

    @pytest.mark.skipif(
        not _sklearn_truncated_svd_importable(),
        reason="sklearn TruncatedSVD not importable (install or repair scikit-learn to run SVD CF checks)",
    )
    def test_engagement_matrix_produces_factors_and_bounded_dot_products(self):
        # 5 mentees × 5 mentors = 25 unique pairs → above min interaction threshold
        uid = np.repeat(np.arange(5), 5)
        mid = np.tile(np.arange(5), 5)
        interaction_features = pd.DataFrame(
            {
                "user_id": uid,
                "mentor_id": mid,
                "interaction_score": np.linspace(1.0, 4.0, 25),
            }
        )
        emb = build_cf_embeddings(
            interaction_features=interaction_features,
            follows_hist=pd.DataFrame(),
            mentorships_hist=pd.DataFrame(),
            posts_hist=None,
            n_factors=8,
        )
        assert emb["user_factors"], "expected user_factors from engagement-only matrix"
        assert emb["item_factors"], "expected item_factors from engagement-only matrix"
        # Cosine-like dots on L2-normalized factors ∈ [-1, 1]
        u0 = emb["user_factors"][0]
        m0 = emb["item_factors"][0]
        dot = float(np.dot(u0, m0))
        assert -1.01 <= dot <= 1.01
