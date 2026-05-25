"""Tests for Conversational Recommendation Explainability.

Tests cover:
  1. Intent classification — recommendation_explanation is detected correctly
  2. Mentor name matching — fuzzy match from user messages
  3. Recommendation memory — store/retrieve with TTL
  4. Detailed explanation generation — bullet-point formatting
  5. Fallback — no previous recommendations
  6. Non-regression — existing intents are NOT broken
"""

import time
# pyrefly: ignore [missing-import]
import pytest

# ── Intent classification tests ──
from services.intent_service import _keyword_fallback, _extract_intent


class TestIntentClassification:
    """Verify recommendation_explanation intent is detected correctly."""

    # Arabic follow-up questions
    @pytest.mark.parametrize("msg", [
        "ليه رشحتلي المنتور ده؟",
        "ليه رشحتلي أحمد؟",
        "ليه ده أحسن mentor ليا؟",
        "إيه اللي خلاه مناسب ليا؟",
        "ايه اللي خلاه مناسب",
        "ليه المنتور ده مش التاني؟",
        "سبب الترشيح",
        "أسباب الترشيح",
        "ليه ده مش محمد",
    ])
    def test_arabic_explanation_questions(self, msg):
        assert _keyword_fallback(msg) == "recommendation_explanation"

    # English follow-up questions
    @pytest.mark.parametrize("msg", [
        "why did you recommend this mentor?",
        "why this mentor?",
        "why is this the best match for me?",
        "explain the recommendation",
        "what made this mentor suitable?",
        "why not the other one?",
    ])
    def test_english_explanation_questions(self, msg):
        assert _keyword_fallback(msg) == "recommendation_explanation"

    # Non-regression: these should NOT be recommendation_explanation
    @pytest.mark.parametrize("msg,expected", [
        ("رشحلي مرشدين", "ask_mentor_recommendation"),
        ("recommend mentors for me", "ask_mentor_recommendation"),
        ("ايه machine learning", "general_question"),
        ("شرحلي OOP", "general_question"),
        ("hi", "greeting"),
        ("مين أحسن mentor في AI", "find_mentor"),
    ])
    def test_non_regression(self, msg, expected):
        assert _keyword_fallback(msg) == expected

    def test_llm_parser_accepts_new_intent(self):
        """Verify _extract_intent can parse the new intent from LLM output."""
        raw = '{"intent": "recommendation_explanation"}'
        assert _extract_intent(raw) == "recommendation_explanation"


# ── Recommendation memory & explanation tests ──

from services.recommendation_service import (
    RecommendationService,
    generate_detailed_explanation,
    _recommendation_memory,
)

# Sample recommendation data (matches what _format_output produces)
_SAMPLE_RECOMMENDATIONS = [
    {
        "mentor_id": "uuid-1",
        "mentor_name": "Ahmed Hassan",
        "domain": "AI",
        "score": 0.92,
        "match_percentage": 95,
        "reason": "Strong match in python, machine learning with same specialization in AI, 2 open programs, and excellent rating (4.8⭐).",
    },
    {
        "mentor_id": "uuid-2",
        "mentor_name": "Sara Mohamed",
        "domain": "Web Development",
        "score": 0.85,
        "match_percentage": 88,
        "reason": "Shared expertise in javascript, react with Web Development domain match, and high rating (4.5⭐).",
    },
    {
        "mentor_id": "uuid-3",
        "mentor_name": "Mohamed Ali",
        "domain": "Data Science",
        "score": 0.78,
        "match_percentage": 82,
        "reason": "Covers 75% of your skills with matching specialization, 5 completed mentorships.",
    },
]


class TestRecommendationMemory:
    """Test the per-user recommendation memory cache."""

    def setup_method(self):
        _recommendation_memory.clear()
        self.service = RecommendationService()

    def test_store_and_retrieve(self):
        _recommendation_memory.set("user-123", _SAMPLE_RECOMMENDATIONS)
        result = self.service.get_last_recommendations("user-123")
        assert result is not None
        assert len(result) == 3
        assert result[0]["mentor_name"] == "Ahmed Hassan"

    def test_no_recommendations_returns_none(self):
        result = self.service.get_last_recommendations("user-nonexistent")
        assert result is None


class TestMentorNameMatching:
    """Test fuzzy matching of mentor names from user messages."""

    def setup_method(self):
        self.service = RecommendationService()

    def test_exact_name_match(self):
        result = self.service._match_mentor_from_message(
            "ليه رشحتلي Ahmed Hassan؟", _SAMPLE_RECOMMENDATIONS
        )
        assert result is not None
        assert result["mentor_name"] == "Ahmed Hassan"

    def test_partial_name_match(self):
        result = self.service._match_mentor_from_message(
            "ليه رشحتلي Sara؟", _SAMPLE_RECOMMENDATIONS
        )
        assert result is not None
        assert result["mentor_name"] == "Sara Mohamed"

    def test_arabic_partial_name(self):
        # Test with Arabic text + English name token
        result = self.service._match_mentor_from_message(
            "ليه Mohamed أحسن؟", _SAMPLE_RECOMMENDATIONS
        )
        assert result is not None
        # Should match "Sara Mohamed" first (Strategy 2, iterates in order)
        # or "Mohamed Ali" — either is a valid match
        assert "Mohamed" in result["mentor_name"]

    def test_ordinal_first(self):
        result = self.service._match_mentor_from_message(
            "ليه الأول ده؟", _SAMPLE_RECOMMENDATIONS
        )
        assert result is not None
        assert result["mentor_name"] == "Ahmed Hassan"

    def test_ordinal_second_arabic(self):
        result = self.service._match_mentor_from_message(
            "ليه التاني ده؟", _SAMPLE_RECOMMENDATIONS
        )
        assert result is not None
        assert result["mentor_name"] == "Sara Mohamed"

    def test_ordinal_third_english(self):
        result = self.service._match_mentor_from_message(
            "why the third one?", _SAMPLE_RECOMMENDATIONS
        )
        assert result is not None
        assert result["mentor_name"] == "Mohamed Ali"

    def test_no_match_returns_none(self):
        result = self.service._match_mentor_from_message(
            "ليه رشحتلي ده؟", _SAMPLE_RECOMMENDATIONS
        )
        assert result is None  # No name or ordinal → caller defaults to #1


class TestDetailedExplanation:
    """Test the detailed multi-line explanation generator."""

    def test_english_explanation_has_header(self):
        explanation = generate_detailed_explanation(_SAMPLE_RECOMMENDATIONS[0], "en")
        assert "Ahmed Hassan" in explanation
        assert "95%" in explanation
        assert "because:" in explanation

    def test_arabic_explanation_has_header(self):
        explanation = generate_detailed_explanation(_SAMPLE_RECOMMENDATIONS[0], "ar")
        assert "Ahmed Hassan" in explanation
        assert "95%" in explanation
        assert "للأسباب دي" in explanation

    def test_skill_bullet_extracted(self):
        explanation = generate_detailed_explanation(_SAMPLE_RECOMMENDATIONS[0], "en")
        # The reason contains "Strong match in python, machine learning"
        assert "skill" in explanation.lower() or "match" in explanation.lower()

    def test_rating_bullet_extracted(self):
        explanation = generate_detailed_explanation(_SAMPLE_RECOMMENDATIONS[0], "en")
        assert "rated" in explanation.lower() or "rating" in explanation.lower() or "⭐" in explanation

    def test_availability_bullet_extracted(self):
        explanation = generate_detailed_explanation(_SAMPLE_RECOMMENDATIONS[0], "en")
        assert "open program" in explanation.lower() or "ready" in explanation.lower()

    def test_domain_bullet_extracted(self):
        explanation = generate_detailed_explanation(_SAMPLE_RECOMMENDATIONS[0], "en")
        assert "AI" in explanation or "specialization" in explanation.lower()

    def test_experience_bullet_extracted(self):
        explanation = generate_detailed_explanation(_SAMPLE_RECOMMENDATIONS[2], "en")
        # reason has "5 completed mentorships"
        assert "mentorship" in explanation.lower() or "track record" in explanation.lower()

    def test_multiline_output(self):
        explanation = generate_detailed_explanation(_SAMPLE_RECOMMENDATIONS[0], "en")
        lines = explanation.strip().split("\n")
        # Should have header + at least 2 bullet lines
        assert len(lines) >= 3, f"Expected 3+ lines, got {len(lines)}: {explanation}"


class TestExplainRecommendationEndToEnd:
    """End-to-end test of explain_recommendation method."""

    def setup_method(self):
        _recommendation_memory.clear()
        self.service = RecommendationService()

    def test_no_previous_recommendations_arabic(self):
        result = self.service.explain_recommendation("user-new", "ليه رشحتلي ده", "ar")
        assert "رشحلي مرشدين" in result or "ترشيحات" in result

    def test_no_previous_recommendations_english(self):
        result = self.service.explain_recommendation("user-new", "why this mentor", "en")
        assert "recommend" in result.lower()

    def test_with_cached_recommendations(self):
        _recommendation_memory.set("user-456", _SAMPLE_RECOMMENDATIONS)
        result = self.service.explain_recommendation("user-456", "ليه رشحتلي Ahmed", "en")
        assert "Ahmed Hassan" in result
        assert "95%" in result

    def test_defaults_to_top_when_no_name(self):
        _recommendation_memory.set("user-789", _SAMPLE_RECOMMENDATIONS)
        result = self.service.explain_recommendation("user-789", "ليه رشحتلي ده؟", "en")
        # Should default to first recommendation (Ahmed Hassan)
        assert "Ahmed Hassan" in result

    def test_arabic_response(self):
        _recommendation_memory.set("user-ar", _SAMPLE_RECOMMENDATIONS)
        result = self.service.explain_recommendation("user-ar", "ليه رشحتلي Ahmed", "ar")
        assert "Ahmed Hassan" in result
        assert "تم ترشيح" in result
        assert "للأسباب دي" in result
