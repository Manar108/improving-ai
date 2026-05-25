from typing import Literal, List, Dict, Any, Optional

from pydantic import BaseModel, Field


IntentType = Literal[
    "greeting",
    "find_mentor",
    "ask_mentor_recommendation",
    "ask_program_recommendation",
    "recommendation_explanation",
    "task_help",
    "submit_task",
    "roadmap_request",
    "materials_request",
    "faq",
    "complaint",
    "support_request",
    "general_question",
    "off_topic",
    "mentor_analytics",      # Mentor-specific: view analytics/stats
    "mentor_workflow_help",  # Mentor-specific: workflow/communication help
]
ResponseType = Literal["recommendation", "materials", "stats", "roadmap", "text"]


# ─────────────────────────────────────────────────────────────────
# Explanation Metadata — Rich explanation data for frontend/chatbot
# ─────────────────────────────────────────────────────────────────

class ExplanationMetadata(BaseModel):
    """Structured explanation metadata for rendering rich recommendations.
    
    Provides chatbot and frontend with detailed signal breakdown to explain
    why a mentor/program was recommended.
    
    May 2026 Updates:
    - Added compatibility_fit_band: describes program/mentor fit quality
    - Added level_gaps: explains level mismatches
    - Added confidence_band: indicates recommendation strength
    """
    matched_skills: List[str] = Field(default_factory=list, description="Skills mentor/program has that match mentee needs")
    matched_subdomains: List[str] = Field(default_factory=list, description="Subdomain specializations")
    domain_alignment: str = Field(default="unknown", description="'exact', 'related', or 'different'")
    quality_contribution: float = Field(default=0.0, description="Quality signal strength (0-1)")
    popularity_contribution: float = Field(default=0.0, description="Popularity signal strength (0-1)")
    cf_contribution: float = Field(default=0.0, description="Collaborative filtering strength (0-1)")
    reranking_adjustments: str = Field(default="", description="Why score was adjusted (e.g. 'strong_skill_match')")
    confidence_score: float = Field(default=0.0, description="Overall confidence (0-1)")
    reason: str = Field(default="", description="Natural language reason for recommendation")
    
    # NEW (May 2026): Compatibility and fit quality information
    compatibility_fit_band: str = Field(
        default="unknown",
        description="Fit quality: 'exact_fit' (perfect), 'near_fit' (good), 'stretch_fit' (challenging), or 'weak_fit' (exploratory)"
    )
    target_level_gap: Optional[int] = Field(default=None, description="Experience level difference (negative=mentee under-leveled)")
    education_level_gap: Optional[int] = Field(default=None, description="Education level difference (negative=mentee under-qualified)")
    missing_required_skills: Optional[int] = Field(default=None, description="Count of required skills mentee doesn't have")
    overall_softness: Optional[float] = Field(default=None, description="Combined compatibility score (0-1)")



# ─────────────────────────────────────────────────────────────────
# Mentor Recommendation Schema
# ─────────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    mentor_id: str
    mentor_name: str
    domain: str
    score: float
    match_percentage: int = 75
    reason: str
    # CRITICAL FIX: Add explanation_metadata for chatbot/frontend rendering
    explanation_metadata: Optional[ExplanationMetadata] = None


# ─────────────────────────────────────────────────────────────────
# Program Recommendation Schema
# ─────────────────────────────────────────────────────────────────

class ProgramRecommendationItem(BaseModel):
    post_id: str
    mentor_id: str
    mentor_name: str = ""
    title: str = ""
    domain: str = ""
    target_level: str = ""
    education_level: str = ""
    score: float
    match_percentage: int = Field(default=75, ge=0, le=100, description="Match % (0-100)")
    # CRITICAL FIX: Add explanation metadata for rich explanations
    explanation_metadata: Optional[ExplanationMetadata] = None
    reason: str = ""
    minimum_requirement_exact_match: bool = False
    minimum_requirement_above_minimum: bool = False
    target_level_pass: bool = False
    education_level_pass: bool = False
    availability_pass: bool = False


class ProgramRecommendationResponse(BaseModel):
    recommendations: list[ProgramRecommendationItem] = Field(default_factory=list)
    error: str = ""


class MaterialItem(BaseModel):
    title: str
    url: str
    kind: Literal["docs", "courses", "videos", "article", "articles", "projects"] = "article"
    source: str = ""
    reason: str = ""


# --- Smart Search response models ---

class SearchResultItem(BaseModel):
    """A single search result returned by the smart search pipeline."""
    title: str
    link: str
    source: str = ""
    reason: str = ""


class MaterialsSearchResponse(BaseModel):
    """Full response from the smart search pipeline."""
    success: bool = True
    intent: str = "general_search"
    topic: str = ""
    results: list[SearchResultItem] = Field(default_factory=list)
    summary: str = ""


class StatCard(BaseModel):
    label: str
    value: str


class ChatRequest(BaseModel):
    user_id: str = Field(default="")
    message: str
    language: Literal["ar", "en"] | None = None
    history: list[dict] = Field(default_factory=list, description="Optional conversation history as list of {role, content}")


class ChatResponse(BaseModel):
    language: Literal["ar", "en"]
    intent: IntentType
    response_type: ResponseType
    answer: str
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    materials: list[MaterialItem] = Field(default_factory=list)
    stats: list[StatCard] = Field(default_factory=list)


# --- Sentiment Analysis models ---

class SentimentRequest(BaseModel):
    """Request body for POST /sentiment/predict."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyse")


class SentimentResponse(BaseModel):
    """Response from sentiment prediction."""
    label: str = Field(..., description="Predicted sentiment: negative, neutral, or positive")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0–1)")
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Probability for each sentiment class",
    )


class SentimentBatchRequest(BaseModel):
    """Request body for POST /sentiment/predict-batch."""
    texts: list[str] = Field(..., min_length=1, max_length=32, description="Texts to analyse")


class SentimentBatchResponse(BaseModel):
    """Response from batch sentiment prediction."""
    results: list[SentimentResponse] = Field(default_factory=list)
    count: int = 0


# --- Mentor Feedback Summary models ---

class FeedbackBreakdown(BaseModel):
    """Sentiment distribution breakdown."""
    positive: int = 0
    neutral: int = 0
    negative: int = 0
    total: int = 0


class MentorFeedbackSummaryResponse(BaseModel):
    """Response from mentor feedback summary endpoint.

    Provides satisfaction rate, sentiment breakdown, and an
    AI-generated summary sentence from the mentor's feedbacks.
    """
    mentor_id: str
    mentor_name: str = ""
    satisfaction_rate: float = Field(0.0, ge=0, le=100, description="Percentage of positive feedbacks (0-100)")
    average_rating: float = Field(0.0, ge=0, le=5, description="Average star rating (0-5)")
    breakdown: FeedbackBreakdown = Field(default_factory=FeedbackBreakdown)
    summary: str = Field("", description="AI-generated summary sentence from feedbacks")
    top_positive_themes: list[str] = Field(default_factory=list, description="Short positive attributes (2-3 words each), e.g. 'شرح واضح', 'تواصل ممتاز'")
    top_negative_themes: list[str] = Field(default_factory=list, description="Short negative attributes (2-3 words each), e.g. 'تأخر في الرد', 'غير منظم'")
