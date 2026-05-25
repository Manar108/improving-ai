import logging

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Query, HTTPException

from schemas import ProgramRecommendationResponse
from services.program_recommendation_service import program_recommendation_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/program-recommend", response_model=ProgramRecommendationResponse)
async def recommend_programs(
    user_id: str = Query(..., description="Required: User UUID"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of recommendations (1-50)")
) -> dict:
    """Get program recommendations for a user.
    
    Args:
        user_id: User UUID (required, must be non-empty)
        top_k: Number of recommendations to return (1-50, default 10)
    
    Returns:
        Dictionary with "recommendations" list and optional "error" field
    
    Raises:
        HTTPException 400: If user_id is missing/invalid or top_k is out of range
    """
    # CRITICAL FIX: Validate user_id is provided and non-empty
    if not user_id or not user_id.strip():
        logger.warning("Program recommendation endpoint: user_id is required and cannot be empty")
        raise HTTPException(status_code=400, detail="user_id is required and cannot be empty")
    
    user_id = user_id.strip()
    
    # Validate top_k is in range (FastAPI ge/le should handle, but be explicit)
    if top_k < 1 or top_k > 50:
        logger.warning("Program recommendation endpoint: top_k out of range: %d", top_k)
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    
    try:
        recommendations = await program_recommendation_service.get_recommendations(user_id=user_id, top_k=top_k)
        return {"recommendations": recommendations, "error": ""}
    except Exception as exc:
        logger.exception("Program recommendation endpoint failed for user_id=%s, top_k=%d: %s", user_id, top_k, exc)
        return {"recommendations": [], "error": "Failed to load program recommendations. Please try again later."}