import logging

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Query, HTTPException

from schemas import RecommendationItem
from services.recommendation_service import recommendation_service

# ✅ ADDED: Import error handler
from services.error_handling import RecommendationErrorHandler

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/recommend")
async def recommend(user_id: str = Query(..., description="Required: User UUID")) -> dict:
    """Get mentor recommendations for a user.
    
    Args:
        user_id: User UUID (required, must be non-empty)
    
    Returns:
        Dictionary with "recommendations" list and optional "error" field
    
    Raises:
        HTTPException 400: If user_id is missing or invalid
    
    ✅ ADDED: Enhanced error handling with fallback chain
    """
    # CRITICAL FIX: Validate user_id is provided and non-empty
    if not user_id or not user_id.strip():
        logger.warning("Recommendation endpoint: user_id is required and cannot be empty")
        raise HTTPException(status_code=400, detail="user_id is required and cannot be empty")
    
    # Additional validation: user_id should look like a UUID (basic check)
    user_id = user_id.strip()
    if len(user_id) < 8:  # UUIDs are at least 36 chars, but be lenient
        logger.warning("Recommendation endpoint: user_id format suspicious: %s", user_id[:8])
        raise HTTPException(status_code=400, detail="user_id format appears invalid")
    
    try:
        # Try to get recommendations with error handling built in
        recommendations = await recommendation_service.get_recommendations(user_id=user_id)
        
        # ✅ ADDED: If we got recommendations, return them even if from fallback
        if recommendations:
            return {
                "recommendations": [RecommendationItem(**item).model_dump() for item in recommendations],
                "status": "success"
            }
        
        # ✅ ADDED: Empty results - return database fallback
        logger.warning(f"Empty recommendations for user {user_id}, trying database fallback")
        fallback_recs = RecommendationErrorHandler.get_database_fallback_recommendations(user_id)
        if fallback_recs:
            return {
                "recommendations": [RecommendationItem(**item).model_dump() for item in fallback_recs],
                "status": "success_fallback"
            }
        
        # ✅ ADDED: Complete failure - return error with some generic options
        logger.error(f"Complete recommendation failure for user {user_id}")
        return {
            "recommendations": [],
            "error": "Failed to load personalized recommendations. Please try again in a few moments.",
            "status": "error"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (input validation)
        raise
    except Exception as exc:
        logger.exception(f"Recommendation endpoint failed for user_id={user_id}: {exc}")
        # ✅ ADDED: Better error handling - try fallback before giving up
        try:
            fallback_recs = RecommendationErrorHandler.get_database_fallback_recommendations(user_id)
            if fallback_recs:
                return {
                    "recommendations": [RecommendationItem(**item).model_dump() for item in fallback_recs],
                    "status": "error_with_fallback"
                }
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
        
        return {
            "recommendations": [],
            "error": "Failed to load recommendations. Please try again later.",
            "status": "error"
        }
