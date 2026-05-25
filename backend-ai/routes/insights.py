# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Query

from services.analytics_service import analytics_service


router = APIRouter()


@router.get("/insights")
def insights(language: str = Query(default="en")) -> dict:
    return {"insights": analytics_service.get_stats(language=language)}
