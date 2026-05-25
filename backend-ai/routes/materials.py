# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Query

from schemas import MaterialsSearchResponse
from services.search_service import search_service


router = APIRouter()


@router.get("/materials", response_model=MaterialsSearchResponse)
async def materials(q: str, language: str = Query(default="en")) -> dict:
    """Smart materials search — LLM understanding → Google CSE → LLM ranking."""
    return await search_service.find_materials(query=q, language=language)
