"""Mentor Analytics Routes — API endpoints for mentor statistics and analytics.

Provides clean APIs for mentor dashboard/UI to fetch:
- Overall mentor stats
- Program statistics
- Mentee activity
- Application metrics
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel

from services.mentor_context_service import mentor_context_service

logger = logging.getLogger(__name__)
router = APIRouter()


# ────────────────────────────────────────────────────────────────────
# Response models
# ────────────────────────────────────────────────────────────────────

class MentorProfileDto(BaseModel):
    """Mentor profile overview."""
    user_id: str
    first_name: str
    last_name: str
    domain_name: str
    years_of_experience: int
    is_verified: bool
    average_rating: float
    total_reviews: int
    program_count: int


class ProgramStatsDto(BaseModel):
    """Program statistics."""
    program_id: int
    title: str
    domain_name: str
    capacity: int
    active_mentees: int
    applications_count: int
    created_at: str


class MenteeActivityDto(BaseModel):
    """Mentee activity info."""
    mentee_id: str
    first_name: str
    last_name: str
    domain_name: str
    start_date: str
    status: str


class AnalyticsOverviewDto(BaseModel):
    """High-level analytics overview."""
    mentor_profile: MentorProfileDto
    programs: list[ProgramStatsDto]
    active_mentees_count: int
    pending_applications_count: int
    average_mentees_per_program: float


# ────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────

@router.get("/mentor/analytics/profile/{mentor_id}", response_model=MentorProfileDto)
async def get_mentor_profile(mentor_id: str = Path(...)) -> MentorProfileDto:
    """Get mentor profile information."""
    ctx = mentor_context_service.get_mentor_context(mentor_id)
    
    if not ctx.get("user_id"):
        raise HTTPException(status_code=404, detail="Mentor not found")
    
    return MentorProfileDto(
        user_id=ctx.get("user_id", ""),
        first_name=ctx.get("first_name", ""),
        last_name=ctx.get("last_name", ""),
        domain_name=ctx.get("domain_name", ""),
        years_of_experience=ctx.get("years_of_experience", 0),
        is_verified=ctx.get("is_verified", False),
        average_rating=ctx.get("average_rating", 0.0),
        total_reviews=ctx.get("total_reviews", 0),
        program_count=ctx.get("program_count", 0),
    )


@router.get("/mentor/analytics/programs/{mentor_id}", response_model=list[ProgramStatsDto])
async def get_mentor_programs(
    mentor_id: str = Path(...),
    limit: int = Query(10, ge=1, le=50),
) -> list[ProgramStatsDto]:
    """Get mentor's programs with statistics."""
    programs = mentor_context_service.get_mentor_programs(mentor_id, limit=limit)
    
    return [
        ProgramStatsDto(
            program_id=p.get("program_id", 0),
            title=p.get("title", ""),
            domain_name=p.get("domain_name", ""),
            capacity=p.get("capacity", 0),
            active_mentees=p.get("active_mentees", 0),
            applications_count=p.get("applications_count", 0),
            created_at=p.get("created_at", ""),
        )
        for p in programs
    ]


@router.get("/mentor/analytics/mentees/{mentor_id}", response_model=list[MenteeActivityDto])
async def get_mentor_active_mentees(
    mentor_id: str = Path(...),
    limit: int = Query(20, ge=1, le=100),
) -> list[MenteeActivityDto]:
    """Get mentor's active mentees."""
    mentees = mentor_context_service.get_mentor_active_mentees(mentor_id, limit=limit)
    
    return [
        MenteeActivityDto(
            mentee_id=m.get("mentee_id", ""),
            first_name=m.get("first_name", ""),
            last_name=m.get("last_name", ""),
            domain_name=m.get("domain_name", ""),
            start_date=m.get("start_date", ""),
            status=m.get("status", ""),
        )
        for m in mentees
    ]


@router.get("/mentor/analytics/applications/{mentor_id}", response_model=list[dict])
async def get_mentor_pending_applications(
    mentor_id: str = Path(...),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict]:
    """Get pending applications for mentor's programs."""
    applications = mentor_context_service.get_mentor_pending_applications(mentor_id, limit=limit)
    return applications


@router.get("/mentor/analytics/overview/{mentor_id}", response_model=AnalyticsOverviewDto)
async def get_analytics_overview(mentor_id: str = Path(...)) -> AnalyticsOverviewDto:
    """Get complete analytics overview for mentor dashboard."""
    # Load all data
    profile = mentor_context_service.get_mentor_context(mentor_id)
    programs = mentor_context_service.get_mentor_programs(mentor_id, limit=100)
    mentees = mentor_context_service.get_mentor_active_mentees(mentor_id, limit=100)
    applications = mentor_context_service.get_mentor_pending_applications(mentor_id, limit=100)

    if not profile.get("user_id"):
        raise HTTPException(status_code=404, detail="Mentor not found")

    # Calculate metrics
    active_mentees_count = len(mentees)
    pending_applications_count = len(applications)
    program_count = len(programs)
    avg_mentees_per_program = (
        active_mentees_count / program_count if program_count > 0 else 0
    )

    return AnalyticsOverviewDto(
        mentor_profile=MentorProfileDto(
            user_id=profile.get("user_id", ""),
            first_name=profile.get("first_name", ""),
            last_name=profile.get("last_name", ""),
            domain_name=profile.get("domain_name", ""),
            years_of_experience=profile.get("years_of_experience", 0),
            is_verified=profile.get("is_verified", False),
            average_rating=profile.get("average_rating", 0.0),
            total_reviews=profile.get("total_reviews", 0),
            program_count=profile.get("program_count", 0),
        ),
        programs=[
            ProgramStatsDto(
                program_id=p.get("program_id", 0),
                title=p.get("title", ""),
                domain_name=p.get("domain_name", ""),
                capacity=p.get("capacity", 0),
                active_mentees=p.get("active_mentees", 0),
                applications_count=p.get("applications_count", 0),
                created_at=p.get("created_at", ""),
            )
            for p in programs
        ],
        active_mentees_count=active_mentees_count,
        pending_applications_count=pending_applications_count,
        average_mentees_per_program=avg_mentees_per_program,
    )
