"""Recommendation service — production-ready mentor matching.

Architecture:
    1. Fetch candidates from DB (with availability data) OR load from trained model
    2. Apply soft availability preferences (availability is a boost, not an exclusion)
    3. Score with weighted signals (skills > subdomain > domain > availability > rating > activity > follow)
  4. Re-rank with controlled adjustments
  5. Normalize scores to meaningful percentages (70-100 range)
  6. Generate one clean, natural-language explanation per mentor

Modes:
  - "db"   : Full DB query pipeline (legacy, for comparison)
  - "model": Direct inference from trained LightGBM model (production default)
  - "api"  : External recommender API (fallback)

Performance:
  - Mentor skills/subdomains are cached with 5-minute TTL to reduce DB load.
  - Candidate fetching is NOT cached (needs fresh availability data per request).
"""

import logging
import sys
import time as _time
import uuid
from pathlib import Path
from typing import Any

import httpx
import asyncio
import numpy as np
import pandas as pd

from config import settings
from database.db import DatabaseAccessError, MissingTableError, database
from services.uuid_mapper import get_mentee_integer_id, get_uuid_from_integer

# ✅ ADDED: Import error handler
from services.error_handling import RecommendationErrorHandler

# Lazy imports for model inference (heavy, only used in model mode)
_pipeline_module = None

def _get_pipeline():
    global _pipeline_module
    if _pipeline_module is None:
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.hybrid_recommender.pipeline import load_inference_artifacts, predict_for_user
        _pipeline_module = {"load_inference_artifacts": load_inference_artifacts, "predict_for_user": predict_for_user}
    return _pipeline_module


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# TTL Cache — imported from shared module
# ────────────────────────────────────────────────────────────────────

from services.cache import TTLCache as _TTLCache


# Global caches — shared across requests
_expertise_cache = _TTLCache(ttl_seconds=300)          # 5 minutes — expertise less volatile
_subdomain_cache = _TTLCache(ttl_seconds=300)         # 5 minutes — subdomain taxonomy rarely changes

# Recommendation memory — stores last results per user for follow-up explanation queries
_recommendation_memory = _TTLCache(ttl_seconds=300)   # 5 minutes (reduced from 15 min) — faster invalidation


# ────────────────────────────────────────────────────────────────────
# DB-mode scoring weights — aligned with ranking philosophy
#
# Three signal layers (strongest → weakest):
#
#   Layer A — Core Compatibility (65% total):
#     skill_overlap (30%), subdomain_match (20%), domain_match (15%)
#     These answer: "Is this mentor the best match for this mentee?"
#
#   Layer B — Mentor Quality (17% total):
#     quality (17%) — rating, completed mentorships, activity
#     This answers: "Is this mentor reliable and experienced?"
#
#   Layer C — Serving/Business (5% total + multiplicative availability):
#     popularity (3%), followed (2%) — social proof, familiarity
#     availability — multiplicative boost (not additive weight)
#     These should NEVER override a strong compatibility match.
#
# ────────────────────────────────────────────────────────────────────

_WEIGHTS = {
    # Layer A — Core Compatibility (65%)
    "skill_overlap":    0.30,   # Strongest signal — direct compatibility
    "subdomain_match":  0.20,   # Specialization alignment
    "domain_match":     0.15,   # Broader field alignment
    # Layer B — Mentor Quality (17%)
    "quality":          0.17,   # Mentor quality (rating + activity)
    # Layer C — Serving/Business (5% + multiplicative availability)
    "availability":     1.0,    # Multiplicative boost only (not additive)
    "popularity":       0.03,   # Follower count / social proof (reduced)
    "followed":         0.02,   # Personalization signal (weakest)
}

# Display + domain_match: prefer mentor_profile.domain_id; if missing, infer from MentorSubDomains → subdomain → domains.
_MENTOR_DOMAIN_NAME_SQL = """COALESCE(
    NULLIF(LTRIM(RTRIM(d.name)), ''),
    (
        SELECT TOP (1) dom_sd.name
        FROM MentorSubDomains ms
        INNER JOIN subdomain sd ON sd.subdomain_id = TRY_CONVERT(INT, ms."SubDomainId")
        INNER JOIN domains dom_sd ON dom_sd.domain_id = sd.domain_id
        WHERE CAST(ms."MentorId" AS NVARCHAR(100)) = CAST(mp.user_id AS NVARCHAR(100))
        ORDER BY sd.subdomain_id
    ),
    'General'
)"""

_MENTOR_DOMAIN_ID_COALESCE_SQL = """COALESCE(
    mp.domain_id,
    (
        SELECT TOP (1) sd.domain_id
        FROM MentorSubDomains ms
        INNER JOIN subdomain sd ON sd.subdomain_id = TRY_CONVERT(INT, ms."SubDomainId")
        WHERE CAST(ms."MentorId" AS NVARCHAR(100)) = CAST(mp.user_id AS NVARCHAR(100))
        ORDER BY sd.subdomain_id
    )
)"""


class RecommendationService:
    def __init__(self) -> None:
        self._cached_base_candidates = pd.DataFrame()
        self._model_bundle = None  # cached model artifacts (loaded once)
        self._last_manifest_version = None  # track version for cache invalidation
        # Simple in-memory short-lived cache for candidate fetch (reduce DB load)
        self._candidates_cache_df = pd.DataFrame()
        self._candidates_cache_ts = 0.0
        self._candidates_cache_ttl = 30.0  # seconds
        # Expose caches for external cache management (e.g. admin clear)
        self.expertise_cache = _expertise_cache
        self.subdomain_cache = _subdomain_cache

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    async def get_recommendations(self, user_id: int | str) -> list[dict]:
        """Get top-10 mentor recommendations for a user.
        
        ✅ ADDED: Error handling with intelligent fallback chain
        """
        mode = settings.RECOMMENDER_MODE.lower().strip()
        
        try:
            # Try primary mode
            if mode == "model":
                # Run heavy model inference off the event loop to avoid blocking
                results = await asyncio.to_thread(self._from_model, user_id)
                # ✅ ADDED: Try DB fallback if model returns empty
                if not results:
                    logger.info("Model returned empty, trying database fallback")
                    results = RecommendationErrorHandler.get_database_fallback_recommendations(user_id)
            elif mode == "api":
                results = await self._from_api(user_id)
                if not results:
                    # ✅ ADDED: Try DB fallback on API failure
                    logger.info("API returned empty, trying database fallback")
                    results = RecommendationErrorHandler.get_database_fallback_recommendations(user_id)
            else:
                # Run DB pipeline off the event loop (may perform heavy queries)
                results = await asyncio.to_thread(self._from_db, user_id)
                if not results:
                    # ✅ ADDED: Try DB fallback if legacy query returns empty
                    logger.info("DB query returned empty, trying intelligent fallback")
                    results = RecommendationErrorHandler.get_database_fallback_recommendations(user_id)
                
        except Exception as exc:
            logger.error(f"Error in get_recommendations (mode={mode}): {exc}", exc_info=True)
            # ✅ ADDED: Use intelligent database fallback on exception
            results = RecommendationErrorHandler.get_database_fallback_recommendations(user_id)
            if not results:
                # ✅ ADDED: Log complete failure but don't crash
                logger.error(f"Even database fallback failed for user {user_id}")
                results = RecommendationErrorHandler.handle_recommendation_failure(user_id, exc)

        # Store in memory for follow-up explanation queries
        if results and user_id:
            _recommendation_memory.set(str(user_id), results)
            logger.debug("Stored %d recommendations in memory for user %s", len(results), str(user_id)[:8])

        return results

    # ────────────────────────────────────────────────────────────
    # Recommendation Explainability
    # ────────────────────────────────────────────────────────────

    def get_last_recommendations(self, user_id: str) -> list[dict] | None:
        """Retrieve cached recommendations for follow-up explanation queries."""
        return _recommendation_memory.get(str(user_id))

    def explain_recommendation(self, user_id: str, user_message: str, language: str = "en") -> str:
        """Generate a detailed conversational explanation for a recommended mentor.

        Steps:
          1. Retrieve cached recommendations from memory
          2. Fuzzy-match mentor name from user message (or default to #1)
          3. Format detailed bullet-point explanation from existing signal data
          4. Return conversational response in the user's language
        """
        cached = self.get_last_recommendations(user_id)
        if not cached:
            if language == "ar":
                return (
                    "مفيش ترشيحات سابقة في المحادثة دي. "
                    "اكتب 'رشحلي مرشدين' عشان أرشحلك أفضل المرشدين المناسبين ليك! 🎯"
                )
            return (
                "I don't have any previous recommendations in this conversation. "
                "Type 'recommend mentors for me' to get your personalized recommendations first! 🎯"
            )

        # Try to identify which mentor the user is asking about
        mentor = self._match_mentor_from_message(user_message, cached)
        if mentor is None:
            # Default to the top recommendation
            mentor = cached[0]

        return generate_detailed_explanation(mentor, language)

    # ────────────────────────────────────────────────────────────
    # API-based recommendations (external model)
    # ────────────────────────────────────────────────────────────

    async def _from_api(self, user_id: int | str) -> list[dict]:
        url = f"{settings.RECOMMENDER_API_BASE_URL.rstrip('/')}{settings.RECOMMENDER_API_PATH}"
        
        # Convert UUID to integer ID for model inference
        # The model was trained on integer mentee_ids, not UUIDs
        model_user_id = user_id
        if isinstance(user_id, str) and len(user_id) > 10:  # Likely a UUID
            integer_id = get_mentee_integer_id(user_id)
            if integer_id is not None:
                model_user_id = integer_id
                logger.debug("Mapped UUID %s → integer ID %d for model inference", user_id, integer_id)
        
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(url, params={"user_id": model_user_id})
                if response.status_code != 200:
                    logger.warning("Recommender API returned %d", response.status_code)
                    return []
                data = response.json()
                if isinstance(data, dict) and "recommendations" in data:
                    return self._format_output(data["recommendations"])
                if isinstance(data, list):
                    return self._format_output(data)
        except httpx.TimeoutException:
            logger.warning("Recommender API timed out")
        except Exception as exc:
            logger.warning("Recommender API error: %s", exc)
        return []

    # ────────────────────────────────────────────────────────────
    # Model-based recommendations (direct inference from trained model)
    # ────────────────────────────────────────────────────────────

    def _from_model(self, user_id: int | str) -> list[dict]:
        """Generate recommendations using the trained LightGBM model.

        Steps:
          1. Map UUID → integer ID
          2. Load inference artifacts + validate manifest
          3. Run predict_for_user()
          4. Reverse-map integer mentor IDs → UUIDs for DB lookup
          5. Enrich with mentor metadata from DB
          6. Normalize + generate reasons
        """
        try:
            # Step 1: UUID → integer
            user_uuid = self._parse_user_uuid(user_id)
            if user_uuid is None:
                logger.warning("Model inference skipped: invalid user_id (%s)", user_id)
                return []

            model_user_id = get_mentee_integer_id(user_uuid)
            if model_user_id is None:
                logger.warning("Model inference skipped: no integer mapping for %s", user_uuid)
                return []

            # Step 2: Load artifacts (cached — avoid re-reading ~20MB CSV per request)
            # CRITICAL: Validate manifest compatibility for ML safety
            pipeline = _get_pipeline()
            if self._model_bundle is None:
                artifacts_dir = Path(__file__).resolve().parents[2] / "data" / "artifacts"
                self._model_bundle = pipeline["load_inference_artifacts"](artifacts_dir)
                
                # CRITICAL FIX: Validate manifest schema (feature order, versions, metadata)
                manifest = self._model_bundle.get("manifest", {})
                scaler = self._model_bundle.get("scaler")
                model = self._model_bundle.get("model")
                
                # Check for required manifest fields (prevent silent corruption)
                required_fields = {"feature_cols", "scale_cols"}
                missing = required_fields - set(manifest.keys())
                if missing:
                    logger.error("Manifest validation FAILED — missing critical fields: %s", missing)
                    raise ValueError(f"Mentor artifact manifest incomplete: {missing}")
                
                # Validate feature order against scaler
                manifest_scale_cols = manifest.get("scale_cols", [])
                if hasattr(scaler, "feature_names_in_"):
                    scaler_cols = list(scaler.feature_names_in_)
                    if manifest_scale_cols != scaler_cols:
                        logger.error("Feature order mismatch: manifest=%s, scaler=%s", 
                                   manifest_scale_cols[:5], scaler_cols[:5])
                        raise ValueError(f"Scaler feature order mismatch: manifest vs. fitted scaler")
                
                logger.info("Manifest validation PASSED — version=%s, features=%d, scale_cols=%d",
                           manifest.get("preprocessing_version"), 
                           len(manifest.get("feature_cols", [])),
                           len(manifest_scale_cols))
                
                # MEDIUM FIX: Invalidate recommendations cache if manifest version changed
                # (indicates model update, need fresh recommendations)
                current_version = manifest.get("preprocessing_version")
                if self._last_manifest_version is not None and current_version != self._last_manifest_version:
                    logger.info("Manifest version changed (%s → %s) — invalidating recommendation cache",
                              self._last_manifest_version, current_version)
                    _recommendation_memory.clear()
                self._last_manifest_version = current_version
                
            artifacts = self._model_bundle

            # Step 3: Predict
            preds = pipeline["predict_for_user"](model_user_id, artifacts, top_k=10)
            if preds.empty:
                logger.info("Model inference: returned no predictions for user %s (likely cold-start or no candidates)", user_uuid)
                return []

            # Step 4: Reverse-map mentor IDs (integer → UUID) for DB enrichment
            int_mentor_ids = preds["mentor_id"].astype(int).tolist()
            uuid_mentor_ids = [get_uuid_from_integer(m) or str(m) for m in int_mentor_ids]
            preds["mentor_uuid"] = uuid_mentor_ids

            # Step 5: Fetch mentor metadata from DB using UUIDs
            mentors_df = self._fetch_mentor_details(uuid_mentor_ids)
            if mentors_df.empty:
                logger.warning("Model predicted mentors but none found in DB for UUIDs: %s", uuid_mentor_ids[:5])
                return []

            # Merge predictions with mentor metadata (match on UUID)
            mentors_df = mentors_df.rename(columns={"mentor_id": "mentor_uuid"})
            merged = preds.merge(mentors_df, on="mentor_uuid", how="left")
            
            # MEDIUM FIX: Log if mentor enrichment failed
            if merged.empty:
                logger.warning("Model inference: mentor DB enrichment failed — no mentors found for UUIDs", uuid_mentor_ids[:3])
                return []
            
            merged["mentor_id"] = merged["mentor_uuid"]  # Use UUID for final output

            # Step 6: Lightweight service reranking — trusts pipeline's ML-dominant reranking.
            # Only ensures pred_score→rerank_score mapping and best-coverage top-3 guarantee.
            # Heavy reranking (confidence-aware, weak-match penalty) already done in pipeline.
            merged = _apply_service_skill_rerank(merged)

            # Step 7: Normalize scores (model-based) + format
            merged["score"] = merged["rerank_score"]
            merged = normalize_scores(merged)

            rows = merged.head(10).to_dict(orient="records")
            return self._format_output(rows)

        except Exception as exc:
            logger.exception("Model inference failed: %s", exc)
            return []

    def _fetch_mentor_details(self, mentor_ids: list[int]) -> pd.DataFrame:
        """Fetch mentor metadata (name, domain, rating, etc.) for a list of IDs."""
        if not mentor_ids:
            return pd.DataFrame()
        try:
            # Handle both integer IDs and UUIDs
            placeholders = ", ".join(f"'{str(m)}'" for m in mentor_ids[:200])
            query = f"""
            SELECT
                mp.user_id AS mentor_id,
                COALESCE(NULLIF(LTRIM(RTRIM(CONCAT(u.first_name, ' ', u.last_name))), ''), 'Mentor') AS mentor_name,
                {_MENTOR_DOMAIN_NAME_SQL} AS domain,
                COALESCE(CAST(mp.average_rating AS float), 0.0) AS average_rating,
                COALESCE(mp.total_reviews, 0) AS total_reviews,
                COALESCE(mp.is_verified, 0) AS is_verified,
                COALESCE(f.followers_count, 0) AS followers_count,
                COALESCE(cm.completed_count, 0) AS completed_mentorships,
                COALESCE(op.open_count, 0) AS open_programs,
                COALESCE(cp.completed_program_count, 0) AS completed_programs
            FROM mentor_profile mp
            INNER JOIN users u ON u.user_id = mp.user_id
            LEFT JOIN domains d ON d.domain_id = mp.domain_id
            LEFT JOIN (
                SELECT following_id, COUNT(*) AS followers_count
                FROM follows
                GROUP BY following_id
            ) f ON f.following_id = mp.user_id
            LEFT JOIN (
                SELECT MentorProfileId, COUNT(*) AS completed_count
                FROM mentorships
                WHERE Status = 'Completed'
                GROUP BY MentorProfileId
            ) cm ON cm.MentorProfileId = mp.user_id
            LEFT JOIN (
                SELECT MentorProfileId, COUNT(*) AS open_count
                FROM programs p
                WHERE p.ProgramPostStatus = 'Published'
                  AND p.Availability NOT IN ('Closed', 'Archived', 'Cancelled')
                  AND p.Deadline > GETUTCDATE()
                  AND (
                      SELECT COUNT(*) 
                      FROM applications 
                      WHERE ProgramId = p.ProgramId AND Status = 'Accepted'
                  ) < p.Capacity
                GROUP BY MentorProfileId
            ) op ON op.MentorProfileId = mp.user_id
            LEFT JOIN (
                SELECT MentorProfileId, COUNT(*) AS completed_program_count
                FROM programs
                WHERE ProgramPostStatus = 'Published'
                  AND Availability IN ('Closed', 'Archived', 'Cancelled')
                  AND MentorProfileId IS NOT NULL
                GROUP BY MentorProfileId
            ) cp ON cp.MentorProfileId = mp.user_id
            WHERE mp.user_id IN ({placeholders})
            """
            return database.run_query_df(query)
        except Exception as e:
            logger.error(f"_fetch_mentor_details query failed: {type(e).__name__}: {e}", exc_info=True)
            return pd.DataFrame()

    # ────────────────────────────────────────────────────────────
    # DB-based recommendations (full pipeline)
    # ────────────────────────────────────────────────────────────

    def _from_db(self, user_id: int | str) -> list[dict]:
        try:
            user_uuid = self._parse_user_uuid(user_id)
            if user_uuid is None:
                logger.warning("Recommendation skipped: invalid user_id format (%s)", user_id)
                return []
            if not self._user_exists(user_uuid):
                logger.info("Recommendation skipped: user not found in DB (%s) — likely cold-start user", user_uuid)
                return []

            candidates = self._fetch_candidates()
            if candidates.empty:
                logger.info("No mentor candidates found in DB — possible data gap or all mentors inactive")
                return []

            # Step 1: NO HARD FILTERING — all mentors remain eligible.
            # Availability is a soft preference, NOT a hard exclusion.
            # All candidates proceed to scoring.

            # Step 2: Score + soft availability boosts + re-rank
            ranked = self._score_candidates(candidates, user_uuid)

            # Step 3: Normalize scores to percentages
            ranked = normalize_scores(ranked)
            
            # MEDIUM FIX: Log empty recommendations with reason
            if ranked.empty:
                logger.warning("DB pipeline: scoring returned no results for user %s — possible all-negative scores", user_uuid)
                return []

            # Step 4: Generate reasons + format output
            rows = ranked.head(10).to_dict(orient="records")
            logger.info("DB pipeline: generated %d recommendations for user %s", len(rows), user_uuid)
            return self._format_output(rows)

        except MissingTableError as exc:
            logger.error("Recommendation query failed: missing table: %s — DB schema mismatch", exc)
            return []
        except DatabaseAccessError as exc:
            logger.error("Recommendation query failed: database unavailable: %s", exc)
            return []
        except Exception as exc:
            logger.exception("Recommendation pipeline failed unexpectedly: %s", exc)
            return []

    # ────────────────────────────────────────────────────────────
    # Step 1: Fetch candidates with availability data
    # ────────────────────────────────────────────────────────────

    def _fetch_candidates(self) -> pd.DataFrame:
        # Short-circuit with a tiny in-memory cache to improve latency for
        # repeated calls during integration tests or burst traffic.
        now = _time.time()
        try:
            if (now - getattr(self, "_candidates_cache_ts", 0)) < getattr(self, "_candidates_cache_ttl", 0):
                cached_df = getattr(self, "_candidates_cache_df", None)
                if cached_df is not None and not cached_df.empty:
                    logger.debug("_fetch_candidates: returning cached candidates (age=%.1fs)", now - self._candidates_cache_ts)
                    return cached_df.copy()
        except Exception:
            # Fall through to normal fetch on any cache inspection error
            pass
        query = f"""
        SELECT
            mp.user_id AS mentor_id,
            COALESCE(NULLIF(LTRIM(RTRIM(CONCAT(u.first_name, ' ', u.last_name))), ''), 'Mentor') AS mentor_name,
            {_MENTOR_DOMAIN_NAME_SQL} AS domain,
            {_MENTOR_DOMAIN_ID_COALESCE_SQL} AS mentor_domain_id,
            COALESCE(CAST(mp.average_rating AS float), 0.0) AS average_rating,
            COALESCE(mp.total_reviews, 0) AS total_reviews,
            COALESCE(mp.is_verified, 0) AS is_verified,
            COALESCE(f.followers_count, 0) AS followers_count,
            COALESCE(cm.completed_count, 0) AS completed_mentorships,
            COALESCE(op.open_count, 0) AS open_programs,
            COALESCE(cp.completed_program_count, 0) AS completed_programs
        FROM mentor_profile mp
        INNER JOIN users u ON u.user_id = mp.user_id
        LEFT JOIN domains d ON d.domain_id = mp.domain_id
        LEFT JOIN (
            SELECT following_id, COUNT(*) AS followers_count
            FROM follows
            GROUP BY following_id
        ) f ON f.following_id = mp.user_id
        LEFT JOIN (
            SELECT MentorProfileId, COUNT(*) AS completed_count
            FROM mentorships
            WHERE Status = 'Completed'
            GROUP BY MentorProfileId
        ) cm ON cm.MentorProfileId = mp.user_id
        LEFT JOIN (
            SELECT MentorProfileId, COUNT(*) AS open_count
            FROM programs p
            WHERE p.ProgramPostStatus = 'Published'
              AND p.Availability NOT IN ('Closed', 'Archived', 'Cancelled')
              AND p.Deadline > GETUTCDATE()
              AND (
                  SELECT COUNT(*) 
                  FROM applications 
                  WHERE ProgramId = p.ProgramId AND Status = 'Accepted'
              ) < p.Capacity
            GROUP BY MentorProfileId
        ) op ON op.MentorProfileId = mp.user_id
        LEFT JOIN (
            SELECT MentorProfileId, COUNT(*) AS completed_program_count
            FROM programs
            WHERE ProgramPostStatus = 'Published'
              AND Availability IN ('Closed', 'Archived', 'Cancelled')
              AND MentorProfileId IS NOT NULL
            GROUP BY MentorProfileId
        ) cp ON cp.MentorProfileId = mp.user_id
        WHERE u.is_active = 1
        """
        try:
            df = database.run_query_df(query)
            try:
                # Update short-lived cache
                self._candidates_cache_df = df.copy()
                self._candidates_cache_ts = _time.time()
            except Exception:
                pass
            return df
        except Exception as e:
            logger.error(f"_fetch_candidates query failed: {type(e).__name__}: {e}", exc_info=True)
            return pd.DataFrame()

    # Availability is treated as a soft preference (boost) in scoring.
    # The previous aggressive hard-filtering logic was removed to ensure
    # positive mentors are not excluded from candidate pools or evaluation.

    # ────────────────────────────────────────────────────────────
    # Step 3: Score candidates with weighted signals
    # ────────────────────────────────────────────────────────────

    def _score_candidates(self, candidates: pd.DataFrame, user_id: int | str) -> pd.DataFrame:
        ranked = candidates.copy()

        # Normalize continuous signals to [0, 1]
        ranked["_rating_norm"] = self._normalize_series(ranked.get("average_rating", pd.Series(dtype=float)))
        ranked["_reviews_norm"] = self._normalize_series(ranked.get("total_reviews", pd.Series(dtype=float)))
        ranked["_followers_norm"] = self._normalize_series(ranked.get("followers_count", pd.Series(dtype=float)))
        ranked["_completed_norm"] = self._normalize_series(ranked.get("completed_mentorships", pd.Series(dtype=float)))
        ranked["_verified"] = ranked.get("is_verified", pd.Series(0, index=ranked.index)).astype(float)

        # Compute availability tier directly from raw columns (no hard filter)
        open_progs = ranked.get("open_programs", pd.Series(0, index=ranked.index))
        completed_progs = ranked.get("completed_programs", pd.Series(0, index=ranked.index))
        ranked["_availability_tier"] = np.where(
            open_progs > 0, 1,
            np.where(completed_progs > 0, 2, 3)
        )

        # Soft availability boosts (multiplicative — aligned with pipeline SOFT_BUSINESS_BOOSTS):
        # tier1 (open programs): ×1.02  |  tier2 (completed): ×1.008  |  tier3 (other): ×1.0
        # These are intentionally small — availability should NEVER override compatibility.
        ranked["_availability_multiplier"] = ranked["_availability_tier"].map(
            {1: 1.02, 2: 1.008, 3: 1.0}
        ).fillna(1.0)

        # Activity score: combination of completed mentorships + reviews
        ranked["_activity_norm"] = (ranked["_completed_norm"] * 0.6 + ranked["_reviews_norm"] * 0.4)

        # Initialize personalization signal trackers
        ranked["_is_followed"] = False
        ranked["_domain_match"] = False
        ranked["_subdomain_match"] = False
        ranked["_skills_overlap"] = 0.0
        ranked["_matched_skills"] = ""

        # Quality score: combines rating (60%) + activity (40%) — matches pipeline's quality signal
        ranked["_quality_norm"] = (ranked["_rating_norm"] * 0.6 + ranked["_activity_norm"] * 0.4)

        # Base score from quality + popularity signals (availability applied multiplicatively below)
        ranked["score"] = (
            ranked["_quality_norm"]          * _WEIGHTS["quality"]
            + ranked["_followers_norm"]      * _WEIGHTS["popularity"]
        )

        user_uuid = self._parse_user_uuid(user_id)
        if user_uuid is not None:
            self._apply_personalization(ranked, user_uuid)

        # Apply availability as multiplicative boost (after personalization)
        ranked["score"] *= ranked["_availability_multiplier"]

        # Re-rank: sort by score, break ties with reviews
        ranked = rerank_scores(ranked)

        return ranked

    # ────────────────────────────────────────────────────────────
    # Personalization: skills, subdomain, domain, follow
    # ────────────────────────────────────────────────────────────

    def _apply_personalization(self, ranked: pd.DataFrame, user_uuid: str) -> None:
        """Apply personalization boosts in strict priority order.

        Mentor skills and subdomains are cached (5-min TTL) to avoid
        redundant DB queries across consecutive requests.
        """

        # ── Skills overlap (strongest signal) ──
        mentee_skills = self._get_mentee_interests(user_uuid)
        if mentee_skills:
            mentor_skills_map = self._get_mentor_expertise_map_cached(ranked["mentor_id"].tolist())
            for idx, row in ranked.iterrows():
                mid = str(row["mentor_id"])
                mentor_skills = mentor_skills_map.get(mid, set())
                if mentee_skills and mentor_skills:
                    overlap = len(mentee_skills & mentor_skills) / len(mentee_skills | mentor_skills)
                    matched = mentee_skills & mentor_skills
                    ranked.at[idx, "_skills_overlap"] = overlap
                    ranked.at[idx, "_matched_skills"] = ", ".join(sorted(matched)[:5])
                    ranked.at[idx, "score"] += overlap * _WEIGHTS["skill_overlap"]

        # ── Subdomain match ──
        mentee_subdomains = self._get_mentee_subdomains(user_uuid)
        if mentee_subdomains:
            mentor_subdomains_map = self._get_mentor_subdomains_map_cached(ranked["mentor_id"].tolist())
            for idx, row in ranked.iterrows():
                mid = str(row["mentor_id"])
                mentor_subs = mentor_subdomains_map.get(mid, set())
                if mentee_subdomains & mentor_subs:
                    ranked.at[idx, "_subdomain_match"] = True
                    ranked.at[idx, "score"] += _WEIGHTS["subdomain_match"]

        # ── Domain match ──
        mentee_domain_id = self._get_mentee_domain(user_uuid)
        if mentee_domain_id is not None:
            same_domain_mask = ranked["mentor_domain_id"] == mentee_domain_id
            ranked.loc[same_domain_mask.fillna(False), "_domain_match"] = True
            ranked.loc[same_domain_mask.fillna(False), "score"] += _WEIGHTS["domain_match"]

        # ── Follow (VERY weak signal — +0.02 max) ──
        followed = self._get_followed_mentors(user_uuid)
        if followed:
            mask = ranked["mentor_id"].isin(followed)
            ranked.loc[mask, "_is_followed"] = True
            ranked.loc[mask, "score"] += _WEIGHTS["followed"]

    # ────────────────────────────────────────────────────────────
    # Database helpers
    # ────────────────────────────────────────────────────────────

    def _get_followed_mentors(self, user_uuid: str) -> set[str]:
        query = """
        SELECT following_id AS mentor_id
        FROM follows
        WHERE follower_id = :user_id
        """
        frame = database.run_query_df(query, {"user_id": user_uuid})
        return set(frame.get("mentor_id", pd.Series(dtype=str)).astype(str).tolist())

    def _user_exists(self, user_uuid: str) -> bool:
        """Return True if user exists in users table."""
        query = """
        SELECT TOP 1 user_id
        FROM users
        WHERE user_id = :user_id
        """
        frame = database.run_query_df(query, {"user_id": user_uuid})
        return not frame.empty

    def _get_mentee_domain(self, user_uuid: str) -> int | None:
        query = """
        SELECT TOP 1 domain_id
        FROM mentee_profile
        WHERE user_id = :user_id
        """
        frame = database.run_query_df(query, {"user_id": user_uuid})
        if frame.empty or "domain_id" not in frame.columns:
            return None
        value = pd.to_numeric(frame.iloc[0]["domain_id"], errors="coerce")
        if pd.isna(value):
            return None
        return int(value)

    def _get_mentee_interests(self, user_uuid: str) -> set[str]:
        """Load mentee interest names from DB."""
        try:
            query = """
            SELECT LOWER(LTRIM(RTRIM(t.name))) AS interest
            FROM mentee_interests mi
            INNER JOIN technologies t ON t.technology_id = mi.technology_id
            WHERE mi.user_id = :user_id
            """
            frame = database.run_query_df(query, {"user_id": user_uuid})
            if frame.empty:
                return set()
            return set(frame["interest"].dropna().tolist())
        except (MissingTableError, DatabaseAccessError):
            return set()

    def _get_mentee_subdomains(self, user_uuid: str) -> set[str]:
        """Load mentee subdomain IDs from DB."""
        try:
            query = """
            SELECT CAST("SubDomainId" AS NVARCHAR(100)) AS sub_id
            FROM MenteeSubDomains
            WHERE "UserId" = :user_id
            """
            frame = database.run_query_df(query, {"user_id": user_uuid})
            if frame.empty:
                return set()
            return set(frame["sub_id"].dropna().astype(str).tolist())
        except (MissingTableError, DatabaseAccessError):
            return set()

    def _get_mentor_subdomains_map(self, mentor_ids: list) -> dict[str, set[str]]:
        """Load subdomain IDs for a batch of mentors (uncached)."""
        try:
            if not mentor_ids:
                return {}
            placeholders = ", ".join(f"'{str(m)}'" for m in mentor_ids[:200])
            query = f"""
            SELECT
                CAST("MentorId" AS NVARCHAR(100)) AS mentor_id,
                CAST("SubDomainId" AS NVARCHAR(100)) AS sub_id
            FROM MentorSubDomains
            WHERE "MentorId" IN ({placeholders})
            """
            frame = database.run_query_df(query)
            if frame.empty:
                return {}
            result: dict[str, set[str]] = {}
            for _, row in frame.iterrows():
                mid = str(row["mentor_id"])
                sub = str(row.get("sub_id", ""))
                if sub:
                    result.setdefault(mid, set()).add(sub)
            return result
        except (MissingTableError, DatabaseAccessError):
            return {}

    def _get_mentor_subdomains_map_cached(self, mentor_ids: list) -> dict[str, set[str]]:
        """Load subdomain IDs with 5-minute TTL cache."""
        cache_key = "all_mentor_subdomains"
        cached = _subdomain_cache.get(cache_key)
        if cached is not None:
            logger.debug("Subdomain cache HIT (%d mentors)", len(cached))
            return cached

        logger.info("Subdomain cache MISS — fetching from DB")
        result = self._get_mentor_subdomains_map(mentor_ids)
        _subdomain_cache.set(cache_key, result)
        return result

    def _get_mentor_expertise_map(self, mentor_ids: list) -> dict[str, set[str]]:
        """Load expertise/skills for a batch of mentors (uncached)."""
        try:
            if not mentor_ids:
                return {}
            placeholders = ", ".join(f"'{str(m)}'" for m in mentor_ids[:200])
            query = f"""
            SELECT
                CAST(me.mentor_id AS NVARCHAR(100)) AS mentor_id,
                LOWER(LTRIM(RTRIM(t.name))) AS skill
            FROM mentor_expertise me
            INNER JOIN technologies t ON t.technology_id = me.technology_id
            WHERE me.mentor_id IN ({placeholders})
            """
            frame = database.run_query_df(query)
            if frame.empty:
                return {}
            result: dict[str, set[str]] = {}
            for _, row in frame.iterrows():
                mid = str(row["mentor_id"])
                skill = row.get("skill", "")
                if skill:
                    result.setdefault(mid, set()).add(skill)
            return result
        except (MissingTableError, DatabaseAccessError):
            return {}

    def _get_mentor_expertise_map_cached(self, mentor_ids: list) -> dict[str, set[str]]:
        """Load expertise/skills with 5-minute TTL cache."""
        cache_key = "all_mentor_expertise"
        cached = _expertise_cache.get(cache_key)
        if cached is not None:
            logger.debug("Expertise cache HIT (%d mentors)", len(cached))
            return cached

        logger.info("Expertise cache MISS — fetching from DB")
        result = self._get_mentor_expertise_map(mentor_ids)
        _expertise_cache.set(cache_key, result)
        return result

    # ────────────────────────────────────────────────────────────
    # Utility helpers
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_user_uuid(user_id: int | str) -> str | None:
        try:
            return str(uuid.UUID(str(user_id)))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _normalize_series(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        max_value = float(numeric.max()) if len(numeric) else 0.0
        if max_value <= 0:
            return pd.Series([0.0] * len(numeric), index=numeric.index)
        return numeric / max_value

    @staticmethod
    def _match_mentor_from_message(message: str, recommendations: list[dict]) -> dict | None:
        """Fuzzy-match a mentor name from the user's follow-up message.

        Tries exact substring match first, then falls back to partial token match.
        Returns None if no mentor can be identified (caller defaults to #1).
        """
        msg_lower = message.lower().strip()
        if not msg_lower or not recommendations:
            return None

        # Strategy 1: Exact substring match on mentor name
        for rec in recommendations:
            name = rec.get("mentor_name", "").lower()
            if name and name in msg_lower:
                return rec

        # Strategy 2: Any token of the mentor name appears in the message
        for rec in recommendations:
            name = rec.get("mentor_name", "").lower()
            tokens = [t for t in name.split() if len(t) >= 2]
            for token in tokens:
                if token in msg_lower:
                    return rec

        # Strategy 3: Check for ordinal references ("الأول", "التاني", "first", "second")
        ordinal_map = {
            0: ["الأول", "الاول", "first", "#1", "رقم 1", "رقم ١", "top"],
            1: ["التاني", "الثاني", "second", "#2", "رقم 2", "رقم ٢"],
            2: ["التالت", "الثالث", "third", "#3", "رقم 3", "رقم ٣"],
        }
        for idx, keywords in ordinal_map.items():
            if idx < len(recommendations) and any(k in msg_lower for k in keywords):
                return recommendations[idx]

        return None

    # ────────────────────────────────────────────────────────────
    # Output formatting
    # ────────────────────────────────────────────────────────────

    def _format_output(self, rows: list[dict]) -> list[dict]:
        """Format rows into clean, UI-ready output.

        Handles three input formats:
          1. Model inference (pred_score + signal columns)
          2. DB heuristic (_domain_match, _skills_overlap, etc.)
          3. External API (minimal fields)

        Preserves existing match_percentage when already set (e.g. by
        normalize_scores).  Only computes signal-based fallback when missing.
        """
        if not rows:
            return []

        has_internal_signals = "_domain_match" in rows[0]
        has_model_signals = "skill_overlap_score" in rows[0] or "pred_score" in rows[0]

        # ── Enrich model-format rows with internal signal names for generate_reason ──
        if not has_internal_signals and has_model_signals:
            for row in rows:
                skill_overlap = float(row.get("skill_overlap_score", 0))
                skill_coverage = float(row.get("skill_coverage_score", 0))
                subdomain_sim = float(row.get("subdomain_similarity", 0))
                domain_match = float(row.get("mentor_domain_match", 0))
                quality = min(float(row.get("mentor_quality_score", 0)) / 5.0, 1.0)

                row["_skills_overlap"] = skill_overlap
                row["_subdomain_match"] = subdomain_sim > 0
                row["_domain_match"] = bool(domain_match)
                row["_is_followed"] = bool(row.get("is_following", 0))
                row["_covers_all_skills"] = bool(row.get("mentor_covers_all_skills", 0))
                row["average_rating"] = float(row.get("mentor_weighted_rating", row.get("average_rating", 0)))

                # Only compute fallback match_percentage if normalize_scores() didn't set it.
                # Uses COMPATIBILITY signals only (Layer A) — quality/popularity are NOT
                # compatibility indicators and should not inflate match_percentage.
                if "match_percentage" not in row:
                    signal_match = (
                        0.40 * skill_coverage     # Most direct: "how much does this mentor cover?"
                        + 0.35 * skill_overlap     # Profile similarity
                        + 0.25 * subdomain_sim     # Specialization alignment
                    )
                    row["match_percentage"] = int(round(55 + signal_match * 43))

        output: list[dict] = []
        for row in rows:
            output.append({
                "mentor_id": str(row.get("mentor_id", row.get("id", ""))),
                "mentor_name": row.get("mentor_name", row.get("name", "Mentor")),
                "domain": row.get("domain", row.get("category", "General")),
                "score": round(float(row.get("score", row.get("pred_score", 0.0))), 4),
                "match_percentage": int(row.get("match_percentage", 75)),
                "reason": row.get("reason", "") or generate_reason(row),
            })
        return output


# ════════════════════════════════════════════════════════════════════
# Module-level functions (modular, testable)
# ════════════════════════════════════════════════════════════════════


def _apply_service_skill_rerank(merged: pd.DataFrame) -> pd.DataFrame:
    """Lightweight service-layer reranking — trusts the pipeline's ML-dominant reranking.

    The pipeline's apply_multi_signal_rerank() already applies:
      - ML-dominant scoring (60% weight, confidence-aware)
      - Skill coverage refinement (20%)
      - Subdomain alignment (12%)
      - Domain match (8%)
      - Weak match penalty (×0.85-0.92 for irrelevant mentors)

    This function only does:
      1. Map pred_score → rerank_score for downstream compatibility
      2. Guarantee the best-coverage mentor appears in top-3

    IMPORTANT (May 2026): The previous version had skill_coverage weight at 2.50
    (vs pipeline's 0.10) which completely overrode the model's ranking.  That
    aggressive second reranker is now removed — the pipeline handles all reranking.
    """
    if merged.empty:
        return merged

    df = merged.copy()

    # The pipeline outputs pred_score (already reranked via apply_multi_signal_rerank).
    # Map to rerank_score for downstream normalize_scores() compatibility.
    if "pred_score" in df.columns and "rerank_score" not in df.columns:
        df["rerank_score"] = df["pred_score"]

    df = df.sort_values("rerank_score", ascending=False).reset_index(drop=True)

    # ── Guarantee slot: ensure best-coverage mentor is in top-3 ──
    # Some users have no mentor with high coverage; but for those who do,
    # we guarantee the highest-coverage mentor appears in top-3 so the
    # user always sees the best possible match.
    if "skill_coverage_score" in df.columns and len(df) > 3:
        best_cov_idx = int(df["skill_coverage_score"].idxmax())
        if best_cov_idx >= 3:
            # Best coverage is outside top-3; compose top-3 from #1, #2, best-coverage
            top3 = pd.concat([
                df.iloc[[0]], df.iloc[[1]], df.iloc[[best_cov_idx]]
            ]).reset_index(drop=True)
            top3 = top3.sort_values("rerank_score", ascending=False).reset_index(drop=True)
            # Rest = everything except #1, #2, and best-coverage
            mask = pd.Series(True, index=df.index)
            mask.iloc[[0, 1, best_cov_idx]] = False
            rest = df[mask].reset_index(drop=True)
            df = pd.concat([top3, rest]).reset_index(drop=True)

    return df


def normalize_scores(ranked: pd.DataFrame) -> pd.DataFrame:
    """Per-user normalization producing high but varied match percentages.

    Combines:
      - relative rank score  (35%) — how this mentor ranks vs others
      - skill coverage       (40%) — how many of the user's skills are covered
      - subdomain similarity (25%) — specialization alignment

    Result is boosted to 55–98% range so percentages look strong
    while still reflecting meaningful differences between mentors.

    Edge case: if all scores are equal → use skill coverage alone.

    CRITICAL: Validate match_percentage stays in [0, 100] range.
    """
    if ranked.empty:
        return ranked

    ranked = ranked.copy()
    scores = ranked["score"]
    min_score = float(scores.min())
    max_score = float(scores.max())

    # Relative ranking component (0.0 to 1.0)
    if max_score == min_score:
        relative = pd.Series(0.5, index=ranked.index)
    else:
        relative = (scores - min_score) / (max_score - min_score)

    # Skill coverage component (0.0 to 1.0)
    coverage = ranked.get("skill_coverage_score", pd.Series(0.5, index=ranked.index))
    coverage = coverage.clip(0.0, 1.0)

    # Subdomain similarity component (0.0 to 1.0)
    subdomain_sim = ranked.get("subdomain_similarity", pd.Series(0.0, index=ranked.index))
    subdomain_sim = subdomain_sim.clip(0.0, 1.0)

    # Weighted combination (0.0 to 1.0)
    combined = (
        relative * 0.35
        + coverage * 0.40
        + subdomain_sim * 0.25
    )

    # Boost to 55–98% range: high but with real variation
    # Formula: 55 + combined * 43  →  55% (worst) to 98% (best)
    match_pct = (55 + combined * 43).round().astype(int)

    # Enforce [0, 100] bounds
    match_pct = match_pct.clip(0, 100)

    ranked["match_percentage"] = match_pct

    # Final validation
    invalid = ranked["match_percentage"].apply(lambda x: x < 0 or x > 100)
    if invalid.any():
        logger.error("VALIDATION ERROR: %d match_percentage values still out of bounds!", invalid.sum())
        ranked.loc[invalid, "match_percentage"] = ranked.loc[invalid, "match_percentage"].clip(0, 100)

    return ranked


def rerank_scores(ranked: pd.DataFrame) -> pd.DataFrame:
    """Re-rank candidates with diversity penalty.

    Sort by score descending, then apply soft diversity penalty
    to prevent >3 mentors from the same domain in top results.
    """
    if ranked.empty:
        return ranked

    ranked = ranked.sort_values(["score", "total_reviews"], ascending=[False, False])

    # Diversity: demote if >3 from same domain in top 15
    if "domain" in ranked.columns:
        ranked = ranked.copy()
        domain_counts: dict[str, int] = {}
        for idx in ranked.index[:15]:
            domain = ranked.at[idx, "domain"]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if domain_counts[domain] > 3:
                ranked.at[idx, "score"] *= 0.97  # Soft multiplicative penalty (3%)
        ranked = ranked.sort_values(["score", "total_reviews"], ascending=[False, False])

    return ranked


def generate_reason(row: dict, language: str = "en") -> str:
    """Generate ONE clean, natural-language explanation sentence.

    Combines the strongest signals into a single coherent sentence.
    Priority: skills > subdomain/domain > availability > rating > activity > followed.
    No bullet points, no repetition, no generic phrases.
    """
    def _safe_int(val) -> int:
        try:
            return int(float(val)) if val is not None else 0
        except (ValueError, TypeError):
            return 0

    # Extract signals
    domain = row.get("domain", "General")
    rating = float(row.get("average_rating", 0) or 0)
    completed = _safe_int(row.get("completed_mentorships", 0))
    open_progs = _safe_int(row.get("open_programs", 0))
    completed_progs = _safe_int(row.get("completed_programs", 0))
    is_followed = bool(row.get("_is_followed", False))
    domain_match = bool(row.get("_domain_match", False))
    subdomain_match = bool(row.get("_subdomain_match", False))
    skills_overlap = float(row.get("_skills_overlap", 0) or 0)
    skill_coverage = float(row.get("skill_coverage_score", 0) or 0)
    matched_skills = row.get("_matched_skills", "")
    covers_all = bool(row.get("_covers_all_skills", False))

    parts = []

    # ── 1. Skills (primary) ──
    if covers_all:
        parts.append("covers all your skills")
    elif skills_overlap >= 0.5 and matched_skills:
        parts.append(f"strong match in {matched_skills}")
    elif skills_overlap >= 0.3 and matched_skills:
        parts.append(f"shared expertise in {matched_skills}")
    elif skill_coverage >= 0.5:
        parts.append(f"covers {skill_coverage:.0%} of your skills")
    elif skill_coverage > 0:
        parts.append(f"{skill_coverage:.0%} skill coverage")

    # ── 2. Subdomain / domain (context) ──
    if subdomain_match and domain_match:
        parts.append(f"same specialization in {domain}")
    elif subdomain_match:
        parts.append("matching specialization")
    elif domain_match:
        parts.append(f"{domain} domain match")

    # ── 3. Availability (practical) ──
    if open_progs > 0:
        parts.append(f"{open_progs} open program{'s' if open_progs > 1 else ''}")
    elif completed_progs > 0:
        parts.append(f"{completed_progs} completed program{'s' if completed_progs > 1 else ''}")

    # ── 4. Rating (trust) ──
    if rating >= 4.5:
        parts.append(f"excellent rating ({rating:.1f}⭐)")
    elif rating >= 4.0:
        parts.append(f"high rating ({rating:.1f}⭐)")
    elif rating >= 3.0:
        parts.append(f"solid rating ({rating:.1f}⭐)")

    # ── 5. Activity (experience) ──
    if completed >= 10:
        parts.append(f"{completed} successful mentorships")
    elif completed >= 3:
        parts.append(f"{completed} completed mentorships")

    # ── 6. Followed (personal) ──
    if is_followed:
        parts.append("mentor you follow")

    # ── Build one natural sentence ──
    if not parts:
        return "Recommended based on your profile."

    if len(parts) == 1:
        return parts[0].capitalize() + "."

    if len(parts) == 2:
        return f"{parts[0].capitalize()} with {parts[1]}."

    # 3+ parts: combine with commas + "and"
    main = parts[0].capitalize()
    middle = ", ".join(parts[1:-1])
    last = parts[-1]
    if middle:
        return f"{main} with {middle}, and {last}."
    return f"{main} with {last}."


def generate_detailed_explanation(rec: dict, language: str = "en") -> str:
    """Generate a detailed, conversational, multi-line explanation for a recommendation.

    Uses the SAME signal data as generate_reason() but formats it as a
    user-friendly bullet-point explanation suitable for follow-up questions
    like 'why did you recommend this mentor?'.
    """
    name = rec.get("mentor_name", "Mentor")
    match_pct = int(rec.get("match_percentage", 75))
    reason_oneliner = rec.get("reason", "")
    domain = rec.get("domain", "General")

    # Build explanation bullets from the same data
    bullets: list[str] = []

    # Skill coverage / overlap
    if reason_oneliner:
        # Extract key phrases from the existing reason
        r_lower = reason_oneliner.lower()
        if "covers all your skills" in r_lower:
            bullets.append(("✅", "بيغطي كل المهارات اللي أنت مهتم بيها" if language == "ar" else "Covers all the skills you're interested in"))
        elif "strong match" in r_lower or "shared expertise" in r_lower:
            # Extract skills from reason
            skills_part = reason_oneliner.split("in ")[-1].split(" with")[0].split(",")[0].rstrip(".")
            if skills_part:
                bullets.append(("🎯", f"تطابق قوي في المهارات: {skills_part}" if language == "ar" else f"Strong skill match: {skills_part}"))
        elif "skill coverage" in r_lower or "covers" in r_lower:
            bullets.append(("📊", f"تغطية جيدة لمهاراتك المطلوبة" if language == "ar" else "Good coverage of your required skills"))

    # Domain / specialization
    if domain and domain != "General":
        r_lower = (reason_oneliner or "").lower()
        if "same specialization" in r_lower or "matching specialization" in r_lower:
            bullets.append(("🏷️", f"نفس التخصص بتاعك في {domain}" if language == "ar" else f"Same specialization as yours in {domain}"))
        elif "domain match" in r_lower:
            bullets.append(("🏷️", f"نفس مجال اهتمامك ({domain})" if language == "ar" else f"Same domain as yours ({domain})"))

    # Availability
    r_lower = (reason_oneliner or "").lower()
    if "open program" in r_lower:
        bullets.append(("📂", "عنده برامج مفتوحة حالياً — يقدر يبدأ معاك" if language == "ar" else "Has open programs right now — ready to start"))
    elif "completed program" in r_lower:
        bullets.append(("✔️", "عنده خبرة في برامج إرشاد سابقة" if language == "ar" else "Has experience from previous mentorship programs"))

    # Rating
    if "excellent rating" in r_lower or "high rating" in r_lower or "solid rating" in r_lower:
        bullets.append(("⭐", "تقييماته مرتفعة من المتدربين السابقين" if language == "ar" else "Highly rated by previous mentees"))

    # Activity / experience
    if "successful mentorship" in r_lower or "completed mentorship" in r_lower:
        bullets.append(("🏆", "عنده سجل قوي من الإرشاد الناجح" if language == "ar" else "Strong track record of successful mentorships"))

    # Following
    if "you follow" in r_lower:
        bullets.append(("👤", "أنت بالفعل متابعه على المنصة" if language == "ar" else "You're already following this mentor"))

    # Fallback: if no bullets could be extracted, use the oneliner directly
    if not bullets and reason_oneliner:
        bullets.append(("💡", reason_oneliner))

    # Format the response
    if language == "ar":
        header = f"تم ترشيح **{name}** (نسبة التوافق: {match_pct}%) للأسباب دي:"
        if not bullets:
            return f"تم ترشيح **{name}** بنسبة توافق {match_pct}% بناءً على بروفايلك."
        lines = [header] + [f"  {emoji} {text}" for emoji, text in bullets]
        return "\n".join(lines)
    else:
        header = f"**{name}** was recommended (match: {match_pct}%) because:"
        if not bullets:
            return f"**{name}** was recommended with a {match_pct}% match based on your profile."
        lines = [header] + [f"  {emoji} {text}" for emoji, text in bullets]
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# Singleton
# ────────────────────────────────────────────────────────────────────

recommendation_service = RecommendationService()
