"""Program recommendation service for mentee → program matching.

This service mirrors the mentor recommendation service structure, but uses the
program recommender artifacts and the real database program schema.
"""

from __future__ import annotations

import logging
import sys
import time as _time
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from database.db import DatabaseAccessError, MissingTableError
from services.uuid_mapper import get_mentee_integer_id

logger = logging.getLogger(__name__)


_program_module = None


from services.cache import TTLCache as _TTLCache


_recommendation_cache = _TTLCache(ttl_seconds=300)



def _get_program_modules() -> dict[str, Any]:
    global _program_module
    if _program_module is None:
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.hybrid_recommender.features import build_mentee_features
        from src.hybrid_recommender.preprocessing import (
            build_time_split_config,
            load_db_datasets_from_db,
            prepare_processed_tables,
        )
        from src.hybrid_recommender.program_recommender.features import (
            build_mentee_program_candidates,
            build_program_cf_embeddings,
            build_program_features,
        )
        from src.hybrid_recommender.program_recommender.io import (
            get_default_program_artifact_paths,
            load_feature_artifact,
            load_json_artifact,
            load_model,
            load_scaler,
        )
        from src.hybrid_recommender.program_recommender.pipeline import _add_program_cf_score
        from src.hybrid_recommender.program_recommender.ranking import (
            generate_program_recommendations,
        )
        from src.hybrid_recommender.program_recommender.preprocessing import align_program_feature_frame, apply_program_scaler, validate_program_artifact_compatibility

        _program_module = {
            "build_mentee_features": build_mentee_features,
            "build_time_split_config": build_time_split_config,
            "load_db_datasets_from_db": load_db_datasets_from_db,
            "prepare_processed_tables": prepare_processed_tables,
            "build_mentee_program_candidates": build_mentee_program_candidates,
            "build_program_cf_embeddings": build_program_cf_embeddings,
            "build_program_features": build_program_features,
            "get_default_program_artifact_paths": get_default_program_artifact_paths,
            "load_feature_artifact": load_feature_artifact,
            "load_json_artifact": load_json_artifact,
            "load_model": load_model,
            "load_scaler": load_scaler,
            "apply_program_scaler": apply_program_scaler,
            "align_program_feature_frame": align_program_feature_frame,
            "validate_program_artifact_compatibility": validate_program_artifact_compatibility,
            "add_program_cf_score": _add_program_cf_score,
            "generate_program_recommendations": generate_program_recommendations,
        }
    return _program_module


class ProgramRecommendationService:
    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None
        self._processed: dict[str, pd.DataFrame] | None = None
        self._program_features: pd.DataFrame | None = None
        self._cf_embeddings: dict[str, dict[int, Any]] | None = None

    async def get_recommendations(self, user_id: int | str, top_k: int = 10) -> list[dict]:
        cache_key = f"{str(user_id).strip()}:{int(top_k)}"
        cached = _recommendation_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            results = self._from_model(user_id=user_id, top_k=top_k)
            if results:
                _recommendation_cache.set(cache_key, results)
            return results
        except Exception as exc:
            logger.exception("Program recommendation failed for user_id=%s: %s", user_id, exc)
            return []

    def _load_bundle(self) -> dict[str, Any]:
        if self._bundle is not None:
            return self._bundle
        try:
            modules = _get_program_modules()
            paths = modules["get_default_program_artifact_paths"]()
            manifest = modules["load_json_artifact"](paths["manifest"])
            bundle = {
                "model": modules["load_model"](paths["model"]),
                "scaler": modules["load_scaler"](paths["scaler"]),
                "manifest": manifest,
                "feature_cols": manifest.get("feature_cols", []),
                "scale_cols": manifest.get("scale_cols", []),
            }
            modules["validate_program_artifact_compatibility"](manifest, bundle["scaler"], bundle["model"])
            self._bundle = bundle
            return bundle
        except Exception as exc:
            logger.exception("Failed to load program artifact bundle: %s", exc)
            raise

    def _load_processed(self) -> dict[str, pd.DataFrame]:
        if self._processed is not None:
            return self._processed
        try:
            modules = _get_program_modules()
            raw_tables = modules["load_db_datasets_from_db"]()
            config = modules["build_time_split_config"](
                raw_tables.get("mentorships", pd.DataFrame()),
                applications=raw_tables.get("mentorship_applications"),
            )
            processed = modules["prepare_processed_tables"](raw_tables, config)
            self._processed = processed
            return processed
        except (DatabaseAccessError, MissingTableError) as exc:
            logger.exception("Database error while loading processed tables: %s", exc)
            raise
        except Exception as exc:
            logger.exception("Unexpected error while preparing processed tables: %s", exc)
            raise

    def _load_program_features(self) -> pd.DataFrame:
        if self._program_features is not None:
            return self._program_features
        try:
            modules = _get_program_modules()
            processed = self._load_processed()
            program_features = modules["build_program_features"](
                processed.get("mentorship_posts", pd.DataFrame()),
                processed.get("mentorship_requirements", pd.DataFrame()),
                program_enrollments=processed.get("mentorships"),
                mentorship_applications=processed.get("mentorship_applications"),
                reference_time=pd.Timestamp.utcnow().tz_localize(None),
            )
            self._program_features = program_features
            return program_features
        except Exception as exc:
            logger.exception("Failed to build program features: %s", exc)
            return pd.DataFrame()

    def _load_cf_embeddings(self) -> dict[str, dict[int, Any]]:
        if self._cf_embeddings is not None:
            return self._cf_embeddings
        try:
            modules = _get_program_modules()
            processed = self._load_processed()
            cf_embeddings = modules["build_program_cf_embeddings"](
                processed.get("mentorships", pd.DataFrame()),
                likes=processed.get("posts_likes_dataset"),
                saves=processed.get("saved_posts_dataset"),
                comments=processed.get("posts_comments"),
                shares=processed.get("shared_posts_dataset"),
                n_factors=16,
            )
            self._cf_embeddings = cf_embeddings
            return cf_embeddings
        except Exception:
            logger.exception("Failed to build CF embeddings, continuing without CF scores")
            self._cf_embeddings = {}
            return {}

    @staticmethod
    def _resolve_user_id(user_id: int | str) -> int | None:
        try:
            if isinstance(user_id, int):
                return user_id
            user_text = str(user_id).strip()
            if not user_text:
                return None
            if user_text.isdigit():
                return int(user_text)
            mapped = get_mentee_integer_id(user_text)
            if mapped is not None:
                return mapped
            return int(user_text)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _looks_like_uuid(value: str) -> bool:
        return len(value) > 10 and "-" in value

    def _from_model(self, user_id: int | str, top_k: int = 10) -> list[dict]:
        modules = _get_program_modules()
        model_user_id = self._resolve_user_id(user_id)
        if model_user_id is None:
            logger.warning("Program recommendation skipped: invalid user_id (%s)", user_id)
            return []

        processed = self._load_processed()
        mentee_features = modules["build_mentee_features"](
            processed["mentee_profile"],
            processed["mentee_subdomains"],
            processed["mentee_interests"],
        )
        target_mentee = mentee_features[mentee_features["mentee_id"] == model_user_id].copy()
        if target_mentee.empty:
            logger.warning("Program recommendation skipped: mentee %s not found", model_user_id)
            return []

        program_features = self._load_program_features()
        candidates = modules["build_mentee_program_candidates"](
            mentee_features=target_mentee,
            program_features=program_features,
            mentee_interest_levels=processed["mentee_interests"],
            top_k_per_mentee=max(len(program_features), top_k * 50),
            enforce_hard_gates=True,
        )
        if candidates.empty:
            logger.info("Program recommendation returned no eligible candidates for %s", user_id)
            return []

        # ── NEW (May 2026): Exclude programs user has already applied to ──
        applications = processed.get("mentorship_applications", pd.DataFrame())
        if not applications.empty:
            user_applications = applications[
                (applications.get("mentee_id") == model_user_id) |
                (applications.get("user_id") == model_user_id)
            ].copy()
            if not user_applications.empty:
                user_applications["post_id"] = pd.to_numeric(user_applications.get("post_id"), errors="coerce")
                already_applied = set(user_applications["post_id"].dropna().astype(int).unique())
                candidates_before = len(candidates)
                candidates = candidates[~candidates["post_id"].isin(already_applied)].copy()
                logger.info(
                    "Excluded %d programs user %s already applied to (kept %d/%d programs)",
                    candidates_before - len(candidates), user_id, len(candidates), candidates_before,
                )
        
        if candidates.empty:
            logger.info("No eligible programs found after excluding already-applied programs for %s", user_id)
            return []

        candidates = modules["add_program_cf_score"](candidates, self._load_cf_embeddings())
        bundle = self._load_bundle()

        candidates = modules["align_program_feature_frame"](candidates, bundle.get("manifest", {}), mode="soft")
        scale_cols = [c for c in bundle.get("scale_cols", []) if c in candidates.columns]
        if scale_cols:
            candidates = modules["apply_program_scaler"](candidates, bundle["scaler"], scale_cols, feature_cols=bundle.get("feature_cols", []))

        recs = modules["generate_program_recommendations"](
            bundle["model"],
            candidates,
            bundle.get("feature_cols", []),
            top_k=top_k,
        )
        if recs.empty:
            return []

        recs = self._enrich_program_recommendations(recs, processed)
        return recs.head(top_k).to_dict(orient="records")

    @staticmethod
    def _enrich_program_recommendations(
        recs: pd.DataFrame,
        processed: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        out = recs.copy()

        posts = processed.get("mentorship_posts", pd.DataFrame()).copy()
        if not posts.empty:
            cols = [c for c in ["post_id", "mentor_id", "title", "description", "target_level", "education_level", "availability", "capacity", "domain_id", "subdomain_id"] if c in posts.columns]
            posts = posts[cols].drop_duplicates(subset=["post_id"])
            out["post_id"] = pd.to_numeric(out["post_id"], errors="coerce").astype("Int64")
            posts["post_id"] = pd.to_numeric(posts["post_id"], errors="coerce").astype("Int64")
            out = out.merge(posts, on="post_id", how="left", suffixes=("", "_program"))

        users = processed.get("users", pd.DataFrame()).copy()
        if not users.empty and {"user_id"}.issubset(users.columns):
            name_cols = [c for c in ["user_id", "first_name", "last_name"] if c in users.columns]
            users = users[name_cols].drop_duplicates(subset=["user_id"])
            users["mentor_name"] = (
                users.get("first_name", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
                + " "
                + users.get("last_name", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
            ).str.strip()
            users.loc[users["mentor_name"] == "", "mentor_name"] = "Mentor"
            out["mentor_id"] = pd.to_numeric(out["mentor_id"], errors="coerce").astype("Int64")
            users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce").astype("Int64")
            out = out.merge(users[["user_id", "mentor_name"]], left_on="mentor_id", right_on="user_id", how="left")
            out.drop(columns=[c for c in ["user_id"] if c in out.columns], inplace=True)

        domains = processed.get("domains", pd.DataFrame()).copy()
        if not domains.empty and {"domain_id", "name"}.issubset(domains.columns):
            out = out.merge(domains[["domain_id", "name"]], on="domain_id", how="left")
            out.rename(columns={"name": "domain"}, inplace=True)

        # If domain names are missing, try to resolve from subdomains table (fallback)
        if ("domain" not in out.columns or out["domain"].isna().all()):
            subdomains = processed.get("subdomains", pd.DataFrame()).copy()
            if not subdomains.empty and {"subdomain_id", "name"}.issubset(subdomains.columns) and "subdomain_id" in out.columns:
                sub_map = subdomains[["subdomain_id", "name"]].drop_duplicates(subset=["subdomain_id"]).rename(columns={"name": "subdomain_name"})
                out = out.merge(sub_map, on="subdomain_id", how="left")
                # Prefer domain from merged domains; otherwise use subdomain name
                out["domain"] = out.get("domain").fillna(out.get("subdomain_name")).fillna("")
                if "subdomain_name" in out.columns:
                    out.drop(columns=["subdomain_name"], inplace=True)

        # As a last resort, if domain still missing but domain_id present, render a readable fallback
        if "domain" not in out.columns:
            out["domain"] = ""
        out["domain"] = out["domain"].fillna("")
        if (out["domain"] == "").all() and "domain_id" in out.columns:
            out.loc[out["domain"] == "", "domain"] = out.loc[out["domain"] == "", "domain_id"].apply(lambda v: f"Domain {int(v)}" if pd.notna(v) else "")

        if "mentor_name" not in out.columns:
            out["mentor_name"] = "Mentor"
        if "domain" not in out.columns:
            out["domain"] = ""
        if "title" not in out.columns:
            out["title"] = ""
        if "target_level" not in out.columns:
            out["target_level"] = ""
        if "education_level" not in out.columns:
            out["education_level"] = ""

        # No reason text in direct API output — chatbot builds explanation on demand
        # from signal fields (target_level_pass, education_level_pass, etc.)

        keep_cols = [
            c for c in [
                "post_id",
                "mentor_id",
                "mentor_name",
                "title",
                "domain",
                "target_level",
                "education_level",
                "score",
                "pred_score",
                "match_percentage",
                "minimum_requirement_exact_match",
                "minimum_requirement_above_minimum",
                "target_level_pass",
                "education_level_pass",
                "availability_pass",
                "requirement_coverage_score",
                "required_skill_level_match_score",
            ]
            if c in out.columns
        ]
        result = out[keep_cols].copy()

        if "score" not in result.columns and "pred_score" in result.columns:
            result["score"] = result["pred_score"]

        for id_col in ["post_id", "mentor_id"]:
            if id_col in result.columns:
                result[id_col] = result[id_col].map(lambda v: None if pd.isna(v) else str(v))

        # Coerce pandas / numpy scalar types to native Python types for JSON/Pydantic
        def _to_native(v):
            try:
                # numpy scalar -> native
                if hasattr(v, "item"):
                    return v.item()
            except Exception:
                pass
            # pandas NA -> None
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            return v

        for col in result.columns:
            result[col] = result[col].map(_to_native)

        return result

    @staticmethod
    def _build_reason(row: pd.Series) -> str:
        """Build a SHORT program-fit explanation focused on eligibility + domain.

        Program explanation should NOT sound like mentor explanation.
        Focus on: level fit, eligibility, domain/subdomain, skill coverage.
        """
        parts: list[str] = []

        # 1. Level fit (most important for programs)
        if bool(row.get("target_level_pass", False)):
            if bool(row.get("minimum_requirement_exact_match", False)):
                parts.append("exact level match")
            elif bool(row.get("minimum_requirement_above_minimum", False)):
                parts.append("exceeds minimum level")

        # 2. Education fit
        if bool(row.get("education_level_pass", False)):
            parts.append("education qualified")

        # 3. Skill coverage
        coverage = float(row.get("requirement_coverage_score", 0) or 0)
        if coverage >= 0.8:
            parts.append(f"{coverage:.0%} skill coverage")
        elif coverage >= 0.5:
            parts.append(f"{coverage:.0%} skill match")

        # 4. Domain context (if available)
        domain = row.get("domain", "")
        if domain and domain != "General" and str(domain).strip():
            parts.append(f"in {domain}")

        if not parts:
            return "Matches your profile."
        return "; ".join(parts).capitalize() + "."


program_recommendation_service = ProgramRecommendationService()