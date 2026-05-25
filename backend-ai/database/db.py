from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class DatabaseAccessError(RuntimeError):
    """Raised when the database cannot be reached or queried."""


class MissingTableError(DatabaseAccessError):
    """Raised when a required table is missing from the database."""


@dataclass
class QueryResult:
    frame: pd.DataFrame
    error: str | None = None


# ---------------------------------------------------------------------------
# All tables defined in the ASP.NET EF ApplicationDbContext
# ---------------------------------------------------------------------------
REQUIRED_TABLES = [
    "users",
    "mentor_profile",
    "mentee_profile",
    "countries",
    "domains",
    "subdomain",
    "technologies",
    "career_goal",
    "learning_style",
    "mentee_interests",
    "mentor_expertise",
    "programs",
    "mentorships",
    "applications",
    "feedbacks",
    "follows",
    "post_likes",
    "Post-Comment",
    "saved_posts",
    "shared_posts",
    "mentorship_requirements",
    "apps_cancellation",
    "mentorships_cancellation",
    "MenteeSubDomains",
    "MentorSubDomains",
]


class DatabaseClient:
    """Reusable SQL Server client for the AI assistant backend.

    Connects using environment variables via ``config.settings``.
    Supports both SQL-auth and Windows-auth (Trusted Connection).
    """

    def __init__(self) -> None:
        self._engine: Engine | None = None

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------

    def get_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(
                self._build_connection_url(),
                pool_pre_ping=True,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                future=True,
            )
        return self._engine

    def _build_connection_url(self) -> str:
        driver = urllib.parse.quote_plus(settings.DB_DRIVER)
        trust_cert = "yes" if settings.DB_TRUST_SERVER_CERTIFICATE else "no"

        if settings.DB_USERNAME and settings.DB_PASSWORD:
            username = urllib.parse.quote_plus(settings.DB_USERNAME)
            password = urllib.parse.quote_plus(settings.DB_PASSWORD)
            return (
                f"mssql+pyodbc://{username}:{password}@{settings.DB_SERVER}/{settings.DB_DATABASE}"
                f"?driver={driver}&TrustServerCertificate={trust_cert}"
            )

        trusted = "yes" if settings.DB_TRUSTED_CONNECTION else "no"
        return (
            f"mssql+pyodbc://@{settings.DB_SERVER}/{settings.DB_DATABASE}"
            f"?driver={driver}&trusted_connection={trusted}&TrustServerCertificate={trust_cert}"
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def run_query_df(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Execute a SQL query and return a pandas DataFrame."""
        try:
            with self.get_engine().connect() as connection:
                return pd.read_sql(text(query), connection, params=params or {})
        except SQLAlchemyError as exc:
            message = str(exc)
            logger.exception("Database query failed")
            if "Invalid object name" in message:
                raise MissingTableError(message) from exc
            raise DatabaseAccessError(message) from exc

    def run_scalar(self, query: str, params: dict[str, Any] | None = None) -> int:
        """Execute a SQL query and return a single integer value."""
        try:
            with self.get_engine().connect() as connection:
                value = connection.execute(text(query), params or {}).scalar()
            return int(value or 0)
        except SQLAlchemyError as exc:
            message = str(exc)
            logger.exception("Database scalar query failed")
            if "Invalid object name" in message:
                raise MissingTableError(message) from exc
            raise DatabaseAccessError(message) from exc

    def run_scalar_float(self, query: str, params: dict[str, Any] | None = None) -> float:
        """Execute a SQL query and return a single float value."""
        try:
            with self.get_engine().connect() as connection:
                value = connection.execute(text(query), params or {}).scalar()
            return round(float(value or 0.0), 2)
        except SQLAlchemyError as exc:
            message = str(exc)
            logger.exception("Database scalar float query failed")
            if "Invalid object name" in message:
                raise MissingTableError(message) from exc
            raise DatabaseAccessError(message) from exc

    def run_single_value(self, query: str, params: dict[str, Any] | None = None) -> str | None:
        """Execute a SQL query and return a single string value, or None."""
        try:
            with self.get_engine().connect() as connection:
                value = connection.execute(text(query), params or {}).scalar()
            return str(value) if value is not None else None
        except SQLAlchemyError as exc:
            message = str(exc)
            logger.exception("Database single-value query failed")
            if "Invalid object name" in message:
                raise MissingTableError(message) from exc
            raise DatabaseAccessError(message) from exc

    def table_exists(self, table_name: str) -> bool:
        """Check whether a table exists in the database."""
        query = """
        SELECT 1
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME = :table_name
        """
        try:
            return self.run_scalar(query, {"table_name": table_name}) == 1
        except DatabaseAccessError:
            return False

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        """Return database connectivity status, discovered tables, and row counts."""
        result: dict[str, Any] = {
            "connected": False,
            "server": settings.DB_SERVER,
            "database": settings.DB_DATABASE,
            "tables": {},
            "missing_tables": [],
            "errors": [],
        }

        try:
            with self.get_engine().connect() as connection:
                connection.execute(text("SELECT 1"))
            result["connected"] = True
        except SQLAlchemyError as exc:
            result["errors"].append(f"Connection failed: {exc}")
            return result

        for table in REQUIRED_TABLES:
            try:
                # Bracket-quote to handle special characters like hyphens
                quoted = table if table.startswith("[") else f"[{table}]"
                count = self.run_scalar(f"SELECT COUNT(1) FROM {quoted}")
                result["tables"][table] = count
            except MissingTableError:
                result["missing_tables"].append(table)
            except DatabaseAccessError as exc:
                result["errors"].append(f"Error reading {table}: {exc}")

        return result


database = DatabaseClient()
