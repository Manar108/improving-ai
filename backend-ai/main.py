import logging
import os

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse
# pyrefly: ignore [missing-import]
from fastapi.exceptions import RequestValidationError                   

from config import settings
from routes.chat import router as chat_router
from routes.mentor_chat import router as mentor_chat_router
from routes.mentor_analytics import router as mentor_analytics_router
from routes.insights import router as insights_router
from routes.materials import router as materials_router
from routes.recommend import router as recommend_router
from routes.program_recommend import router as program_recommend_router
from routes.sentiment import router as sentiment_router
from database.db import database


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME, version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix=settings.API_PREFIX)
app.include_router(mentor_chat_router, prefix=settings.API_PREFIX)
app.include_router(mentor_analytics_router, prefix=settings.API_PREFIX)
app.include_router(recommend_router, prefix=settings.API_PREFIX)
app.include_router(program_recommend_router, prefix=settings.API_PREFIX)
app.include_router(materials_router, prefix=settings.API_PREFIX)
app.include_router(insights_router, prefix=settings.API_PREFIX)
app.include_router(sentiment_router, prefix=settings.API_PREFIX)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": settings.APP_NAME}


@app.get("/db-health")
def db_health() -> dict:
    """Verify database connectivity, table availability, and row counts.

    Returns a full diagnostic report including:
    - connection status
    - list of discovered tables with row counts
    - list of missing tables
    - any errors encountered
    """
    report = database.health_check()
    report["summary"] = {
        "total_tables_checked": len(report.get("tables", {})) + len(report.get("missing_tables", [])),
        "tables_found": len(report.get("tables", {})),
        "tables_missing": len(report.get("missing_tables", [])),
        "total_rows": sum(report.get("tables", {}).values()),
    }
    return report


@app.on_event("startup")
async def startup_event():
    """Log database connectivity on startup."""
    logger.info("Starting %s v2.0.0", settings.APP_NAME)
    try:
        check = database.health_check()
        if check["connected"]:
            table_count = len(check.get("tables", {}))
            missing_count = len(check.get("missing_tables", []))
            total_rows = sum(check.get("tables", {}).values())
            logger.info(
                "Database connected: %s/%s | Tables: %d found, %d missing | Total rows: %d",
                settings.DB_SERVER, settings.DB_DATABASE,
                table_count, missing_count, total_rows,
            )
            if check["missing_tables"]:
                logger.warning("Missing tables: %s", check["missing_tables"])
        else:
            logger.error("Database connection FAILED: %s", check.get("errors"))
    except Exception as exc:
        logger.error("Startup DB check failed: %s", exc)
