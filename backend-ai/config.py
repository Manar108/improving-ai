from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_BASE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BASE_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[
            _BASE_DIR / ".env",
            _ROOT_DIR / ".env",
        ],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Mentorship Platform AI Assistant"
    APP_ENV: str = "development"
    API_PREFIX: str = "/api/v1"

    GOOGLE_API_KEY: str = ""
    CSE_ID: str = ""
    GROQ_API_KEY: str = ""
    MODEL_NAME: str = "llama-3.3-70b-versatile"

    RECOMMENDER_MODE: str = "model"  # "model" = direct inference, "db" = legacy query, "api" = external
    RECOMMENDER_API_BASE_URL: str = "http://localhost:8001"
    RECOMMENDER_API_PATH: str = "/recommend"
    RECOMMENDER_LOCAL_MODULE: str = "src.hybrid_recommender.pipeline"
    RECOMMENDER_LOCAL_FUNCTION: str = "predict_for_user"
    RECOMMENDER_ARTIFACTS_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "artifacts"

    DB_SERVER: str = "."
    DB_DATABASE: str = "MentorshipPlatformDB"
    DB_USERNAME: str = ""
    DB_PASSWORD: str = ""
    DB_DRIVER: str = "ODBC Driver 17 for SQL Server"
    DB_TRUSTED_CONNECTION: bool = True
    DB_TRUST_SERVER_CERTIFICATE: bool = True
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    DATA_DIR: Path = Path(__file__).resolve().parent / "data"
    FAQ_PATH: Path = DATA_DIR / "faq.json"
    MOCK_RECOMMENDATIONS_PATH: Path = DATA_DIR / "mock_recommendations.json"

    # Sentiment analysis model (BERT fine-tuned, 3 classes)
    SENTIMENT_MODEL_PATH: Path = Path(__file__).resolve().parents[2] / "sentiment_model"


settings = Settings()
