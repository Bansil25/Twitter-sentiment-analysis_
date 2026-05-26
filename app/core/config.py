"""
Centralised configuration — all settings come from environment variables.
Never hardcode secrets. Use .env for local dev, proper secrets manager in prod.
"""

from functools import lru_cache
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    APP_NAME: str = "SentimentAI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    API_VERSION: str = "v1"
    ENABLE_DOCS: bool = True  # Disable in prod via env var

    # ── Server ─────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1  # Increase for production (2–4 × CPU cores)
    RELOAD: bool = False

    # ── Security ───────────────────────────────────────────────────────────
    SECRET_KEY: str = Field(..., min_length=32)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 60
    API_KEY_HEADER: str = "X-API-Key"

    # ── CORS ───────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    # ── Redis ──────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600  # 1 hour cache for identical inputs

    # ── Rate limiting ──────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── ML Model paths ─────────────────────────────────────────────────────
    MODEL_DIR: str = "ml/saved_models"
    BILSTM_MODEL_PATH: str = "ml/saved_models/bilstm/model.keras"
    BILSTM_TOKENIZER_PATH: str = "ml/saved_models/bilstm/tokenizer.pkl"
    DISTILBERT_MODEL_PATH: str = "ml/saved_models/distilbert"  # HuggingFace dir
    DEFAULT_MODEL: Literal["bilstm", "distilbert"] = "distilbert"

    # ── MLflow ─────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "sentiment-analysis"

    # ── Inference ─────────────────────────────────────────────────────────
    MAX_SEQUENCE_LENGTH: int = 128
    BATCH_SIZE: int = 32
    MAX_BATCH_CSV_ROWS: int = 10_000
    INFERENCE_TIMEOUT_SECONDS: float = 10.0

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"  # json in prod, console in dev

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings — read once, reuse everywhere."""
    return Settings()


# Module-level singleton for convenient imports: `from app.core.config import settings`
settings = get_settings()
