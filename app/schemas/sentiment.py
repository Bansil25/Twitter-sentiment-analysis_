"""
Pydantic v2 schemas for all API request/response models.
Strong typing = self-documenting API + free input validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    IRRELEVANT = "irrelevant"


class ModelName(str, Enum):
    BILSTM = "bilstm"
    DISTILBERT = "distilbert"


# ── Sentiment prediction ───────────────────────────────────────────────────

class SentimentRequest(BaseModel):
    """Single tweet sentiment prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Tweet text to analyse",
        examples=["Apple just dropped the best product ever. I'm buying it today!"],
    )
    model: ModelName = Field(
        ModelName.DISTILBERT,
        description="Model to use for inference. distilbert is recommended for production.",
    )
    explain: bool = Field(
        False,
        description="Return SHAP token-level explanations. Increases latency ~300ms.",
    )

    @field_validator("text")
    @classmethod
    def sanitise_text(cls, v: str) -> str:
        return v.strip()


class TokenExplanation(BaseModel):
    token: str
    score: float = Field(description="SHAP value — positive pushes toward predicted class")


class SentimentResponse(BaseModel):
    """Sentiment prediction result."""

    text: str
    sentiment: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    probabilities: Dict[str, float] = Field(
        description="Per-class probability distribution. Sums to 1.0."
    )
    model_used: ModelName
    processing_time_ms: float
    explanation: Optional[List[TokenExplanation]] = Field(
        None,
        description="SHAP token explanations. Only present when explain=true.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "text": "Apple just dropped the best product ever!",
            "sentiment": "positive",
            "confidence": 0.94,
            "probabilities": {"positive": 0.94, "negative": 0.03, "neutral": 0.02, "irrelevant": 0.01},
            "model_used": "distilbert",
            "processing_time_ms": 48.2,
            "explanation": None,
        }
    }}


# ── Batch prediction ───────────────────────────────────────────────────────

class BatchSentimentRequest(BaseModel):
    """Batch of tweets for bulk prediction (JSON body, up to 512 items)."""

    texts: List[str] = Field(..., min_length=1, max_length=512)
    model: ModelName = ModelName.DISTILBERT

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty text is required")
        return cleaned


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total: int
    model_used: ModelName
    total_processing_time_ms: float


# ── CSV batch upload ───────────────────────────────────────────────────────

class CSVBatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class CSVBatchJobResponse(BaseModel):
    job_id: str
    status: CSVBatchStatus
    message: str
    rows_total: Optional[int] = None
    rows_processed: Optional[int] = None
    download_url: Optional[str] = None


# ── Model info ─────────────────────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    name: ModelName
    version: str
    description: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    loaded: bool
    parameters: Optional[int] = None


# ── Health ─────────────────────────────────────────────────────────────────

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    status: HealthStatus
    version: str
    environment: str
    models_loaded: List[str]
    redis_connected: bool
    uptime_seconds: float


# ── Errors ─────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    type: str
    field: Optional[str] = None

    model_config = {"json_schema_extra": {
        "example": {"detail": "Text must be at least 1 character", "type": "ValidationError", "field": "text"}
    }}
