"""
Sentiment analysis endpoints.
All endpoints are async — FastAPI runs them in async event loop
while CPU-heavy inference happens in a thread pool (run_in_executor).
"""

import time
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.security import CurrentUser
from app.schemas.sentiment import (
    ModelName,
    SentimentRequest,
    SentimentResponse,
)
from app.services.model_service import ModelService

log = structlog.get_logger()
router = APIRouter()


def get_model_service(request: Request) -> ModelService:
    """Dependency injection — pulls the singleton from app state."""
    return request.app.state.model_service


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


@router.post(
    "/predict",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse sentiment of a single tweet",
    description=(
        "Returns sentiment label (positive / negative / neutral / irrelevant), "
        "confidence score, per-class probability distribution, and optional "
        "SHAP token-level explanations."
    ),
    responses={
        200: {"description": "Prediction successful"},
        422: {"description": "Validation error — check request body"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_sentiment(
    payload: SentimentRequest,
    model_service: ModelServiceDep,
    current_user: CurrentUser,
) -> SentimentResponse:
    """
    **Example request:**
    ```json
    {
      "text": "Apple just dropped the best product ever!",
      "model": "distilbert",
      "explain": false
    }
    ```
    """
    log.info(
        "predict.request",
        user=current_user.sub,
        model=payload.model,
        text_length=len(payload.text),
        explain=payload.explain,
    )

    start = time.perf_counter()

    try:
        result = await model_service.predict(
            text=payload.text,
            model_name=payload.model,
            explain=payload.explain,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        log.error("predict.failed", error=str(e))
        raise HTTPException(status_code=503, detail="Model inference failed")

    elapsed_ms = (time.perf_counter() - start) * 1000

    log.info(
        "predict.complete",
        sentiment=result["sentiment"],
        confidence=round(result["confidence"], 3),
        latency_ms=round(elapsed_ms, 1),
    )

    return SentimentResponse(
        text=payload.text,
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_used=payload.model,
        processing_time_ms=round(elapsed_ms, 2),
        explanation=result.get("explanation"),
    )


@router.post(
    "/predict/public",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Public predict endpoint (no auth — for demo/testing only)",
    description="Rate-limited unauthenticated endpoint for demos. Do not use in production.",
    include_in_schema=True,
)
async def predict_sentiment_public(
    payload: SentimentRequest,
    model_service: ModelServiceDep,
) -> SentimentResponse:
    """No auth required. Rate-limited to 10 req/min via middleware."""
    # Force default model, disable explain for public endpoint
    payload.model = ModelName.DISTILBERT
    payload.explain = False

    start = time.perf_counter()
    try:
        result = await model_service.predict(
            text=payload.text,
            model_name=payload.model,
            explain=False,
        )
    except Exception as e:
        log.error("predict_public.failed", error=str(e))
        raise HTTPException(status_code=503, detail="Inference failed")

    elapsed_ms = (time.perf_counter() - start) * 1000
    return SentimentResponse(
        text=payload.text,
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_used=payload.model,
        processing_time_ms=round(elapsed_ms, 2),
    )
