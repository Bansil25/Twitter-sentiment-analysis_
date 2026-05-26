"""
Model info endpoints — expose model metadata for the frontend dashboard.
"""

from typing import List

import structlog
from fastapi import APIRouter, Depends, Request

from app.core.security import CurrentUser
from app.schemas.sentiment import ModelInfoResponse, ModelName

log = structlog.get_logger()
router = APIRouter()

# Static model metadata — in production this comes from MLflow Model Registry
MODEL_METADATA = {
    ModelName.BILSTM: ModelInfoResponse(
        name=ModelName.BILSTM,
        version="1.0.0",
        description=(
            "Bidirectional LSTM trained on Sentiment140 + Twitter entity dataset. "
            "Custom embedding layer (50k vocab). Good baseline with low memory footprint."
        ),
        accuracy=0.832,
        f1_score=0.828,
        avg_latency_ms=12.0,
        parameters=4_200_000,
        loaded=False,
    ),
    ModelName.DISTILBERT: ModelInfoResponse(
        name=ModelName.DISTILBERT,
        version="1.0.0",
        description=(
            "Fine-tuned DistilBERT (distilbert-base-uncased) on Twitter sentiment data. "
            "66M parameters, 40% smaller than BERT-base, 60% faster inference. "
            "Recommended for production workloads."
        ),
        accuracy=0.921,
        f1_score=0.918,
        avg_latency_ms=48.0,
        parameters=66_000_000,
        loaded=False,
    ),
}


@router.get(
    "/models",
    response_model=List[ModelInfoResponse],
    summary="List all available models with metadata",
)
async def list_models(request: Request) -> List[ModelInfoResponse]:
    model_service = request.app.state.model_service
    loaded = set(model_service.loaded_models)

    result = []
    for name, info in MODEL_METADATA.items():
        info_dict = info.model_dump()
        info_dict["loaded"] = name in loaded
        result.append(ModelInfoResponse(**info_dict))
    return result


@router.get(
    "/models/{model_name}",
    response_model=ModelInfoResponse,
    summary="Get details for a specific model",
)
async def get_model_info(
    model_name: ModelName,
    request: Request,
) -> ModelInfoResponse:
    model_service = request.app.state.model_service
    loaded = set(model_service.loaded_models)
    info = MODEL_METADATA[model_name]
    return ModelInfoResponse(**{**info.model_dump(), "loaded": model_name in loaded})


@router.post(
    "/models/{model_name}/reload",
    summary="Hot-reload a model (admin only)",
)
async def reload_model(
    model_name: ModelName,
    request: Request,
    current_user: CurrentUser,
) -> dict:
    """
    Reload a model from disk without restarting the server.
    Useful after deploying a new model artifact.
    """
    if "admin" not in current_user.scopes:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Admin scope required")

    model_service = request.app.state.model_service
    await model_service.reload(model_name)
    log.info("model.reloaded", model=model_name, user=current_user.sub)
    return {"status": "reloaded", "model": model_name}
