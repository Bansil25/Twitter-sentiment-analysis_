"""
Health check endpoints — required by every load balancer and K8s probe.
/health       → liveness probe (is the process alive?)
/health/ready → readiness probe (is it ready to serve traffic?)
/health/live  → same as /health, explicit k8s alias
"""

import time
from typing import List

import structlog
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.schemas.sentiment import HealthResponse, HealthStatus, ModelInfoResponse

log = structlog.get_logger()
router = APIRouter()

_START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check (liveness probe)",
)
async def health_check(request: Request) -> HealthResponse:
    """
    Returns 200 if the service is alive.
    Used by Kubernetes liveness probes and load balancers.
    """
    model_service = getattr(request.app.state, "model_service", None)

    try:
        redis_ok = await _check_redis()
    except Exception:
        redis_ok = False

    loaded = model_service.loaded_models if model_service else []
    overall = HealthStatus.HEALTHY

    if not loaded:
        overall = HealthStatus.DEGRADED
    if not redis_ok and settings.REDIS_URL:
        overall = HealthStatus.DEGRADED

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        models_loaded=loaded,
        redis_connected=redis_ok,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.get(
    "/health/ready",
    tags=["Health"],
    summary="Readiness probe — is the service ready for traffic?",
)
async def readiness_check(request: Request):
    """
    Returns 200 only when at least one model is loaded.
    K8s will withhold traffic until this passes.
    """
    model_service = getattr(request.app.state, "model_service", None)
    if not model_service or not model_service.loaded_models:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "No models loaded"},
        )
    return {"status": "ready", "models": model_service.loaded_models}


@router.get(
    "/health/live",
    tags=["Health"],
    summary="Liveness probe alias",
    include_in_schema=False,
)
async def liveness():
    return {"status": "alive"}


async def _check_redis() -> bool:
    """Non-blocking Redis ping with timeout."""
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(settings.REDIS_URL, socket_connect_timeout=1)
        await client.ping()
        await client.aclose()
        return True
    except Exception:
        return False
