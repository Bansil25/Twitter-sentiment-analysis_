"""
SentimentAI — Production FastAPI Application
Entry point for the Twitter Sentiment Analysis API.
"""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.api.v1.routes import sentiment, health, models, batch
from app.core.config import settings
from app.core.logging import configure_logging
from app.middleware.rate_limit import RateLimitMiddleware

# ── Logging ────────────────────────────────────────────────────────────────
configure_logging()
log = structlog.get_logger()

# ── Prometheus metrics ─────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "sentimentai_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "sentimentai_request_latency_seconds",
    "API request latency",
    ["endpoint"],
)
PREDICTION_COUNT = Counter(
    "sentimentai_predictions_total",
    "Total predictions made",
    ["model", "sentiment"],
)


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup; release resources on shutdown."""
    log.info("startup.begin", env=settings.ENVIRONMENT, version=settings.APP_VERSION)

    from app.services.model_service import ModelService
    app.state.model_service = ModelService()
    await app.state.model_service.load()

    log.info("startup.complete", models_loaded=app.state.model_service.loaded_models)
    yield

    log.info("shutdown.begin")
    await app.state.model_service.unload()
    log.info("shutdown.complete")


# ── Application factory ────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="SentimentAI",
        description=(
            "Production-grade Twitter/X Sentiment Analysis API. "
            "Supports single-tweet inference, batch CSV processing, "
            "model comparison (Bi-LSTM vs DistilBERT), and explainability."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
        lifespan=lifespan,
    )

    # ── Middleware stack (order matters — outermost first) ─────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RateLimitMiddleware)

    # ── Request timing + metrics middleware ────────────────────────────────
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)

        response.headers["X-Process-Time"] = f"{duration:.4f}"
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "")
        return response

    # ── Routers ────────────────────────────────────────────────────────────
    prefix = f"/api/{settings.API_VERSION}"
    app.include_router(health.router, tags=["Health"])
    app.include_router(sentiment.router, prefix=prefix, tags=["Sentiment"])
    app.include_router(batch.router,    prefix=prefix, tags=["Batch"])
    app.include_router(models.router,   prefix=prefix, tags=["Models"])

    # ── Prometheus scrape endpoint ─────────────────────────────────────────
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ── Global exception handler ───────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        log.error(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": "UnhandledError"},
        )

    return app


app = create_app()
