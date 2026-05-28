# =============================================================================
# Multi-stage Dockerfile for SentimentAI
# =============================================================================
# Teaching note: a "stage" is just a temporary image used to build the final.
# Stage 1 (builder) installs compilers and Python packages.
# Stage 2 (runtime) copies only the installed packages — no gcc, no pip cache.
# Result: ~40% smaller final image, much smaller attack surface.
# =============================================================================


# ─── STAGE 1: builder ────────────────────────────────────────────────────────
# Why "python:3.11-slim" not "python:3.11"?
#   - Non-slim is ~1GB. Slim is ~150MB. Same Python, fewer extras.
#   - Slim is missing build tools (gcc), so we install only what we need.

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps (only in this stage — they don't end up in runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements.txt first (not the whole codebase yet).
# Why? Docker caches each step. If code changes but requirements doesn't,
# Docker reuses this cached layer and skips reinstalling everything.
# This single trick turns 5-minute rebuilds into 10 seconds.
COPY requirements.txt .

# --prefix=/install routes everything to /install, easy to COPY later.
# --no-cache-dir prevents pip from saving downloaded wheels (~200MB saved).
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─── STAGE 2: runtime ────────────────────────────────────────────────────────
# This is what ships. Nothing from builder is included unless explicitly COPY'd.

FROM python:3.11-slim AS runtime

# Security: never run as root. If someone exploits a vuln in our app,
# they shouldn't get root privileges in the container.
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy the installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code with proper ownership in one step
COPY --chown=appuser:appuser app/      ./app/
COPY --chown=appuser:appuser ml/       ./ml/
COPY --chown=appuser:appuser scripts/  ./scripts/

RUN mkdir -p ml/saved_models logs && \
    chown -R appuser:appuser /app

USER appuser

# Environment defaults — override at runtime via -e or K8s ConfigMaps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    WORKERS=2

EXPOSE 8000

# HEALTHCHECK tells Docker (and K8s) if the app is alive.
# --start-period gives the model 60s to load before probes begin.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production server: Gunicorn (process manager) + Uvicorn (async ASGI workers).
# Why both? Gunicorn handles graceful restarts and worker management.
# Uvicorn is the actual async server FastAPI needs.
# Together: Uvicorn's speed + Gunicorn's robustness.
CMD ["sh", "-c", \
     "gunicorn app.main:app \
      --workers ${WORKERS} \
      --worker-class uvicorn.workers.UvicornWorker \
      --bind 0.0.0.0:${PORT} \
      --timeout 120 \
      --keep-alive 5 \
      --access-logfile - \
      --error-logfile - \
      --log-level info"]
