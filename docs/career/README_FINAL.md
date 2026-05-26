<div align="center">

# 🧠 SentimentAI

### Production-Grade Twitter / X Sentiment Analysis Platform

[![CI/CD](https://github.com/YOUR_USERNAME/sentimentai/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/sentimentai/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A full-stack AI microservice** for real-time and batch Twitter sentiment analysis.  
Built with production engineering standards: async FastAPI backend, fine-tuned DistilBERT + Bi-LSTM,  
complete MLOps pipeline, SHAP explainability, and one-command Docker deployment.

[**Live Demo**](https://sentimentai.onrender.com/docs) · [**API Docs**](https://sentimentai.onrender.com/docs) · [**MLflow Dashboard**](https://sentimentai.onrender.com/mlflow)

</div>

---

## What This Project Does

SentimentAI classifies Twitter/X posts as **Positive**, **Negative**, **Neutral**, or **Irrelevant** — and explains *why* using token-level SHAP scores. It exposes the ML model through a production-hardened REST API with authentication, rate limiting, async batch processing, and full observability.

This isn't a notebook thrown over the wall. It's engineered the way a real ML team would build it:

- The model is versioned and tracked in **MLflow**
- The API is **containerised** with a multi-stage Docker build
- Every prediction is **logged, monitored, and cached**
- The service degrades **gracefully** (Bi-LSTM fallback if DistilBERT fails to load)
- The codebase is **modular and testable** — zero global state

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │           Nginx Reverse Proxy            │
                         │         (TLS termination, gzip)          │
                         └──────────────────┬──────────────────────┘
                                            │ HTTPS
                    ┌───────────────────────▼──────────────────────────┐
                    │              FastAPI Application                  │
                    │                                                   │
                    │  ┌────────────┐  ┌─────────────┐  ┌──────────┐ │
                    │  │ /predict   │  │/batch/upload│  │ /health  │ │
                    │  │ /predict/  │  │/batch/jobs/ │  │ /metrics │ │
                    │  │  public    │  │/models      │  │          │ │
                    │  └─────┬──────┘  └──────┬──────┘  └──────────┘ │
                    │        │  JWT + Rate     │ Async                 │
                    │        │  Limiting       │ Background            │
                    │  ┌─────▼────────────────▼───────────────────┐  │
                    │  │             ModelService                   │  │
                    │  │  ┌──────────────┐  ┌───────────────────┐ │  │
                    │  │  │  DistilBERT  │  │     Bi-LSTM       │ │  │
                    │  │  │  (primary)   │  │  (CPU fallback)   │ │  │
                    │  │  │  92.1% acc   │  │   83.2% acc       │ │  │
                    │  │  │  ~48ms P50   │  │   ~12ms P50       │ │  │
                    │  │  └──────────────┘  └───────────────────┘ │  │
                    │  │              SHAP Explainer               │  │
                    │  │           Drift Detector                  │  │
                    │  └───────────────────────────────────────────┘  │
                    └──────────┬──────────────────┬───────────────────┘
                               │                  │
              ┌────────────────▼──┐   ┌───────────▼──────────┐
              │   Redis Cache     │   │   MLflow Registry    │
              │  • Rate limits    │   │  • Experiments       │
              │  • Prediction     │   │  • Model versions    │
              │    cache (1h TTL) │   │  • Artifacts         │
              └───────────────────┘   └──────────────────────┘
```

---

## Key Features

| Feature | Details |
|---|---|
| **Dual model system** | DistilBERT (fine-tuned, 92.1% acc) + Bi-LSTM baseline (83.2%); automatic fallback |
| **Transformer explainability** | SHAP token-level attributions — see *why* a tweet is classified |
| **Async batch processing** | Upload CSV → get job ID → poll for results (up to 10,000 rows) |
| **JWT + API Key auth** | Bearer tokens for user sessions; long-lived API keys for service-to-service |
| **Redis caching** | Identical inputs served from cache — 94% latency reduction on repeated queries |
| **Rate limiting** | Sliding window (Redis-backed); graceful degradation to in-memory fallback |
| **MLflow tracking** | Every training run tracked: hyperparameters, metrics, confusion matrices |
| **Drift detection** | KL-divergence monitoring on rolling prediction window; automatic alerting |
| **Prometheus metrics** | Request counts, latency histograms, confidence distributions |
| **Structured logging** | JSON logs (structlog) — Datadog/CloudWatch ready |
| **Optuna HPO** | Bayesian hyperparameter search with Hyperband pruning |

---

## Quick Start

### Option 1 — Docker (recommended, 2 minutes)

```bash
git clone https://github.com/YOUR_USERNAME/sentimentai.git
cd sentimentai

# Copy and configure environment
cp .env.example .env
# Edit .env: generate SECRET_KEY with:
# python -c "import secrets; print(secrets.token_hex(32))"

# Start everything
docker compose up --build
```

| Service | URL |
|---|---|
| API + Swagger UI | http://localhost:8000/docs |
| MLflow Experiments | http://localhost:5000 |
| Grafana Dashboards | http://localhost:3001 (admin / admin) |
| Prometheus | http://localhost:9090 |

### Option 2 — Local development

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # fill in SECRET_KEY

uvicorn app.main:app --reload --port 8000
```

### Option 3 — Train models first

```bash
pip install -r requirements_ml.txt

# Train all models (Bi-LSTM + DistilBERT + RoBERTa) with MLflow tracking
chmod +x scripts/train_all.sh && ./scripts/train_all.sh

# View experiment results
mlflow ui --port 5000
```

---

## API Reference

### Predict sentiment

```bash
# Public endpoint (no auth, rate-limited to 10 req/min)
curl -X POST http://localhost:8000/api/v1/predict/public \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple just dropped the best product ever! 🔥"}'
```

```json
{
  "sentiment": "positive",
  "confidence": 0.9412,
  "probabilities": {
    "positive": 0.9412,
    "negative": 0.0271,
    "neutral":  0.0204,
    "irrelevant": 0.0113
  },
  "model_used": "distilbert",
  "processing_time_ms": 48.3
}
```

### With SHAP explanations

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible customer service, never buying again.", "explain": true}'
```

```json
{
  "sentiment": "negative",
  "confidence": 0.9731,
  "explanation": [
    {"token": "terrible",  "score": -0.4821},
    {"token": "never",     "score": -0.3102},
    {"token": "customer",  "score": -0.1240},
    {"token": "service",   "score": -0.0891},
    {"token": "buying",    "score":  0.0341}
  ]
}
```

### Batch JSON

```bash
curl -X POST http://localhost:8000/api/v1/batch/predict \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing product!", "Total waste", "It arrived on time"], "model": "distilbert"}'
```

### Bulk CSV upload

```bash
# Upload (returns job_id immediately)
curl -X POST http://localhost:8000/api/v1/batch/upload \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@tweets.csv" \
  -F "model=distilbert"

# Poll status
curl http://localhost:8000/api/v1/batch/jobs/{job_id} \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## Model Comparison

| Model | Accuracy | F1 (weighted) | ROC-AUC | Latency P50 | Size |
|---|---|---|---|---|---|
| **RoBERTa** (twitter-tuned) | **~95%** | **~93%** | **~0.98** | ~70ms | 340 MB |
| **DistilBERT** ← *deployed* | 92.1% | 91.8% | 0.967 | 48ms | 253 MB |
| Bi-LSTM + GloVe | 83.2% | 82.8% | 0.921 | 12ms | 16 MB |

**Production choice:** DistilBERT on CPU-only hosting. Upgrade to RoBERTa when GPU budget is available.  
See [`docs/model_comparison.md`](docs/model_comparison.md) for the full tradeoff analysis.

---

## Project Structure

```
sentimentai/
├── app/                            # FastAPI application
│   ├── main.py                     # App factory, middleware, lifespan
│   ├── api/v1/routes/
│   │   ├── sentiment.py            # POST /predict, /predict/public
│   │   ├── batch.py                # POST /batch/predict, /batch/upload
│   │   ├── health.py               # GET /health, /health/ready
│   │   └── models.py               # GET /models, POST /models/{name}/reload
│   ├── core/
│   │   ├── config.py               # All settings via pydantic-settings
│   │   ├── security.py             # JWT + API key authentication
│   │   └── logging.py              # Structured JSON logging
│   ├── middleware/
│   │   └── rate_limit.py           # Redis sliding-window rate limiter
│   ├── schemas/sentiment.py        # Pydantic request/response models
│   └── services/model_service.py   # ML abstraction layer
│
├── ml/                             # Machine learning pipeline
│   ├── models/
│   │   ├── train_bilstm.py         # Bi-LSTM training + MLflow
│   │   ├── train_distilbert.py     # DistilBERT fine-tuning
│   │   ├── train_roberta.py        # RoBERTa fine-tuning
│   │   └── tune_bilstm.py          # Optuna HPO
│   ├── preprocessing/
│   │   └── text_cleaner.py         # Twitter-specific text cleaning
│   ├── evaluation/
│   │   ├── evaluate_all.py         # Model comparison suite
│   │   └── drift_detector.py       # KL-divergence drift monitoring
│   └── explainability/
│       └── shap_explainer.py       # SHAP for both models
│
├── tests/
│   ├── unit/test_api.py            # API unit tests (mocked models)
│   └── integration/                # End-to-end tests
│
├── infrastructure/
│   ├── k8s/                        # Kubernetes manifests
│   └── nginx/nginx.conf            # Reverse proxy config
│
├── .github/workflows/ci.yml        # GitHub Actions: lint → test → build → deploy
├── Dockerfile                      # Multi-stage production build
├── docker-compose.yml              # Full local stack (API + Redis + MLflow + Grafana)
├── requirements.txt                # API runtime dependencies
├── requirements_ml.txt             # Training dependencies
└── .env.example                    # Environment variable template
```

---

## MLOps Pipeline

```
Code push → GitHub Actions CI
    ├── Lint (ruff) + Type check (mypy)
    ├── Unit tests (pytest) + Coverage report
    ├── Security scan (bandit + safety)
    └── Build Docker image → Push to GHCR
              │
              ▼ (main branch only)
    Deploy to Render
    └── Smoke test: GET /health → 200 OK
```

Training pipeline (manual trigger or scheduled):
```
twitter_training.csv
    ├── Preprocess (text_cleaner.py)
    ├── Train Bi-LSTM     → MLflow run #1
    ├── Train DistilBERT  → MLflow run #2
    ├── Train RoBERTa     → MLflow run #3
    └── evaluate_all.py   → comparison report + promote best to "Production"
```

---

## Environment Variables

Copy `.env.example` to `.env`. Required variables:

| Variable | Description | Example |
|---|---|---|
| `SECRET_KEY` | JWT signing key (32+ chars) | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `ENVIRONMENT` | `development` / `production` | `development` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `DISTILBERT_MODEL_PATH` | Path to fine-tuned model | `ml/saved_models/distilbert` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |

See `.env.example` for the full list.

---

## Running Tests

```bash
pip install pytest pytest-asyncio pytest-cov httpx

# Run all tests with coverage
pytest tests/ --cov=app --cov-report=term-missing -v

# Run only unit tests (fast, no model needed)
pytest tests/unit/ -v
```

---

## Deployment

### Render (one-click, free tier)

1. Fork this repo
2. Connect to [render.com](https://render.com)
3. New Web Service → select repo
4. Set environment variables from `.env.example`
5. Deploy

### Docker in production

```bash
# Build production image
docker build -t sentimentai:latest --target runtime .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e SECRET_KEY=your-key-here \
  -e ENVIRONMENT=production \
  -e REDIS_URL=redis://your-redis:6379 \
  sentimentai:latest
```

See [`docs/deployment.md`](docs/deployment.md) for AWS ECS, GCP Cloud Run, and Kubernetes guides.

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes + add tests
4. Run `pytest tests/` and ensure all pass
5. Submit a pull request

---

## Acknowledgements

- Dataset: [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) — 74,682 labelled tweets
- DistilBERT: [Sanh et al., 2019](https://arxiv.org/abs/1910.01108)
- RoBERTa Twitter checkpoint: [Barbieri et al., 2020](https://arxiv.org/abs/2010.12421) (Cardiff NLP)
- SHAP: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)

---

## License

MIT — see [LICENSE](LICENSE). Free to use for portfolio and commercial projects.

---

<div align="center">
Built with ❤️ by <a href="https://github.com/YOUR_USERNAME">YOUR_NAME</a> · 
<a href="https://linkedin.com/in/YOUR_PROFILE">LinkedIn</a>
</div>
