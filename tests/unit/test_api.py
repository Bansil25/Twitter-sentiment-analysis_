"""
Unit tests for the sentiment API.
Uses FastAPI's TestClient — no real models needed (mock the service layer).
Run: pytest tests/ -v --asyncio-mode=auto
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def mock_model_service():
    """Returns a MagicMock that behaves like a loaded ModelService."""
    service = MagicMock()
    service.loaded_models = ["distilbert", "bilstm"]
    service.predict = AsyncMock(return_value={
        "sentiment": "positive",
        "confidence": 0.94,
        "probabilities": {
            "positive": 0.94, "negative": 0.03,
            "neutral": 0.02, "irrelevant": 0.01,
        },
        "explanation": None,
    })
    service.batch_predict = AsyncMock(return_value=[
        {"text": "Great!", "sentiment": "positive", "confidence": 0.92,
         "probabilities": {"positive": 0.92, "negative": 0.04, "neutral": 0.03, "irrelevant": 0.01}},
        {"text": "Terrible!", "sentiment": "negative", "confidence": 0.88,
         "probabilities": {"positive": 0.04, "negative": 0.88, "neutral": 0.06, "irrelevant": 0.02}},
    ])
    return service


@pytest.fixture
def app(mock_model_service):
    """Create the FastAPI app with mocked model service."""
    from app.main import create_app
    application = create_app()
    application.state.model_service = mock_model_service
    return application


@pytest.fixture
def client(app):
    return TestClient(app)


def _auth_header(client):
    """Get a valid JWT token for test requests."""
    from app.core.security import create_access_token
    token = create_access_token(subject="test_user", scopes=["predict"])
    return {"Authorization": f"Bearer {token.access_token}"}


# ── Health tests ───────────────────────────────────────────────────────────

def test_health_returns_200(client, mock_model_service):
    with patch("app.api.v1.routes.health._check_redis", new_callable=AsyncMock, return_value=True):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "version" in data
    assert "uptime_seconds" in data


def test_readiness_with_models_loaded(client):
    response = client.get("/health/ready")
    assert response.status_code == 200


# ── Sentiment prediction tests ─────────────────────────────────────────────

def test_predict_requires_auth(client):
    response = client.post("/api/v1/predict", json={"text": "Hello"})
    assert response.status_code == 401


def test_predict_returns_sentiment(client, mock_model_service):
    headers = _auth_header(client)
    response = client.post(
        "/api/v1/predict",
        json={"text": "Apple's new product is amazing!", "model": "distilbert"},
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert 0 <= data["confidence"] <= 1
    assert "probabilities" in data
    assert "processing_time_ms" in data
    assert data["model_used"] == "distilbert"


def test_predict_empty_text_fails_validation(client):
    headers = _auth_header(client)
    response = client.post(
        "/api/v1/predict",
        json={"text": ""},
        headers=headers,
    )
    assert response.status_code == 422


def test_predict_text_too_long_fails(client):
    headers = _auth_header(client)
    response = client.post(
        "/api/v1/predict",
        json={"text": "x" * 2001},
        headers=headers,
    )
    assert response.status_code == 422


def test_public_predict_no_auth_required(client):
    response = client.post(
        "/api/v1/predict/public",
        json={"text": "Testing the public endpoint"},
    )
    assert response.status_code == 200


# ── Batch prediction tests ─────────────────────────────────────────────────

def test_batch_predict(client, mock_model_service):
    headers = _auth_header(client)
    response = client.post(
        "/api/v1/batch/predict",
        json={"texts": ["Great product!", "Terrible waste of money."], "model": "distilbert"},
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2


def test_batch_empty_list_fails(client):
    headers = _auth_header(client)
    response = client.post(
        "/api/v1/batch/predict",
        json={"texts": []},
        headers=headers,
    )
    assert response.status_code == 422


# ── Preprocessing tests ────────────────────────────────────────────────────

def test_bilstm_cleaning_removes_urls():
    from ml.preprocessing.text_cleaner import clean_for_bilstm
    cleaned = clean_for_bilstm("Check this out https://example.com great stuff!")
    assert "http" not in cleaned
    assert "example.com" not in cleaned


def test_distilbert_cleaning_replaces_urls_with_token():
    from ml.preprocessing.text_cleaner import clean_for_distilbert
    cleaned = clean_for_distilbert("Check this out https://example.com great stuff!")
    assert "[URL]" in cleaned


def test_cleaning_removes_mentions():
    from ml.preprocessing.text_cleaner import clean_for_bilstm
    cleaned = clean_for_bilstm("@Apple thanks for the update!")
    assert "@Apple" not in cleaned
    assert "apple" not in cleaned.split()  # mention removed, not hashtag text


def test_cleaning_expands_slang():
    from ml.preprocessing.text_cleaner import clean_for_bilstm
    cleaned = clean_for_bilstm("tbh this is great")
    assert "to be honest" in cleaned


# ── Model info tests ───────────────────────────────────────────────────────

def test_model_list_returns_both_models(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    names = [m["name"] for m in data]
    assert "distilbert" in names
    assert "bilstm" in names
