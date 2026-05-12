"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from api.main import app, load_model


@pytest.fixture
def client():
    """Create test client."""
    load_model()  # Ensure model is loaded before tests
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info(client):
    """Test model info endpoint."""
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert data["model_name"] == "ds_model"


def test_predict_endpoint(client):
    """Test prediction endpoint."""
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "timestamp" in data


def test_predict_batch_endpoint(client):
    """Test batch prediction endpoint."""
    response = client.post(
        "/predict_batch",
        json={"features": [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert len(data["probabilities"]) == 2
