"""
tests/test_api.py
-----------------
Unit tests for the FastAPI prediction endpoints.
Uses httpx TestClient — no server needed.

Run:
    pytest tests/ -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_predictor():
    """Return a mock ChurnPredictor that doesn't need model files."""
    mock = MagicMock()
    mock.predict.return_value = {
        "churn_probability":  0.74,
        "churn_prediction":   1,
        "risk_tier":          "High Risk",
        "recommended_action": "Priority outreach: device upgrade offer.",
    }
    return mock


@pytest.fixture
def client(mock_predictor):
    """Create a test client with the mock predictor injected."""
    with patch("api.main.predictor", mock_predictor):
        from api.main import app
        return TestClient(app)


# ── Sample payload ─────────────────────────────────────────────────────────────

SAMPLE_CUSTOMER = {
    "months":        18,
    "eqpdays":       480,
    "rev_Mean":      55.2,
    "mou_Mean":      320.0,
    "change_mou":   -120.0,
    "change_rev":    -8.5,
    "totmrc_Mean":   42.0,
    "ovrmou_Mean":    0.0,
    "drop_vce_Mean":  3.2,
    "plcd_vce_Mean": 85.0,
    "comp_vce_Mean": 80.5,
    "custcare_Mean":  2.1,
}


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_health_check(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_predict_returns_200(client):
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200


def test_predict_response_schema(client):
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    data = resp.json()
    assert "churn_probability"  in data
    assert "churn_prediction"   in data
    assert "risk_tier"          in data
    assert "recommended_action" in data


def test_predict_probability_range(client):
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    prob = resp.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_binary_prediction(client):
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    pred = resp.json()["churn_prediction"]
    assert pred in (0, 1)


def test_predict_valid_risk_tier(client):
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    tier = resp.json()["risk_tier"]
    assert tier in ("Low Risk", "Medium Risk", "High Risk")


def test_batch_predict(client):
    batch = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    resp = client.post("/predict/batch", json=batch)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_predict_with_minimal_fields(client):
    """API should handle a sparse customer record without crashing."""
    minimal = {"months": 12, "rev_Mean": 45.0}
    resp = client.post("/predict", json=minimal)
    assert resp.status_code == 200


def test_model_info_endpoint(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model"     in data
    assert "risk_tiers" in data
