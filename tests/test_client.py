"""
tests/test_client.py
--------------------
Tests for the Python client (clients/python) against the real API, run
in-process through FastAPI's TestClient — the same requests a developer's
code would make over the network.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.v1 as v1
from api.auth import get_accounts, request_limiter
from api.main import app
from churn_engine import (
    AuthenticationError, ChurnEngine, NotFoundError, RateLimitError, TrainingFailedError, ValidationError,
)
from src.accounts import TIERS, AccountStore
from src.registry import Registry
from src.sample_data import make_customers


@pytest.fixture
def accounts(tmp_path):
    registry = Registry(tmp_path / "storage")
    store = AccountStore(tmp_path / "storage" / "accounts.db")
    app.dependency_overrides[v1.get_registry] = lambda: registry
    app.dependency_overrides[get_accounts] = lambda: store
    request_limiter.reset()
    yield store
    app.dependency_overrides.clear()


def make_client(accounts, tier="standard", **kwargs) -> ChurnEngine:
    _, key = accounts.create_account("Dev", tier=tier)
    return ChurnEngine(api_key=key, http_client=TestClient(app), **kwargs)


@pytest.fixture
def client(accounts):
    return make_client(accounts)


@pytest.fixture
def model(client):
    return client.train(make_customers(1500, seed=3), poll_interval=0)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("CHURN_ENGINE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key"):
        ChurnEngine()


def test_train_from_dataframe_detects_target(model):
    assert model.ready
    assert model.target == "churned"
    assert model.auc > 0.7
    assert "tenure_months" in model.input_columns


def test_train_from_path(client, tmp_path):
    path = tmp_path / "customers.csv"
    make_customers(1500, seed=4).to_csv(path, index=False)
    model = client.train(path, target="churned", name="from file", poll_interval=0)
    assert (model.ready, model.name) == (True, "from file")


def test_predict_records_and_dataframe(client, model):
    df = make_customers(5, seed=8, labelled=False)
    from_df = client.predict(model.id, df)
    from_dicts = client.predict(model.id, df.to_dict(orient="records"))
    assert from_df == from_dicts
    assert [r["id"] for r in from_df] == df["customer_id"].tolist()


def test_predict_splits_large_batches(client, model):
    results = client.predict(model.id, make_customers(2300, seed=8, labelled=False))
    assert len(results) == 2300


def test_score_returns_dataframe_and_writes_file(client, model, tmp_path):
    out = tmp_path / "scored.csv"
    scored = client.score(model.id, make_customers(30, seed=9, labelled=False), output=out)
    assert isinstance(scored, pd.DataFrame) and len(scored) == 30
    assert "risk_tier" in scored.columns
    assert pd.read_csv(out).shape == scored.shape


def test_list_get_delete(client, model):
    assert [m["id"] for m in client.list_models()] == [model.id]
    client.delete_model(model.id)
    with pytest.raises(NotFoundError) as e:
        client.get_model(model.id)
    assert e.value.code == "model_not_found"


def test_account(client, model):
    assert client.account()["usage"]["models"] == 1


def test_bad_key_raises_authentication_error(accounts):
    client = ChurnEngine(api_key="ck_wrong", http_client=TestClient(app))
    with pytest.raises(AuthenticationError) as e:
        client.list_models()
    assert e.value.code == "invalid_api_key"


def test_bad_target_raises_validation_error(client):
    with pytest.raises(ValidationError) as e:
        client.train(make_customers(300, seed=4), target="region")
    assert e.value.code == "invalid_target"


def test_failed_training_raises(client, monkeypatch):
    from src.engine import TrainingError

    def fail(*args, **kwargs):
        raise TrainingError("not enough signal")
    monkeypatch.setattr(v1, "train_pipeline", fail)
    with pytest.raises(TrainingFailedError, match="not enough signal"):
        client.train(make_customers(300, seed=4), poll_interval=0)


def test_rate_limit_retries_then_raises(accounts, monkeypatch):
    monkeypatch.setitem(TIERS, "free", {**TIERS["free"], "requests_per_minute": 1})
    client = make_client(accounts, tier="free", max_retries=2)
    sleeps = []
    monkeypatch.setattr("churn_engine.client.time.sleep", sleeps.append)

    client.account()
    with pytest.raises(RateLimitError) as e:
        client.account()
    assert e.value.retry_after >= 1
    assert len(sleeps) == 2, "should retry max_retries times, honouring Retry-After"
