"""
tests/test_engine_api.py
------------------------
End-to-end tests for the bring-your-own-data API: upload a CSV, train a real
model, score customers. Storage goes to a temp directory; nothing is mocked.
"""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.engine import get_registry
from api.main import app
from src.registry import Registry
from src.sample_data import make_customers


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "storage")


@pytest.fixture
def client(registry):
    app.dependency_overrides[get_registry] = lambda: registry
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client, df, name="customers.csv"):
    return client.post("/datasets", files={"file": (name, _csv(df), "text/csv")})


@pytest.fixture
def trained_model(client):
    dataset = upload(client, make_customers(1500, seed=3)).json()
    # TestClient runs background tasks before returning, so training is done here
    resp = client.post("/models", json={"dataset_id": dataset["id"], "target": "churned"})
    assert resp.status_code == 202
    return client.get(f"/models/{resp.json()['id']}").json(), dataset


# ── Full flow ──────────────────────────────────────────────────────────────────

def test_upload_infers_schema(client):
    resp = upload(client, make_customers(300, seed=4))
    assert resp.status_code == 201
    data = resp.json()
    assert data["rows"] == 300
    assert data["suggested_target"] == "churned"
    assert data["suggested_id"] == "customer_id"
    assert data["target_values"] == ["No", "Yes"]


def test_train_produces_ready_model(trained_model):
    model, _ = trained_model
    assert model["status"] == "ready", model.get("error")
    report = model["report"]
    assert report["metrics"]["auc_roc"] > 0.7
    assert {s["name"] for s in report["input_schema"]} >= {"tenure_months", "contract"}


def test_dataset_deleted_after_successful_training(client, trained_model):
    _, dataset = trained_model
    assert client.get(f"/datasets/{dataset['id']}").status_code == 404


def test_predict_json(client, trained_model):
    model, _ = trained_model
    customers = make_customers(5, seed=8, labelled=False).to_dict(orient="records")
    resp = client.post(f"/models/{model['id']}/predict", json={"customers": customers})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 5
    assert data["missing_columns"] == []
    assert data["results"][0]["customer_id"] == customers[0]["customer_id"]
    assert all(0 <= r["churn_probability"] <= 1 for r in data["results"])


def test_predict_csv_download(client, trained_model):
    model, _ = trained_model
    to_score = make_customers(40, seed=9, labelled=False)
    resp = client.post(f"/models/{model['id']}/predict/csv",
                       files={"file": ("current.csv", _csv(to_score), "text/csv")})
    assert resp.status_code == 200
    assert 'filename="scored_current.csv"' in resp.headers["content-disposition"]
    scored = pd.read_csv(io.StringIO(resp.text))
    assert len(scored) == 40
    assert list(scored.columns) == list(to_score.columns) + [
        "churn_probability", "churn_prediction", "risk_tier", "recommended_action"]


def test_list_and_delete_model(client, trained_model):
    model, _ = trained_model
    assert [m["id"] for m in client.get("/models").json()] == [model["id"]]
    assert client.delete(f"/models/{model['id']}").status_code == 204
    assert client.get(f"/models/{model['id']}").status_code == 404


# ── Errors ─────────────────────────────────────────────────────────────────────

def test_rejects_non_csv(client):
    resp = client.post("/datasets", files={"file": ("data.xlsx", b"junk", "application/octet-stream")})
    assert resp.status_code == 400


def test_rejects_bad_target_before_training(client, registry):
    dataset = upload(client, make_customers(300, seed=4)).json()
    resp = client.post("/models", json={"dataset_id": dataset["id"], "target": "region"})
    assert resp.status_code == 422
    assert "exactly 2" in resp.json()["detail"]
    assert registry.list_models() == []


def test_failed_training_keeps_dataset_for_retry(client, monkeypatch):
    import api.engine as engine_api
    from src.engine import TrainingError

    def fail(*args, **kwargs):
        raise TrainingError("boom")
    monkeypatch.setattr(engine_api, "train_pipeline", fail)

    dataset = upload(client, make_customers(300, seed=4)).json()
    model = client.post("/models", json={"dataset_id": dataset["id"], "target": "churned"}).json()
    meta = client.get(f"/models/{model['id']}").json()
    assert (meta["status"], meta["error"]) == ("failed", "boom")
    assert client.get(f"/datasets/{dataset['id']}").status_code == 200


def test_predict_rejects_unrelated_data(client, trained_model):
    model, _ = trained_model
    resp = client.post(f"/models/{model['id']}/predict", json={"customers": [{"foo": 1, "bar": 2}]})
    assert resp.status_code == 422
    assert "Missing" in resp.json()["detail"]


@pytest.mark.parametrize("model_id", ["doesnotexist", "../../etc", "0123456789ab"])
def test_unknown_or_malicious_ids_404(client, model_id):
    assert client.get(f"/models/{model_id}").status_code == 404


def test_interrupted_jobs_marked_failed(registry):
    meta = registry.create_model(name="x")
    registry.update_model(meta["id"], status="training")
    assert registry.recover_interrupted() == 1
    assert registry.get_model(meta["id"])["status"] == "failed"
