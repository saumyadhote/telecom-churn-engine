"""
tests/test_v1_api.py
--------------------
End-to-end tests for the public /v1 developer API: auth, per-account
isolation, quotas, rate limits, the error contract, and the full
upload → train → score flow. Real models, temp storage, nothing mocked.
"""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.v1 as v1
from api.auth import get_accounts, request_limiter, signup_limiter
from api.main import app
from src.accounts import TIERS, AccountStore
from src.registry import Registry
from src.sample_data import make_customers


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "storage")


@pytest.fixture
def accounts(tmp_path):
    return AccountStore(tmp_path / "storage" / "accounts.db")


@pytest.fixture
def anon(registry, accounts):
    """Client with no API key."""
    app.dependency_overrides[v1.get_registry] = lambda: registry
    app.dependency_overrides[get_accounts] = lambda: accounts
    request_limiter.reset()
    signup_limiter.reset()
    yield TestClient(app)
    app.dependency_overrides.clear()


def _client_for(accounts, name="Ada", tier="standard"):
    _, key = accounts.create_account(name, tier=tier)
    return TestClient(app, headers={"Authorization": f"Bearer {key}"}), key


@pytest.fixture
def client(anon, accounts):
    return _client_for(accounts)[0]


def upload(client, df, name="customers.csv"):
    return client.post("/v1/datasets", files={"file": (name, _csv(df), "text/csv")})


def train(client, rows=1500, seed=3, **options):
    dataset = upload(client, make_customers(rows, seed=seed)).json()
    # TestClient runs background tasks before returning, so training is done here
    resp = client.post("/v1/models", json={"dataset_id": dataset["id"], "target": "churned", **options})
    assert resp.status_code == 202, resp.json()
    return client.get(f"/v1/models/{resp.json()['id']}").json()


@pytest.fixture
def model(client):
    return train(client)


# ── Auth ───────────────────────────────────────────────────────────────────────

def test_health_needs_no_key(anon):
    assert anon.get("/v1/health").json()["status"] == "ok"


def test_missing_key(anon):
    resp = anon.get("/v1/models")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "missing_api_key"
    assert resp.headers["www-authenticate"] == "Bearer"


def test_invalid_key(anon):
    resp = anon.get("/v1/models", headers={"Authorization": "Bearer ck_nope"})
    assert (resp.status_code, resp.json()["error"]["code"]) == (401, "invalid_api_key")


def test_x_api_key_header_works(anon, accounts):
    _, key = accounts.create_account("Bob")
    assert anon.get("/v1/account", headers={"X-API-Key": key}).status_code == 200


def test_revoked_key_rejected(anon, accounts):
    client, key = _client_for(accounts)
    assert client.get("/v1/account").status_code == 200
    accounts.revoke_key(key)
    assert client.get("/v1/account").json()["error"]["code"] == "invalid_api_key"


def test_account_shows_limits_and_usage(client, model):
    data = client.get("/v1/account").json()
    assert data["tier"] == "standard"
    assert data["limits"] == TIERS["standard"]
    assert data["usage"] == {"models": 1, "trainings_today": 1}


# ── Full flow ──────────────────────────────────────────────────────────────────

def test_upload_infers_schema(client):
    resp = upload(client, make_customers(300, seed=4))
    assert resp.status_code == 201
    data = resp.json()
    assert (data["rows"], data["suggested_target"], data["suggested_id"]) == (300, "churned", "customer_id")
    assert data["target_values"] == ["No", "Yes"]
    assert {c["name"]: c["role"] for c in data["columns"]}["contract"] == "categorical"
    assert "owner" not in data


def test_train_produces_ready_model(model):
    assert model["status"] == "ready", model.get("error")
    report = model["report"]
    assert report["metrics"]["auc_roc"] > 0.7
    assert {s["name"] for s in report["input_schema"]} >= {"tenure_months", "contract"}
    assert "owner" not in model


def test_dataset_deleted_after_successful_training(client):
    dataset = upload(client, make_customers(1500, seed=3)).json()
    client.post("/v1/models", json={"dataset_id": dataset["id"], "target": "churned"})
    assert client.get(f"/v1/datasets/{dataset['id']}").status_code == 404


def test_predict_json(client, model):
    customers = make_customers(5, seed=8, labelled=False).to_dict(orient="records")
    resp = client.post(f"/v1/models/{model['id']}/predict", json={"customers": customers})
    assert resp.status_code == 200
    data = resp.json()
    assert (data["count"], data["missing_columns"]) == (5, [])
    assert data["results"][0]["id"] == customers[0]["customer_id"]
    assert all(0 <= r["churn_probability"] <= 1 for r in data["results"])


def test_predict_csv_download(client, model):
    to_score = make_customers(40, seed=9, labelled=False)
    resp = client.post(f"/v1/models/{model['id']}/predict/csv",
                       files={"file": ("current.csv", _csv(to_score), "text/csv")})
    assert resp.status_code == 200
    assert 'filename="scored_current.csv"' in resp.headers["content-disposition"]
    assert "x-ratelimit-remaining" in resp.headers
    scored = pd.read_csv(io.StringIO(resp.text))
    assert list(scored.columns) == list(to_score.columns) + [
        "churn_probability", "churn_prediction", "risk_tier", "recommended_action"]


def test_list_and_delete_model(client, model):
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == [model["id"]]
    assert client.delete(f"/v1/models/{model['id']}").status_code == 204
    assert client.get(f"/v1/models/{model['id']}").json()["error"]["code"] == "model_not_found"


# ── Isolation between accounts ─────────────────────────────────────────────────

def test_accounts_cannot_see_each_others_data(client, model, accounts):
    other, _ = _client_for(accounts, "Eve")
    assert other.get("/v1/models").json()["data"] == []
    for method, path in [("get", f"/v1/models/{model['id']}"), ("delete", f"/v1/models/{model['id']}")]:
        assert getattr(other, method)(path).status_code == 404
    resp = other.post(f"/v1/models/{model['id']}/predict", json={"customers": [{"tenure_months": 1}]})
    assert resp.status_code == 404
    # ...and the owner still has it
    assert client.get(f"/v1/models/{model['id']}").status_code == 200


def test_cannot_train_on_someone_elses_dataset(client, accounts):
    dataset = upload(client, make_customers(300, seed=4)).json()
    other, _ = _client_for(accounts, "Eve")
    resp = other.post("/v1/models", json={"dataset_id": dataset["id"], "target": "churned"})
    assert resp.json()["error"]["code"] == "dataset_not_found"


# ── Limits ─────────────────────────────────────────────────────────────────────

def test_rate_limit(anon, accounts, monkeypatch):
    monkeypatch.setitem(TIERS, "free", {**TIERS["free"], "requests_per_minute": 3})
    client, _ = _client_for(accounts, tier="free")
    for remaining in ("2", "1", "0"):
        resp = client.get("/v1/account")
        assert resp.headers["x-ratelimit-remaining"] == remaining
    resp = client.get("/v1/account")
    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "rate_limited"
    assert int(resp.headers["retry-after"]) >= 1


def test_model_limit(anon, accounts, monkeypatch):
    monkeypatch.setitem(TIERS, "free", {**TIERS["free"], "max_models": 1})
    client, _ = _client_for(accounts, tier="free")
    train(client)
    dataset = upload(client, make_customers(300, seed=4)).json()
    resp = client.post("/v1/models", json={"dataset_id": dataset["id"], "target": "churned"})
    assert (resp.status_code, resp.json()["error"]["code"]) == (409, "model_limit_reached")


def test_upload_size_limit(anon, accounts, monkeypatch):
    monkeypatch.setitem(TIERS, "free", {**TIERS["free"], "max_upload_mb": 0.001})
    client, _ = _client_for(accounts, tier="free")
    resp = upload(client, make_customers(300, seed=4))
    assert (resp.status_code, resp.json()["error"]["code"]) == (413, "file_too_large")


# ── Signup ─────────────────────────────────────────────────────────────────────

def test_signup_disabled_by_default(anon):
    resp = anon.post("/v1/signup", json={"name": "Ada"})
    assert (resp.status_code, resp.json()["error"]["code"]) == (403, "signup_disabled")


def test_open_signup_issues_working_free_key(anon, monkeypatch):
    monkeypatch.setattr(v1, "OPEN_SIGNUP", True)
    resp = anon.post("/v1/signup", json={"name": "Ada", "email": "ada@example.com"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["api_key"].startswith("ck_")
    assert body["account"]["tier"] == "free"
    me = anon.get("/v1/account", headers={"Authorization": f"Bearer {body['api_key']}"})
    assert me.json()["id"] == body["account"]["id"]


def test_signup_rate_limited_per_ip(anon, monkeypatch):
    monkeypatch.setattr(v1, "OPEN_SIGNUP", True)
    codes = [anon.post("/v1/signup", json={"name": f"user{i}"}).status_code for i in range(6)]
    assert codes == [201] * 5 + [429]


# ── Error contract ─────────────────────────────────────────────────────────────

def test_validation_errors_use_error_shape(client):
    resp = client.post("/v1/models", json={"target": "churned"})
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["field"] == "dataset_id"


def test_unknown_v1_route_uses_error_shape(client):
    resp = client.get("/v1/nope")
    assert (resp.status_code, resp.json()["error"]["code"]) == (404, "not_found")


def test_demo_endpoints_keep_original_error_shape(anon):
    assert "detail" in anon.post("/predict", json={"months": "not a number"}).json()


def test_rejects_non_csv(client):
    resp = client.post("/v1/datasets", files={"file": ("data.xlsx", b"junk", "application/octet-stream")})
    assert resp.json()["error"]["code"] == "invalid_file_type"


def test_rejects_bad_target_before_training(client, registry):
    dataset = upload(client, make_customers(300, seed=4)).json()
    resp = client.post("/v1/models", json={"dataset_id": dataset["id"], "target": "region"})
    assert (resp.status_code, resp.json()["error"]["code"]) == (422, "invalid_target")
    assert registry.list_models() == []


def test_failed_training_keeps_dataset_for_retry(client, monkeypatch):
    from src.engine import TrainingError

    def fail(*args, **kwargs):
        raise TrainingError("boom")
    monkeypatch.setattr(v1, "train_pipeline", fail)

    dataset = upload(client, make_customers(300, seed=4)).json()
    model = client.post("/v1/models", json={"dataset_id": dataset["id"], "target": "churned"}).json()
    meta = client.get(f"/v1/models/{model['id']}").json()
    assert (meta["status"], meta["error"]) == ("failed", "boom")
    assert client.get(f"/v1/datasets/{dataset['id']}").status_code == 200


def test_predict_on_unready_model(client, registry, accounts):
    owner = client.get("/v1/account").json()["id"]
    meta = registry.create_model(owner=owner, name="x", dataset_filename="x.csv", target="t", positive_label="1")
    resp = client.post(f"/v1/models/{meta['id']}/predict", json={"customers": [{"a": 1}]})
    assert (resp.status_code, resp.json()["error"]["code"]) == (409, "model_not_ready")


def test_predict_rejects_unrelated_data(client, model):
    resp = client.post(f"/v1/models/{model['id']}/predict", json={"customers": [{"foo": 1, "bar": 2}]})
    error = resp.json()["error"]
    assert (resp.status_code, error["code"]) == (422, "insufficient_columns")
    assert "tenure_months" in error["details"]["missing_columns"]


@pytest.mark.parametrize("model_id", ["doesnotexist", "0123456789ab", "..%2F..%2Fetc"])
def test_unknown_or_malicious_ids_404(client, model_id):
    assert client.get(f"/v1/models/{model_id}").status_code == 404


def test_interrupted_jobs_marked_failed(registry):
    meta = registry.create_model(name="x")
    registry.update_model(meta["id"], status="training")
    assert registry.recover_interrupted() == 1
    assert registry.get_model(meta["id"])["status"] == "failed"


def test_openapi_documents_v1(anon):
    spec = anon.get("/openapi.json").json()
    assert {"/v1/models", "/v1/models/{model_id}/predict", "/v1/datasets"} <= set(spec["paths"])
    assert {"Model", "TrainingReport", "PredictResponse", "ErrorResponse"} <= set(spec["components"]["schemas"])
