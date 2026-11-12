"""
tests/test_end_to_end.py
------------------------
The telecom path with nothing mocked: synthetic Client.csv / Record.csv →
src.train.train() → real artefacts on disk → real ChurnPredictor → real API.

This is the test that catches training/serving skew (a renamed column, a
changed feature order, an encoder that isn't applied at inference) — the
mocked tests in test_api.py can't see that class of bug.
"""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.predict import ChurnPredictor
from src.train import train

SAMPLE_CUSTOMER = json.loads(open("data/sample_input.json").read())


def _synthetic_telecom(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small dataset with the real schema's key columns and a learnable churn signal."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1_000_000, 1_000_000 + n)

    months     = rng.integers(1, 60, n)
    eqpdays    = rng.integers(0, 1200, n)
    change_mou = rng.normal(-20, 120, n)
    plcd       = rng.uniform(10, 200, n)
    logit = -0.5 - 0.04 * months + 0.002 * eqpdays - 0.006 * change_mou + rng.normal(0, 0.5, n)
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    record = pd.DataFrame({
        "Customer_ID":   ids,
        "months":        months,
        "eqpdays":       eqpdays,
        "rev_Mean":      rng.uniform(10, 120, n).round(2),
        "mou_Mean":      rng.uniform(0, 1500, n).round(1),
        "change_mou":    change_mou.round(1),
        "change_rev":    rng.normal(0, 10, n).round(2),
        "totmrc_Mean":   rng.uniform(10, 90, n).round(2),
        "ovrmou_Mean":   np.where(rng.random(n) < 0.3, rng.uniform(1, 100, n), 0).round(1),
        "plcd_vce_Mean": plcd.round(1),
        "comp_vce_Mean": (plcd * rng.uniform(0.8, 1.0, n)).round(1),
        "drop_vce_Mean": (plcd * rng.uniform(0, 0.05, n)).round(1),
        "custcare_Mean": rng.poisson(2, n),
        "churn":         churn,
    })
    client = pd.DataFrame({
        "Customer_ID": ids,
        "area":        rng.choice(["CHICAGO IL", "NEW YORK NY", "LOS ANGELES CA"], n),
        "crclscod":    rng.choice(["A", "B", "C"], n),
        "new_cell":    rng.choice(["Y", "N", "U"], n),
        "asl_flag":    rng.choice(["Y", "N"], n),
    })
    return client, record


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("telecom")
    client, record = _synthetic_telecom()
    client.to_csv(root / "Client.csv", index=False)
    record.to_csv(root / "Record.csv", index=False)
    results = train(str(root / "Client.csv"), str(root / "Record.csv"), str(root / "models"))
    return root / "models", results


def test_training_writes_artefacts_and_honest_split(trained):
    model_dir, results = trained
    for name in ("lgbm_churn.pkl", "encoders.pkl", "feature_cols.pkl", "results.json"):
        assert (model_dir / name).exists()
    # 60 / 20 / 20 — the test rows are disjoint from the early-stopping rows
    assert results["train_rows"] == 360
    assert results["validation_rows"] == 120
    assert results["test_rows"] == 120
    assert results["lightgbm"]["auc_roc"] > 0.6     # the synthetic signal is learnable


def test_real_predictor_scores_sample_customer(trained):
    predictor = ChurnPredictor(trained[0])
    result = predictor.predict(SAMPLE_CUSTOMER)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_tier"] in ("Low Risk", "Medium Risk", "High Risk")


def test_batch_matches_single(trained):
    """The vectorized batch path must agree with one-at-a-time scoring."""
    predictor = ChurnPredictor(trained[0])
    _, record = _synthetic_telecom(n=25, seed=3)
    customers = record.drop(columns=["Customer_ID", "churn"]).to_dict("records")
    customers[0]["area"] = "SOMEWHERE NEW"           # unseen category
    customers[1]["area"] = None                       # missing category

    batch = predictor.predict_batch(customers)
    single = [predictor.predict(c) for c in customers]
    assert [r["churn_probability"] for r in batch] == pytest.approx(
        [r["churn_probability"] for r in single], abs=1e-4)
    assert [r["risk_tier"] for r in batch] == [r["risk_tier"] for r in single]


@pytest.mark.parametrize("prob, tier", [
    (0.60, "High Risk"), (0.5999, "Medium Risk"), (0.40, "Medium Risk"), (0.3999, "Low Risk"), (0.0, "Low Risk"),
])
def test_tier_boundaries(prob, tier):
    assert ChurnPredictor._score_to_tier(prob)[0] == tier


def test_api_with_real_model(trained, monkeypatch):
    import api.main
    monkeypatch.setattr(api.main, "predictor", ChurnPredictor(trained[0]))
    client = TestClient(api.main.app)

    single = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert single.status_code == 200

    batch = client.post("/predict/batch", json={"customers": [SAMPLE_CUSTOMER, {"months": 3}]})
    assert batch.status_code == 200
    body = batch.json()
    assert body["count"] == 2
    assert body["results"][0]["churn_probability"] == pytest.approx(single.json()["churn_probability"], abs=1e-4)
