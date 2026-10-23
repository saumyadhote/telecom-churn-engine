"""
tests/test_engine.py
--------------------
Unit tests for schema inference and the bring-your-own-data training engine.
These train real models on synthetic data — nothing is mocked.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.engine import TrainingError, train_pipeline
from src.sample_data import make_customers
from src.schema import encode_target, infer_schema, suggest_target


@pytest.fixture(scope="module")
def customers():
    return make_customers(2000, seed=1)


@pytest.fixture(scope="module")
def trained(customers):
    return train_pipeline(customers, target="churned")


# ── Schema inference ───────────────────────────────────────────────────────────

def test_infer_schema_roles(customers):
    roles = {c["name"]: c["role"] for c in infer_schema(customers)["columns"]}
    assert roles["customer_id"] == "id"
    assert roles["churned"] == "target"
    assert roles["contract"] == "categorical"
    assert roles["tenure_months"] == "numeric"
    # Mostly numbers with a few blanks → still numeric
    assert roles["total_charges"] == "numeric"


def test_infer_schema_ignores_unusable_columns():
    df = pd.DataFrame({
        "constant":  ["x"] * 300,
        "empty":     [np.nan] * 299 + [1.0],
        "free_text": [f"note {i}" for i in range(300)],
        "Exited":    [0, 1] * 150,
    })
    schema = infer_schema(df)
    roles = {c["name"]: c["role"] for c in schema["columns"]}
    assert roles == {"constant": "ignored", "empty": "ignored", "free_text": "ignored", "Exited": "target"}
    assert schema["suggested_target"] == "Exited"


def test_suggest_target_none_when_no_hint():
    assert suggest_target(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})) is None


@pytest.mark.parametrize("values, positive", [
    (["Yes", "No", "No"], "Yes"),
    ([1, 0, 0], "1"),
    ([True, False, False], "True"),
    (["Churned", "Stayed", "Stayed"], "Churned"),
])
def test_encode_target_detects_positive_label(values, positive):
    y, label = encode_target(pd.Series(values))
    assert label == positive
    assert y.tolist() == [1, 0, 0]


def test_encode_target_explicit_label():
    y, label = encode_target(pd.Series(["Gone", "Active"]), positive_label="Gone")
    assert (label, y.tolist()) == ("Gone", [1, 0])


def test_encode_target_rejects_non_binary():
    with pytest.raises(ValueError, match="exactly 2"):
        encode_target(pd.Series(["a", "b", "c"]))


def test_encode_target_ambiguous_needs_label():
    with pytest.raises(ValueError, match="positive_label"):
        encode_target(pd.Series(["Gone", "Active"]))


# ── Training ───────────────────────────────────────────────────────────────────

def test_report_has_honest_split_and_metrics(trained):
    _, report = trained
    assert report["split"] == {"train": 1200, "validation": 400, "test": 400}
    assert report["model_type"] in ("lightgbm", "logistic_regression")
    assert set(report["comparison"]) == {"lightgbm", "logistic_regression"}
    assert report["metrics"]["auc_roc"] > 0.7, "synthetic data has a strong signal"
    assert report["id_column"] == "customer_id"
    assert report["positive_label"] == "Yes"


def test_tiers_rank_risk(trained):
    """Customers in higher tiers should actually churn more on held-out data."""
    _, report = trained
    rates = {t["tier"]: t["actual_churn_rate"] for t in report["tiers"]}
    assert rates["High Risk"] > rates["Medium Risk"] > rates["Low Risk"]


def test_top_features_are_real_columns(trained, customers):
    _, report = trained
    names = [f["feature"] for f in report["top_features"]]
    assert set(names) <= set(customers.columns)
    assert "customer_id" not in names and "churned" not in names
    assert "contract" in names[:3], "contract type is the strongest synthetic driver"


def test_score_output(trained):
    pipeline, _ = trained
    new = make_customers(50, seed=5, labelled=False)
    scored = pipeline.score(new)
    assert list(scored.columns) == ["customer_id", "churn_probability", "churn_prediction",
                                    "risk_tier", "recommended_action"]
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["risk_tier"]) <= {"High Risk", "Medium Risk", "Low Risk"}
    assert (scored["customer_id"] == new["customer_id"]).all()


def test_score_tolerates_messy_input(trained):
    pipeline, _ = trained
    messy = pd.DataFrame([
        {"contract": "Weekly", "tenure_months": 3},      # unseen category
        {},                                              # nothing at all
        {"tenure_months": "abc", "unrelated": 1},        # junk value + extra column
    ])
    assert pipeline.score(messy)["churn_probability"].between(0, 1).all()


def test_pipeline_survives_save_and_load(trained, tmp_path):
    pipeline, _ = trained
    joblib.dump(pipeline, tmp_path / "p.joblib")
    loaded = joblib.load(tmp_path / "p.joblib")
    new = make_customers(100, seed=6, labelled=False)
    pd.testing.assert_frame_equal(pipeline.score(new), loaded.score(new))


@pytest.mark.parametrize("model_type", ["lightgbm", "logistic_regression"])
def test_both_model_types_score(customers, model_type, monkeypatch):
    """Whichever model wins validation, scoring and feature importances must work."""
    import src.engine as engine
    original = engine._evaluate
    def rigged(model, X, y):
        metrics, proba = original(model, X, y)
        is_target = (model_type == "lightgbm") == hasattr(model, "booster_")
        return {**metrics, "auc_roc": 1.0 if is_target else 0.0}, proba
    monkeypatch.setattr(engine, "_evaluate", rigged)
    pipeline, report = train_pipeline(customers, target="churned")
    assert report["model_type"] == model_type
    assert report["top_features"]
    assert pipeline.score(customers.head(10))["churn_probability"].between(0, 1).all()


def test_exclude_columns(customers):
    _, report = train_pipeline(customers, target="churned", exclude_cols=["contract"])
    assert "contract" not in report["features_used"]["categorical"]
    assert {"name": "contract", "reason": "excluded by you"} in report["ignored_columns"]


def test_custom_tier_thresholds(customers):
    pipeline, report = train_pipeline(customers, target="churned",
                                      tier_thresholds={"High Risk": 0.9, "Medium Risk": 0.8})
    assert report["tier_thresholds"] == {"High Risk": 0.9, "Medium Risk": 0.8}
    assert pipeline.tier(np.array([0.85]))[0] == "Medium Risk"


def test_leakage_warning(customers):
    leaky = customers.assign(days_since_cancellation=np.where(customers["churned"] == "Yes", 5.0, 0.0))
    _, report = train_pipeline(leaky, target="churned")
    assert any("days_since_cancellation" in w for w in report["warnings"])


@pytest.mark.parametrize("df, message", [
    (make_customers(100, seed=2), "at least 200"),
    (make_customers(2000, seed=2).assign(churned="No").pipe(
        lambda d: d.assign(churned=["Yes"] * 10 + ["No"] * (len(d) - 10))), "at least 50 churned"),
    (make_customers(500, seed=2).assign(churned="Maybe"), "exactly 2"),
])
def test_unusable_data_rejected(df, message):
    with pytest.raises(TrainingError, match=message):
        train_pipeline(df, target="churned")
