"""
tests/test_preprocess.py
------------------------
Unit tests for data loading and feature engineering.
"""

import pandas as pd
import numpy as np
import pytest
from src.preprocess import engineer_features, encode_categoricals


@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe mimicking the merged dataset."""
    return pd.DataFrame({
        "Customer_ID":    ["A001", "A002", "A003"],
        "comp_vce_Mean":  [80.0, 60.0, 90.0],
        "plcd_vce_Mean":  [85.0, 65.0, 95.0],
        "drop_vce_Mean":  [3.0,  4.0,  1.0],
        "rev_Mean":       [55.0, 42.0, 70.0],
        "mou_Mean":       [320.0, 200.0, 410.0],
        "ovrmou_Mean":    [0.0,  10.0,  0.0],
        "change_mou":     [-50.0, 20.0, 10.0],
        "months":         [18,   6,    36],
        "churn":          [1,    0,    0],
        "area":           ["CHICAGO IL", "DALLAS TX", None],
    })


def test_engineer_features_adds_columns(sample_df):
    result = engineer_features(sample_df)
    for col in ["call_completion_rate", "drop_rate", "rev_per_min", "overage_flag"]:
        assert col in result.columns, f"Missing engineered column: {col}"


def test_call_completion_rate_range(sample_df):
    result = engineer_features(sample_df)
    assert (result["call_completion_rate"] >= 0).all()
    assert (result["call_completion_rate"] <= 1.01).all()  # slight epsilon tolerance


def test_overage_flag_binary(sample_df):
    result = engineer_features(sample_df)
    assert set(result["overage_flag"].unique()).issubset({0, 1})


def test_overage_flag_correct(sample_df):
    result = engineer_features(sample_df)
    # Row 1 has ovrmou_Mean=10 → flag=1; rows 0,2 have 0 → flag=0
    assert result["overage_flag"].tolist() == [0, 1, 0]


def test_encode_categoricals_no_nans(sample_df):
    df_eng = engineer_features(sample_df)
    df_enc, encoders = encode_categoricals(df_eng, fit_encoders=True)
    assert df_enc["area"].isna().sum() == 0


def test_encode_categoricals_returns_numeric(sample_df):
    df_eng = engineer_features(sample_df)
    df_enc, _ = encode_categoricals(df_eng, fit_encoders=True)
    assert df_enc["area"].dtype in [np.int64, np.int32, int]


def test_engineer_features_no_inf(sample_df):
    result = engineer_features(sample_df)
    numeric = result.select_dtypes(include=np.number)
    assert not np.isinf(numeric.values).any(), "Infinite values found after feature engineering"
