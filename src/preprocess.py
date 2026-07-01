"""
preprocess.py
-------------
Data loading, merging, and feature engineering for the
Telecom Churn Prediction Engine.

Usage:
    from src.preprocess import load_data, engineer_features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


# ── Column groups ─────────────────────────────────────────────────────────────

TARGET = "churn"

DROP_COLS = ["Customer_ID", "churn"]

CATEGORICAL_COLS = [
    "area", "dualband", "hnd_webcap", "ownrent", "lor",
    "dwlltype", "marital", "infobase", "ethnic", "kid0_2",
    "kid3_5", "kid6_10", "kid11_15", "kid16_17",
    "creditcd", "new_cell", "crclscod", "asl_flag",
    "prizm_social_one", "refurb_new", "hnd_price",
    "phones", "models", "truck", "rv", "forgntvl",
    "occup", "hhstatin", "dwllsize",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(
    client_path: str | Path,
    record_path: str | Path,
) -> pd.DataFrame:
    """
    Load Client.csv and Record.csv, merge on Customer_ID.

    Parameters
    ----------
    client_path : path to Client.csv
    record_path : path to Record.csv

    Returns
    -------
    pd.DataFrame : merged dataframe (100k rows × ~100 cols)
    """
    client = pd.read_csv(client_path)
    record = pd.read_csv(record_path)

    df = record.merge(client, on="Customer_ID", how="inner")
    logger.info(f"Loaded merged dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Churn rate: {df[TARGET].mean()*100:.2f}%")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 4 derived features that improve model performance:

    1. call_completion_rate  — % of placed voice calls that completed
    2. drop_rate             — % of placed voice calls that were dropped
    3. rev_per_min           — revenue per minute of use (value efficiency)
    4. overage_flag          — binary: did the customer incur any overage?

    Parameters
    ----------
    df : merged dataframe from load_data()

    Returns
    -------
    pd.DataFrame : dataframe with 4 new columns appended
    """
    df = df.copy()

    # Avoid division by zero with a small epsilon
    eps = 1e-5

    df["call_completion_rate"] = df["comp_vce_Mean"] / (df["plcd_vce_Mean"] + eps)
    df["drop_rate"]            = df["drop_vce_Mean"] / (df["plcd_vce_Mean"] + eps)
    df["rev_per_min"]          = df["rev_Mean"]      / (df["mou_Mean"]       + eps)
    df["overage_flag"]         = (df["ovrmou_Mean"] > 0).astype(int)

    logger.info("Engineered 4 new features: call_completion_rate, drop_rate, rev_per_min, overage_flag")
    return df


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    fit_encoders: bool = True,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all object/categorical columns.

    Parameters
    ----------
    df           : input dataframe
    fit_encoders : if True, fit new encoders; if False, use provided encoders
    encoders     : dict of {col: LabelEncoder} — required if fit_encoders=False

    Returns
    -------
    (encoded_df, encoders_dict)
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if fit_encoders:
        encoders = {}
        for col in cat_cols:
            df[col] = df[col].fillna("MISSING")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        assert encoders is not None, "Must provide encoders when fit_encoders=False"
        for col in cat_cols:
            if col in encoders:
                df[col] = df[col].fillna("MISSING")
                # Handle unseen labels gracefully
                known = set(encoders[col].classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else "MISSING"
                )
                df[col] = encoders[col].transform(df[col])

    return df, encoders


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    fit_encoders: bool = True,
    encoders: dict | None = None,
    drop_cols: list | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, dict]:
    """
    Run the complete preprocessing pipeline:
    engineer features → encode categoricals → split X / y.

    Parameters
    ----------
    df           : raw merged dataframe
    fit_encoders : whether to fit new label encoders
    encoders     : pre-fitted encoders (used at inference time)
    drop_cols    : columns to drop (defaults to DROP_COLS + ['tenure_group'])

    Returns
    -------
    (X, y, encoders)
        X        : feature matrix (pd.DataFrame)
        y        : target series, or None if TARGET not in df
        encoders : fitted encoder dict
    """
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, fit_encoders=fit_encoders, encoders=encoders)

    if drop_cols is None:
        drop_cols = DROP_COLS + (["tenure_group"] if "tenure_group" in df.columns else [])

    y = df[TARGET] if TARGET in df.columns else None
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

    logger.info(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    return X, y, encoders
