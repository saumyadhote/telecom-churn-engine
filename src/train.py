"""
train.py
--------
Train Logistic Regression (baseline) and LightGBM (final model),
evaluate both, and persist the best model + encoders to disk.

Usage:
    python -m src.train \
        --client data/Client.csv \
        --record data/Record.csv \
        --output models/
"""

import argparse
import joblib
import json
import logging
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.preprocess import load_data, build_feature_matrix
from src.evaluate import compute_metrics, print_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Hyperparameters ────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "n_estimators":    300,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "min_child_samples": 30,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "random_state":    42,
    "verbose":         -1,
}

LR_PARAMS = {
    "max_iter":    500,
    "random_state": 42,
    "C":           0.1,
}

TEST_SIZE   = 0.2
RANDOM_SEED = 42


# ── Training ───────────────────────────────────────────────────────────────────

def train(client_path: str, record_path: str, output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & preprocess
    logger.info("Loading data...")
    df = load_data(client_path, record_path)
    X, y, encoders = build_feature_matrix(df, fit_encoders=True)

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 3. Baseline — Logistic Regression
    logger.info("Training Logistic Regression baseline...")
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_train, y_train)
    lr_metrics = compute_metrics(y_test, lr.predict(X_test), lr.predict_proba(X_test)[:, 1])
    print_metrics("Logistic Regression (baseline)", lr_metrics)

    # 4. LightGBM
    logger.info("Training LightGBM...")
    lgbm = lgb.LGBMClassifier(**LGBM_PARAMS)
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    lgbm_metrics = compute_metrics(
        y_test, lgbm.predict(X_test), lgbm.predict_proba(X_test)[:, 1]
    )
    print_metrics("LightGBM (final model)", lgbm_metrics)

    # 5. Save artefacts
    joblib.dump(lgbm,     output_dir / "lgbm_churn.pkl")
    joblib.dump(encoders, output_dir / "encoders.pkl")
    joblib.dump(list(X.columns), output_dir / "feature_cols.pkl")

    # Save metrics to JSON for README badge / CI checks
    results = {
        "logistic_regression": lr_metrics,
        "lightgbm":            lgbm_metrics,
        "feature_count":       X.shape[1],
        "train_rows":          len(X_train),
        "test_rows":           len(X_test),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Model saved → {output_dir / 'lgbm_churn.pkl'}")
    logger.info(f"Results saved → {output_dir / 'results.json'}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train telecom churn model")
    parser.add_argument("--client", required=True, help="Path to Client.csv")
    parser.add_argument("--record", required=True, help="Path to Record.csv")
    parser.add_argument("--output", default="models/", help="Output directory")
    args = parser.parse_args()

    train(args.client, args.record, args.output)
