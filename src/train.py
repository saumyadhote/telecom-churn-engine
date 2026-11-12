"""
train.py
--------
Train Logistic Regression (baseline) and LightGBM (final model),
evaluate both, and persist the LightGBM model + encoders to disk.

Data is split three ways (60 / 20 / 20, stratified):
    train      → fit both models
    validation → LightGBM early stopping picks the number of trees here
    test       → touched once, at the end, for the reported metrics

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
    "n_estimators":    300,          # upper bound — early stopping on validation picks the count
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

VAL_SIZE    = 0.2
TEST_SIZE   = 0.2
RANDOM_SEED = 42
EARLY_STOPPING_ROUNDS = 50


# ── Training ───────────────────────────────────────────────────────────────────

def train(client_path: str, record_path: str, output_dir: str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & preprocess
    logger.info("Loading data...")
    df = load_data(client_path, record_path)
    X, y, encoders = build_feature_matrix(df, fit_encoders=True)

    # 2. Train / validation / test split — the test set is only used for the final report
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED, stratify=y_rest,
    )
    logger.info(f"Train: {len(X_train):,}  |  Validation: {len(X_val):,}  |  Test: {len(X_test):,}")

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
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
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
        "validation_rows":     len(X_val),
        "test_rows":           len(X_test),
        "lightgbm_trees":      int(lgbm.best_iteration_ or lgbm.n_estimators),
        "split":               "60/20/20 stratified; early stopping on validation, metrics on test",
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Model saved → {output_dir / 'lgbm_churn.pkl'}")
    logger.info(f"Results saved → {output_dir / 'results.json'}")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train telecom churn model")
    parser.add_argument("--client", required=True, help="Path to Client.csv")
    parser.add_argument("--record", required=True, help="Path to Record.csv")
    parser.add_argument("--output", default="models/", help="Output directory")
    args = parser.parse_args()

    train(args.client, args.record, args.output)
