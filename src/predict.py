"""
predict.py
----------
Load the trained LightGBM model and score a new customer
(or a batch of customers) returning churn probability + risk tier.

Usage:
    from src.predict import ChurnPredictor
    predictor = ChurnPredictor("models/")
    result = predictor.predict(customer_dict)
"""

import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Risk tier thresholds
TIER_THRESHOLDS = {
    "High Risk":   0.60,
    "Medium Risk": 0.40,
    "Low Risk":    0.00,
}

# Recommended actions per tier
TIER_ACTIONS = {
    "High Risk": (
        "Priority outreach: dedicated account manager call, "
        "device upgrade offer, customised plan review."
    ),
    "Medium Risk": (
        "Targeted campaign: personalised loyalty reward SMS/email, "
        "plan optimisation offer, satisfaction survey."
    ),
    "Low Risk": (
        "Passive retention: automated satisfaction email, "
        "referral programme invitation, feature education."
    ),
}


class ChurnPredictor:
    """
    Wraps the trained LightGBM model with a clean predict() interface.

    Parameters
    ----------
    model_dir : directory containing lgbm_churn.pkl, encoders.pkl,
                and feature_cols.pkl (output of src/train.py)
    """

    def __init__(self, model_dir: str | Path = "models/") -> None:
        model_dir = Path(model_dir)
        self.model        = joblib.load(model_dir / "lgbm_churn.pkl")
        self.encoders     = joblib.load(model_dir / "encoders.pkl")
        self.feature_cols = joblib.load(model_dir / "feature_cols.pkl")
        logger.info(f"Model loaded from {model_dir}")

    def _preprocess_single(self, customer: dict) -> pd.DataFrame:
        """Convert a raw customer dict into a model-ready feature row."""
        from src.preprocess import engineer_features, encode_categoricals

        df = pd.DataFrame([customer])
        df = engineer_features(df)
        df, _ = encode_categoricals(df, fit_encoders=False, encoders=self.encoders)

        # Align columns — fill any missing with 0
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        return df[self.feature_cols].fillna(0)

    def _score_to_tier(self, prob: float) -> tuple[str, str]:
        """Map a churn probability to a risk tier and recommended action."""
        if prob >= TIER_THRESHOLDS["High Risk"]:
            tier = "High Risk"
        elif prob >= TIER_THRESHOLDS["Medium Risk"]:
            tier = "Medium Risk"
        else:
            tier = "Low Risk"
        return tier, TIER_ACTIONS[tier]

    def predict(self, customer: dict) -> dict:
        """
        Score a single customer.

        Parameters
        ----------
        customer : dict of raw feature values (same schema as the CSVs)

        Returns
        -------
        dict with keys:
            churn_probability  : float (0–1)
            churn_prediction   : int (0 or 1, threshold 0.5)
            risk_tier          : str
            recommended_action : str
        """
        X = self._preprocess_single(customer)
        prob = float(self.model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
        tier, action = self._score_to_tier(prob)

        return {
            "churn_probability":  round(prob, 4),
            "churn_prediction":   pred,
            "risk_tier":          tier,
            "recommended_action": action,
        }

    def predict_batch(self, customers: list[dict]) -> list[dict]:
        """
        Score a list of customers.

        Parameters
        ----------
        customers : list of customer dicts

        Returns
        -------
        list of result dicts (same schema as predict())
        """
        return [self.predict(c) for c in customers]

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a full dataframe and append churn_prob + risk_tier columns.

        Parameters
        ----------
        df : raw dataframe (must include all feature columns)

        Returns
        -------
        original df with 3 new columns: churn_prob, churn_pred, risk_tier
        """
        from src.preprocess import engineer_features, encode_categoricals

        df = df.copy()
        df = engineer_features(df)
        df, _ = encode_categoricals(df, fit_encoders=False, encoders=self.encoders)

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[self.feature_cols].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]

        df["churn_prob"] = proba
        df["churn_pred"] = (proba >= 0.5).astype(int)
        df["risk_tier"]  = pd.cut(
            df["churn_prob"],
            bins=[0, 0.4, 0.6, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )
        return df
