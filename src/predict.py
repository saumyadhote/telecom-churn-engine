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

# Two independent cut-offs, on purpose:
#   * churn_prediction is a plain yes/no at PREDICTION_CUTOFF (0.5)
#   * risk_tier groups customers for *action*, so it uses its own bands
# A customer at 0.45 is "predicted to stay" but still Medium Risk — worth a
# cheap loyalty nudge. Tiers are lower-bound inclusive (0.60 → High Risk).
PREDICTION_CUTOFF = 0.5

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


# Raw columns engineer_features() reads
ENGINEERING_INPUTS = ("comp_vce_Mean", "plcd_vce_Mean", "drop_vce_Mean",
                      "rev_Mean", "mou_Mean", "ovrmou_Mean")


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

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Raw customer rows → model-ready frame. The single and batch paths both go
        through here, so they can't drift apart.

        Columns the caller didn't send: numeric → 0 (the documented default);
        categorical → MISSING, never 0, because 0 is a real encoded category.
        """
        from src.preprocess import engineer_features, encode_categoricals

        df = df.copy()
        for col in ENGINEERING_INPUTS:
            if col not in df.columns:
                df[col] = 0
        for col in self.encoders:
            if col not in df.columns:
                df[col] = pd.Series([None] * len(df), index=df.index, dtype=object)
        df = engineer_features(df)
        df, _ = encode_categoricals(df, fit_encoders=False, encoders=self.encoders)

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        return df

    @staticmethod
    def _score_to_tier(prob: float) -> tuple[str, str]:
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
            churn_prediction   : int (0 or 1, at PREDICTION_CUTOFF)
            risk_tier          : str
            recommended_action : str
        """
        X = self._features(pd.DataFrame([customer]))[self.feature_cols].fillna(0)
        prob = float(self.model.predict_proba(X)[0, 1])
        pred = int(prob >= PREDICTION_CUTOFF)
        tier, action = self._score_to_tier(prob)

        return {
            "churn_probability":  round(prob, 4),
            "churn_prediction":   pred,
            "risk_tier":          tier,
            "recommended_action": action,
        }

    def predict_batch(self, customers: list[dict]) -> list[dict]:
        """
        Score a list of customers in one vectorized model call.

        Parameters
        ----------
        customers : list of customer dicts

        Returns
        -------
        list of result dicts (same schema as predict(), same order as input)
        """
        if not customers:
            return []
        scored = self.predict_dataframe(pd.DataFrame(customers))
        results = []
        for prob in scored["churn_prob"]:
            prob = float(prob)
            tier, action = self._score_to_tier(prob)
            results.append({
                "churn_probability":  round(prob, 4),
                "churn_prediction":   int(prob >= PREDICTION_CUTOFF),
                "risk_tier":          tier,
                "recommended_action": action,
            })
        return results

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a full dataframe and append churn_prob + risk_tier columns.

        Parameters
        ----------
        df : raw dataframe (missing feature columns are filled with 0)

        Returns
        -------
        the model-ready rows plus 3 new columns: churn_prob, churn_pred, risk_tier
        """
        df = self._features(df)

        X = df[self.feature_cols].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]

        df["churn_prob"] = proba
        df["churn_pred"] = (proba >= PREDICTION_CUTOFF).astype(int)
        # Same lower-bound-inclusive rule as predict(); pd.cut would put 0.60 in Medium
        df["risk_tier"]  = [self._score_to_tier(float(p))[0] for p in proba]
        return df
