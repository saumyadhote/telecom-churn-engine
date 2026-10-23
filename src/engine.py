"""
engine.py
---------
Train a churn model on *any* customer dataset ("bring your own data") and
score new customers with it.

Unlike src/train.py — which is hard-wired to the Client.csv / Record.csv
telecom schema — this works on whatever columns the uploaded CSV has:
    1. infer each column's role (src/schema.py)
    2. three-way split: train / validation (early stopping) / test (reported once)
    3. LightGBM on native categoricals, plus a one-hot logistic-regression baseline
    4. a report a non-expert can read: AUC vs baseline, risk tiers vs actual
       churn, top drivers, and warnings (e.g. a column that leaks the label)

Usage:
    from src.engine import train_pipeline
    pipeline, report = train_pipeline(df, target="Churn")
    scored = pipeline.score(new_customers_df)
"""

import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.evaluate import compute_metrics
from src.predict import TIER_THRESHOLDS
from src.preprocess import engineer_features
from src.schema import coerce_numeric, encode_target, infer_schema

logger = logging.getLogger(__name__)


# ── Settings ──────────────────────────────────────────────────────────────────

MIN_ROWS          = 200    # below this the test-set metrics are meaningless
MIN_PER_CLASS     = 50     # need enough churners *and* stayers
VAL_SIZE          = 0.2
TEST_SIZE         = 0.2
RANDOM_SEED       = 42
PREDICTION_CUTOFF = 0.5    # churn_prediction = 1 when probability ≥ this
LEAKAGE_AUC       = 0.97   # a single column this predictive probably leaks the label
MIN_INPUT_COVERAGE = 0.5   # scoring data must contain at least half the model's columns

LGBM_PARAMS = {
    "n_estimators":      2000,     # upper bound — early stopping picks the real count
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 20,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "random_state":      RANDOM_SEED,
    "verbose":           -1,
}

# Industry-neutral actions (src/predict.TIER_ACTIONS is worded for telecom)
TIER_ACTIONS = {
    "High Risk": (
        "Priority outreach: personal call from an account manager, "
        "tailored retention offer, review of their plan and pricing."
    ),
    "Medium Risk": (
        "Targeted campaign: personalised loyalty reward or discount, "
        "check-in email, satisfaction survey."
    ),
    "Low Risk": (
        "Passive retention: regular newsletters, referral programme invitation, "
        "feature education."
    ),
}

# Columns src/preprocess.engineer_features() needs — present in the telecom data
TELECOM_SOURCE_COLS = ["comp_vce_Mean", "plcd_vce_Mean", "drop_vce_Mean",
                       "rev_Mean", "mou_Mean", "ovrmou_Mean"]


class TrainingError(ValueError):
    """The dataset can't be trained on; the message is safe to show to users."""


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ChurnPipeline:
    """
    Everything needed to score new customers with a model trained on one
    uploaded dataset: column roles, category vocabularies, the model, and the
    tier thresholds. Saved and loaded as a single joblib file.
    """

    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        id_col: str | None,
        target: str,
        positive_label: str,
        tier_thresholds: dict[str, float],
        telecom_features: bool = False,
    ) -> None:
        self.numeric_cols     = numeric_cols
        self.categorical_cols = categorical_cols
        self.id_col           = id_col
        self.target           = target
        self.positive_label   = positive_label
        self.tier_thresholds  = tier_thresholds
        self.telecom_features = telecom_features
        self.categories: dict[str, list[str]] = {}
        self.model = None          # LGBMClassifier or sklearn Pipeline (whichever won)
        self.model_type: str | None = None

    # ── Features ──

    @property
    def input_cols(self) -> list[str]:
        """Raw columns a caller should send."""
        return self.numeric_cols + self.categorical_cols

    @property
    def feature_cols(self) -> list[str]:
        """Columns the model actually sees (inputs + any engineered features)."""
        extra = ["call_completion_rate", "drop_rate", "rev_per_min", "overage_flag"] \
            if self.telecom_features else []
        return self.input_cols + extra

    def fit_categories(self, df: pd.DataFrame) -> None:
        for col in self.categorical_cols:
            values = df[col].dropna().astype(str).str.strip()
            self.categories[col] = sorted(values.unique().tolist())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Raw rows → model-ready frame. Missing columns become NaN (LightGBM
        handles missing values natively); categories never seen in training
        also become NaN instead of raising.
        """
        X = pd.DataFrame(index=df.index)
        for col in self.numeric_cols:
            if col in df.columns:
                values = coerce_numeric(df[col])
                X[col] = values if values is not None else pd.to_numeric(df[col], errors="coerce")
            else:
                X[col] = np.nan
        for col in self.categorical_cols:
            raw = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
            text = raw.where(raw.isna(), raw.astype(str).str.strip())
            X[col] = pd.Categorical(text, categories=self.categories[col])

        if self.telecom_features:
            X = engineer_features(X)
        return X[self.feature_cols]

    def coverage(self, df: pd.DataFrame) -> tuple[float, list[str]]:
        """Share of the model's input columns present in df, and which are missing."""
        missing = [c for c in self.input_cols if c not in df.columns]
        return 1 - len(missing) / max(len(self.input_cols), 1), missing

    # ── Scoring ──

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.transform(df))[:, 1]

    def tier(self, proba: np.ndarray) -> np.ndarray:
        high, medium = self.tier_thresholds["High Risk"], self.tier_thresholds["Medium Risk"]
        return np.where(proba >= high, "High Risk",
                        np.where(proba >= medium, "Medium Risk", "Low Risk"))

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a dataframe of customers in one vectorized call.

        Returns a frame with the ID column (if the model has one and df has it)
        and churn_probability, churn_prediction, risk_tier, recommended_action.
        """
        proba = self.predict_proba(df)
        tiers = self.tier(proba)
        out = pd.DataFrame(index=df.index)
        if self.id_col and self.id_col in df.columns:
            out[self.id_col] = df[self.id_col]
        out["churn_probability"]  = np.round(proba, 4)
        out["churn_prediction"]   = (proba >= PREDICTION_CUTOFF).astype(int)
        out["risk_tier"]          = tiers
        out["recommended_action"] = [TIER_ACTIONS[t] for t in tiers]
        return out


# ── Training ──────────────────────────────────────────────────────────────────

def _validate_thresholds(thresholds: dict | None) -> dict[str, float]:
    if thresholds is None:
        return {"High Risk": TIER_THRESHOLDS["High Risk"], "Medium Risk": TIER_THRESHOLDS["Medium Risk"]}
    high, medium = float(thresholds["High Risk"]), float(thresholds["Medium Risk"])
    if not 0 < medium < high < 1:
        raise TrainingError("Tier thresholds must satisfy 0 < Medium Risk < High Risk < 1.")
    return {"High Risk": high, "Medium Risk": medium}


def prepare_labels(
    df: pd.DataFrame, target: str, positive_label: str | None = None
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Encode the label, drop unlabelled rows, and check there is enough data."""
    if target not in df.columns:
        raise TrainingError(f"Column {target!r} is not in the dataset.")
    try:
        y, positive_label = encode_target(df[target], positive_label)
    except ValueError as e:
        raise TrainingError(str(e)) from e

    labelled = y.notna()
    df, y = df[labelled], y[labelled].astype(int)

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if len(df) < MIN_ROWS:
        raise TrainingError(f"Need at least {MIN_ROWS} labelled rows; got {len(df)}.")
    if min(n_pos, n_neg) < MIN_PER_CLASS:
        raise TrainingError(
            f"Need at least {MIN_PER_CLASS} churned and {MIN_PER_CLASS} retained customers; "
            f"got {n_pos} churned and {n_neg} retained."
        )
    return df, y, positive_label


def _leakage_warnings(X: pd.DataFrame, y: pd.Series, numeric_cols: list[str]) -> list[str]:
    """Flag numeric columns that on their own almost perfectly predict churn."""
    warnings = []
    for col in numeric_cols:
        values = X[col]
        mask = values.notna()
        if mask.sum() < MIN_PER_CLASS or y[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y[mask], values[mask])
        if max(auc, 1 - auc) >= LEAKAGE_AUC:
            warnings.append(
                f"'{col}' alone predicts churn almost perfectly (AUC {max(auc, 1 - auc):.2f}). "
                "If it's only known after a customer leaves (e.g. a cancellation date), "
                "exclude it — otherwise the model won't work on current customers."
            )
    return warnings


def _categoricals_as_text(X: pd.DataFrame) -> pd.DataFrame:
    """sklearn's encoders can't take pandas Categoricals with NaN — stringify them."""
    return X.astype(object).where(X.notna(), "MISSING").astype(str)


def _fit_logistic(X_train, y_train, numeric_cols, categorical_cols):
    """Logistic regression with its own fair encoding: scaled numerics + one-hot categoricals."""
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
        ("cat", make_pipeline(
            FunctionTransformer(_categoricals_as_text, feature_names_out="one-to-one"),
            OneHotEncoder(handle_unknown="ignore", min_frequency=5),
        ), categorical_cols),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=1000, C=1.0))
    model.fit(X_train, y_train)
    return model


def _fit_lightgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # Stop on log-loss: smoother than AUC on small validation sets
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    return model


def _evaluate(model, X, y) -> tuple[dict, np.ndarray]:
    proba = model.predict_proba(X)[:, 1]
    return compute_metrics(y, (proba >= PREDICTION_CUTOFF).astype(int), proba), proba


def train_pipeline(
    df: pd.DataFrame,
    target: str,
    id_col: str | None = None,
    positive_label: str | None = None,
    exclude_cols: list[str] | None = None,
    tier_thresholds: dict | None = None,
) -> tuple[ChurnPipeline, dict]:
    """
    Train a churn model on an arbitrary labelled customer dataframe.

    Parameters
    ----------
    df              : raw dataframe, one row per customer, including the label
    target          : name of the churn label column
    id_col          : customer ID column (kept in scored output, never trained on);
                      guessed from column names if omitted
    positive_label  : which label value means "churned" (guessed if omitted)
    exclude_cols    : extra columns to leave out of training
    tier_thresholds : {"High Risk": float, "Medium Risk": float}

    Returns
    -------
    (pipeline, report)

    Raises
    ------
    TrainingError if the data is unusable (message is user-facing).
    """
    thresholds = _validate_thresholds(tier_thresholds)
    df, y, positive_label = prepare_labels(df, target, positive_label)
    exclude = set(exclude_cols or [])

    schema = infer_schema(df, target=target)
    if id_col is None:
        id_col = schema["suggested_id"]
    elif id_col not in df.columns:
        raise TrainingError(f"ID column {id_col!r} is not in the dataset.")

    used, ignored = {"numeric": [], "categorical": []}, []
    for col in schema["columns"]:
        name, role = col["name"], col["role"]
        if name in (target, id_col):
            continue
        if name in exclude:
            ignored.append({"name": name, "reason": "excluded by you"})
        elif role in used:
            used[role].append(name)
        else:
            ignored.append({"name": name, "reason": col["reason"]})

    if not used["numeric"] and not used["categorical"]:
        raise TrainingError("No usable feature columns were found besides the label and ID.")

    pipeline = ChurnPipeline(
        numeric_cols=used["numeric"],
        categorical_cols=used["categorical"],
        id_col=id_col,
        target=target,
        positive_label=positive_label,
        tier_thresholds=thresholds,
        telecom_features=all(c in used["numeric"] for c in TELECOM_SOURCE_COLS),
    )

    # Three-way split: early stopping sees validation only; test is touched once.
    idx_rest, idx_test = train_test_split(df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
    idx_train, idx_val = train_test_split(
        idx_rest, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y[idx_rest], random_state=RANDOM_SEED
    )
    pipeline.fit_categories(df.loc[idx_train])
    X = pipeline.transform(df)
    X_train, X_val, X_test = X.loc[idx_train], X.loc[idx_val], X.loc[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    logger.info(f"Training on {len(X_train):,} rows, {X.shape[1]} features "
                f"(val {len(X_val):,}, test {len(X_test):,})")
    candidates = {
        "lightgbm":            _fit_lightgbm(X_train, y_train, X_val, y_val),
        "logistic_regression": _fit_logistic(X_train, y_train, pipeline.numeric_cols, pipeline.categorical_cols),
    }

    # Choose the served model on validation AUC — the test set stays untouched until the report.
    val_auc = {name: _evaluate(m, X_val, y_val)[0]["auc_roc"] for name, m in candidates.items()}
    chosen = max(val_auc, key=val_auc.get)
    pipeline.model, pipeline.model_type = candidates[chosen], chosen

    test = {name: _evaluate(m, X_test, y_test) for name, m in candidates.items()}
    metrics, proba = test[chosen]
    lgbm = candidates["lightgbm"]

    report = {
        "rows":            int(len(df)),
        "churn_rate":      round(float(y.mean()), 4),
        "positive_label":  positive_label,
        "split":           {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "model_type":      chosen,
        "metrics":         metrics,
        "comparison":      {name: {"validation_auc": val_auc[name], "test": test[name][0]}
                            for name in candidates},
        "lightgbm_trees":  int(lgbm.best_iteration_ or lgbm.n_estimators),
        "tiers":           _tier_report(pipeline.tier(proba), y_test),
        "tier_thresholds": thresholds,
        "top_features":    _top_features(pipeline, X.columns),
        "features_used":   {"numeric": pipeline.numeric_cols, "categorical": pipeline.categorical_cols,
                            "engineered_telecom_features": pipeline.telecom_features},
        "ignored_columns": ignored,
        "id_column":       id_col,
        "input_schema":    _input_schema(pipeline, df.loc[idx_train]),
        "warnings":        _leakage_warnings(X_train, y_train, pipeline.numeric_cols),
    }
    if metrics["auc_roc"] < 0.6:
        report["warnings"].append(
            f"Test AUC is {metrics['auc_roc']:.2f} — barely better than guessing (0.5). "
            "These columns may not carry much churn signal."
        )
    return pipeline, report


def _tier_report(tiers: np.ndarray, y_test: pd.Series) -> list[dict]:
    """How many test customers land in each tier, and how many of them really churned."""
    rows = []
    for tier in ("High Risk", "Medium Risk", "Low Risk"):
        mask = tiers == tier
        rows.append({
            "tier":              tier,
            "customers":         int(mask.sum()),
            "actual_churn_rate": round(float(y_test[mask].mean()), 4) if mask.any() else None,
        })
    return rows


def _top_features(pipeline: ChurnPipeline, columns, n: int = 10) -> list[dict]:
    """Share of the model's decision driven by each input column."""
    if pipeline.model_type == "lightgbm":
        weights = pd.Series(pipeline.model.booster_.feature_importance(importance_type="gain"),
                            index=[str(c) for c in columns])
    else:
        # |coefficient| on standardised inputs, summed over each column's one-hot levels
        pre, lr = pipeline.model[0], pipeline.model[-1]
        names = pre.get_feature_names_out()
        source = []
        for name in names:
            kind, rest = name.split("__", 1)
            if kind == "num":
                source.append(rest)
            else:
                source.append(max((c for c in pipeline.categorical_cols if rest.startswith(f"{c}_")), key=len))
        weights = pd.Series(np.abs(lr.coef_[0]), index=source).groupby(level=0).sum()

    total = weights.sum()
    share = (weights / total if total > 0 else weights).sort_values(ascending=False)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in share.head(n).items()]


def _input_schema(pipeline: ChurnPipeline, train_df: pd.DataFrame) -> list[dict]:
    """What each input column looks like — lets a UI build a scoring form."""
    schema = []
    for col in pipeline.numeric_cols:
        values = coerce_numeric(train_df[col])
        schema.append({
            "name": col, "type": "numeric",
            "min": float(values.min()), "median": float(values.median()), "max": float(values.max()),
        })
    for col in pipeline.categorical_cols:
        counts = train_df[col].dropna().astype(str).str.strip().value_counts()
        schema.append({
            "name": col, "type": "categorical",
            "categories": counts.index[:50].tolist(), "most_common": counts.index[0],
        })
    return schema
