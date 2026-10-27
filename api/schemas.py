"""
api/schemas.py
--------------
Request and response models for the public /v1 API. These are the contract:
they drive validation, the OpenAPI spec at /docs, and the Python client.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

RiskTier = Literal["High Risk", "Medium Risk", "Low Risk"]
ModelStatus = Literal["queued", "training", "ready", "failed"]


# ── Errors ─────────────────────────────────────────────────────────────────────

class ErrorBody(BaseModel):
    code:    str = Field(..., description="Stable, machine-readable error code", examples=["model_not_found"])
    message: str = Field(..., description="Human-readable explanation")
    details: Optional[Any] = Field(None, description="Extra context, e.g. per-field validation errors")


class ErrorResponse(BaseModel):
    error: ErrorBody


# ── Account ────────────────────────────────────────────────────────────────────

class Limits(BaseModel):
    requests_per_minute: int
    max_models:          int
    trainings_per_day:   int
    max_upload_mb:       int


class Usage(BaseModel):
    models:          int
    trainings_today: int


class Account(BaseModel):
    id:         str
    name:       str
    email:      Optional[str]
    tier:       str
    created_at: str
    limits:     Limits
    usage:      Usage


class SignupRequest(BaseModel):
    name:  str = Field(..., min_length=1, max_length=100, examples=["Ada Lovelace"])
    email: Optional[str] = Field(None, max_length=200, examples=["ada@example.com"])


class SignupResponse(BaseModel):
    api_key: str = Field(..., description="Shown once — store it securely")
    account: Account


# ── Datasets ───────────────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    name:        str
    role:        Literal["numeric", "categorical", "id", "ignored", "target"]
    reason:      str
    dtype:       str
    missing_pct: float
    n_unique:    int
    samples:     list[str]


class Dataset(BaseModel):
    id:               str
    filename:         str
    rows:             int
    columns:          list[ColumnInfo]
    suggested_target: Optional[str] = Field(None, description="Best guess at the churn label column")
    suggested_id:     Optional[str] = Field(None, description="Best guess at the customer ID column")
    target_values:    Optional[list[str]] = Field(None, description="Values of the suggested target")
    created_at:       str


# ── Training ───────────────────────────────────────────────────────────────────

class TierThresholds(BaseModel):
    high:   float = Field(0.60, gt=0, lt=1, description="Probability at or above which a customer is High Risk")
    medium: float = Field(0.40, gt=0, lt=1, description="Probability at or above which a customer is Medium Risk")

    @model_validator(mode="after")
    def _ordered(self):
        if self.medium >= self.high:
            raise ValueError("medium must be lower than high")
        return self


class TrainRequest(BaseModel):
    dataset_id:      str
    target:          str = Field(..., description="Column that says whether the customer churned", examples=["churned"])
    id_column:       Optional[str] = Field(None, description="Customer ID column (guessed if omitted)")
    positive_label:  Optional[str] = Field(None, description="Value of the target meaning 'churned' (guessed if omitted)")
    exclude_columns: list[str] = Field(default_factory=list, description="Columns to leave out of training")
    tier_thresholds: TierThresholds = Field(default_factory=TierThresholds)
    name:            Optional[str] = Field(None, max_length=100)
    keep_dataset:    bool = Field(False, description="Keep the uploaded data after training (deleted by default)")


class Metrics(BaseModel):
    auc_roc:  float
    f1:       float
    accuracy: float


class Candidate(BaseModel):
    validation_auc: float
    test:           Metrics


class TierResult(BaseModel):
    tier:              RiskTier
    customers:         int = Field(..., description="Test-set customers placed in this tier")
    actual_churn_rate: Optional[float] = Field(None, description="Share of them who really churned")


class FeatureImportance(BaseModel):
    feature:    str
    importance: float = Field(..., description="Share of the model's decisions (sums to ~1)")


class InputColumn(BaseModel):
    name:        str
    type:        Literal["numeric", "categorical"]
    min:         Optional[float] = None
    median:      Optional[float] = None
    max:         Optional[float] = None
    categories:  Optional[list[str]] = None
    most_common: Optional[str] = None


class IgnoredColumn(BaseModel):
    name:   str
    reason: str


class TrainingReport(BaseModel):
    rows:            int
    churn_rate:      float
    positive_label:  str
    split:           dict[str, int]
    model_type:      Literal["lightgbm", "logistic_regression"]
    metrics:         Metrics = Field(..., description="Served model, on the held-out test set")
    comparison:      dict[str, Candidate]
    lightgbm_trees:  int
    tiers:           list[TierResult]
    tier_thresholds: dict[str, float]
    top_features:    list[FeatureImportance]
    features_used:   dict[str, Any]
    ignored_columns: list[IgnoredColumn]
    id_column:       Optional[str]
    input_schema:    list[InputColumn] = Field(..., description="Columns to send when scoring")
    warnings:        list[str]


class Model(BaseModel):
    id:               str
    name:             Optional[str]
    status:           ModelStatus
    target:           str
    positive_label:   str
    dataset_filename: str
    created_at:       str
    updated_at:       Optional[str] = None
    error:            Optional[str] = Field(None, description="Why training failed (status=failed)")
    report:           Optional[TrainingReport] = Field(None, description="Present once status=ready")


class ModelSummary(BaseModel):
    id:         str
    name:       Optional[str]
    status:     ModelStatus
    created_at: str
    auc_roc:    Optional[float]
    rows:       Optional[int]


class ModelList(BaseModel):
    data: list[ModelSummary]


# ── Scoring ────────────────────────────────────────────────────────────────────

MAX_BATCH = 1000


class PredictRequest(BaseModel):
    customers: list[dict[str, Any]] = Field(
        ..., min_length=1, max_length=MAX_BATCH,
        description="Customer rows keyed by the model's input column names (see report.input_schema). "
                    "At least half of those columns must be present; missing ones are treated as unknown.",
        examples=[[{"customer_id": "C1", "tenure_months": 3, "contract": "Month-to-month"}]],
    )


class Prediction(BaseModel):
    id:                 Optional[Any] = Field(None, description="Value of the model's ID column, if sent")
    churn_probability:  float
    churn_prediction:   int = Field(..., description="1 if churn_probability ≥ 0.5")
    risk_tier:          RiskTier
    recommended_action: str


class PredictResponse(BaseModel):
    count:           int
    missing_columns: list[str] = Field(..., description="Model inputs absent from the request (treated as unknown)")
    results:         list[Prediction]
