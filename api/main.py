"""
api/main.py
-----------
FastAPI REST API for the Telecom Churn Prediction Engine.

Endpoints:
    GET  /             → health check
    GET  /model-info   → model metadata and performance metrics
    POST /predict      → score a single customer
    POST /predict/batch → score a list of customers (max 1000)

    Developer API (api/v1.py) — API key required:
    POST /v1/datasets → POST /v1/models → GET /v1/models/{id} → POST /v1/models/{id}/predict[/csv]

Run locally:
    uvicorn api.main:app --reload --port 8000

Then visit:
    http://localhost:8000/docs   ← interactive Swagger UI
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from api.auth import rate_limit_headers_middleware
from api.errors import install_error_handlers
from api.v1 import get_registry, public as v1_public, router as v1_router
from src.predict import ChurnPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load the demo model once at startup ────────────────────────────────────────

MODEL_DIR = Path("models/")
predictor: Optional[ChurnPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = ChurnPredictor(MODEL_DIR)
        logger.info("ChurnPredictor loaded successfully")
    except FileNotFoundError:
        logger.warning(
            "Model files not found in models/. "
            "Run `python -m src.train` first."
        )
    interrupted = get_registry().recover_interrupted()
    if interrupted:
        logger.warning(f"Marked {interrupted} interrupted training job(s) as failed")
    yield


# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Telecom Churn Prediction Engine",
    description=(
        "Train a churn model on your own customer data, then score customers by churn risk "
        "with a recommended retention action.\n\n"
        "**Developer API (`/v1`)** — authenticate with `Authorization: Bearer <api key>`. "
        "Upload a labelled CSV to `/v1/datasets`, train with `/v1/models`, poll until `ready`, "
        "then score with `/v1/models/{id}/predict`. Errors always look like "
        "`{\"error\": {\"code\", \"message\", \"details\"}}`; branch on `code`.\n\n"
        "**Telecom demo** (`/predict`, no key): the original LightGBM model "
        "(test AUC-ROC 0.69 from the last full-data run).\n\n"
        "Built by Saumya — GCI World 2026 / personal portfolio project."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # Key auth uses a header, not cookies, so open CORS doesn't expose ambient credentials
    expose_headers=["Content-Disposition", "X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
)
app.middleware("http")(rate_limit_headers_middleware)
install_error_handlers(app)

app.include_router(v1_public)
app.include_router(v1_router)

# ── Schemas ────────────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """
    Raw customer feature inputs.
    All fields match the original dataset column names.
    Provide as many as you have (at least one) — missing numeric fields count as 0,
    missing categorical fields as unknown.
    """

    model_config = ConfigDict(
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "months":        18,
                "eqpdays":       480,
                "rev_Mean":      55.2,
                "mou_Mean":      320.0,
                "change_mou":   -120.0,
                "change_rev":    -8.5,
                "totmrc_Mean":   42.0,
                "ovrmou_Mean":    0.0,
                "drop_vce_Mean":  3.2,
                "plcd_vce_Mean": 85.0,
                "comp_vce_Mean": 80.5,
                "custcare_Mean":  2.1,
                "area":          "CHICAGO IL",
                "crclscod":      "B",
            }
        },
    )

    # Key predictors (most impactful). Bounds reject nonsense, not unusual customers:
    # counts and durations can't be negative; trends and charges can (credits, refunds).
    months:         Optional[float] = Field(None, ge=0, le=600,  description="Tenure in months")
    eqpdays:        Optional[float] = Field(None, ge=-30, le=10_000, description="Days on current handset")
    rev_Mean:       Optional[float] = Field(None, ge=-1_000, le=100_000, description="Mean monthly revenue ($)")
    mou_Mean:       Optional[float] = Field(None, ge=0, le=100_000, description="Mean monthly minutes of use")
    change_mou:     Optional[float] = Field(None, ge=-100_000, le=100_000, description="Change in MOU vs 3-month avg")
    change_rev:     Optional[float] = Field(None, ge=-100_000, le=100_000, description="Change in revenue vs 3-month avg")
    totmrc_Mean:    Optional[float] = Field(None, ge=-1_000, le=100_000, description="Mean total monthly recurring charge")
    ovrmou_Mean:    Optional[float] = Field(None, ge=0, le=100_000, description="Mean overage minutes")
    drop_vce_Mean:  Optional[float] = Field(None, ge=0, le=100_000, description="Mean dropped voice calls")
    plcd_vce_Mean:  Optional[float] = Field(None, ge=0, le=100_000, description="Mean placed voice calls")
    comp_vce_Mean:  Optional[float] = Field(None, ge=0, le=100_000, description="Mean completed voice calls")
    custcare_Mean:  Optional[float] = Field(None, ge=0, le=100_000, description="Mean customer care calls")

    # Demographics / account (optional enrichment)
    area:           Optional[str]   = Field(None, max_length=100, description="Geographic area")
    crclscod:       Optional[str]   = Field(None, max_length=10,  description="Credit class code")
    new_cell:       Optional[str]   = Field(None, max_length=10,  description="New cell flag")
    asl_flag:       Optional[str]   = Field(None, max_length=10,  description="ASL flag")

    @model_validator(mode="after")
    def at_least_one_field(self):
        if all(v is None for v in self.model_dump().values()):
            raise PydanticCustomError(
                "empty_record", "Provide at least one customer field; an empty record can't be scored."
            )
        return self

    def to_model_input(self) -> dict:
        """Numeric gaps → 0 (the documented default); categorical gaps stay None so the
        encoders map them to their MISSING bucket instead of a real category."""
        out = {}
        for name, value in self.model_dump().items():
            is_text = name in _CATEGORICAL_FIELDS
            out[name] = value if (value is not None or is_text) else 0
        return out


_CATEGORICAL_FIELDS = {"area", "crclscod", "new_cell", "asl_flag"}


class PredictionResult(BaseModel):
    churn_probability:  float = Field(..., description="Probability of churning (0–1)")
    churn_prediction:   int   = Field(..., description="Binary churn prediction (0=stay, 1=churn)")
    risk_tier:          str   = Field(..., description="Low Risk / Medium Risk / High Risk")
    recommended_action: str   = Field(..., description="Suggested retention intervention")


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(
        ..., max_length=1000, description="List of customers to score (max 1000)"
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "ok",
        "service": "Telecom Churn Prediction Engine",
        "version": app.version,
        "docs":    "/docs",
        "developer_api": "/v1 — see /docs",
    }


@app.get("/model-info", tags=["Model"])
def model_info():
    """Return model metadata and performance metrics."""
    results_path = MODEL_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {"note": "Run src/train.py to generate results.json"}

    return {
        "model":           "LightGBM (gradient boosted trees)",
        "target":          "churn (binary: will customer cancel within 60 days?)",
        "training_data":   "100,000 US telecom customers (Company A dataset)",
        "feature_count":   results.get("feature_count"),
        "engineered_features": ["call_completion_rate", "drop_rate", "rev_per_min", "overage_flag"],
        "performance":     results.get("lightgbm", {}),
        "baseline":        results.get("logistic_regression", {}),
        "risk_tiers": {
            "High Risk":   "churn_probability >= 0.60",
            "Medium Risk": "0.40 <= churn_probability < 0.60",
            "Low Risk":    "churn_probability < 0.40",
        },
        "churn_prediction_cutoff": 0.5,
    }


def _require_predictor() -> ChurnPredictor:
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` first.",
        )
    return predictor


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Score a single customer and return their churn probability,
    risk tier, and recommended retention action.

    `churn_prediction` (yes/no at 0.5) and `risk_tier` (bands at 0.40 / 0.60) are
    deliberately independent: a 0.45 customer is predicted to stay but is still
    Medium Risk.
    """
    model = _require_predictor()
    try:
        return PredictionResult(**model.predict(customer.to_model_input()))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed.") from e


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """
    Score a batch of up to 1,000 customers in one request (one vectorized model call).
    Returns a list of prediction results in the same order.
    """
    model = _require_predictor()
    try:
        results = model.predict_batch([c.to_model_input() for c in request.customers])
    except Exception as e:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed.") from e

    return {
        "count":   len(results),
        "results": results,
    }
