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
from pydantic import BaseModel, Field

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
        "**Telecom demo** (`/predict`, no key): the original LightGBM model, AUC-ROC 0.69.\n\n"
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
    Provide as many as you have — missing fields default to 0.
    """

    # Key predictors (most impactful)
    months:         Optional[float] = Field(None, description="Tenure in months")
    eqpdays:        Optional[float] = Field(None, description="Days on current handset")
    rev_Mean:       Optional[float] = Field(None, description="Mean monthly revenue ($)")
    mou_Mean:       Optional[float] = Field(None, description="Mean monthly minutes of use")
    change_mou:     Optional[float] = Field(None, description="Change in MOU vs 3-month avg")
    change_rev:     Optional[float] = Field(None, description="Change in revenue vs 3-month avg")
    totmrc_Mean:    Optional[float] = Field(None, description="Mean total monthly recurring charge")
    ovrmou_Mean:    Optional[float] = Field(None, description="Mean overage minutes")
    drop_vce_Mean:  Optional[float] = Field(None, description="Mean dropped voice calls")
    plcd_vce_Mean:  Optional[float] = Field(None, description="Mean placed voice calls")
    comp_vce_Mean:  Optional[float] = Field(None, description="Mean completed voice calls")
    custcare_Mean:  Optional[float] = Field(None, description="Mean customer care calls")

    # Demographics / account (optional enrichment)
    area:           Optional[str]   = Field(None, description="Geographic area")
    crclscod:       Optional[str]   = Field(None, description="Credit class code")
    new_cell:       Optional[str]   = Field(None, description="New cell flag")
    asl_flag:       Optional[str]   = Field(None, description="ASL flag")

    class Config:
        json_schema_extra = {
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
        }


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
        "version": "1.0.0",
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
        "features":        "100 raw + 4 engineered = 104 total",
        "performance":     results.get("lightgbm", {}),
        "baseline":        results.get("logistic_regression", {}),
        "risk_tiers": {
            "High Risk":   "churn_probability > 0.60",
            "Medium Risk": "churn_probability 0.40–0.60",
            "Low Risk":    "churn_probability < 0.40",
        },
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Score a single customer and return their churn probability,
    risk tier, and recommended retention action.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` first.",
        )

    customer_dict = customer.model_dump(exclude_none=False)
    # Replace None with 0 for numeric fields
    customer_dict = {k: (v if v is not None else 0) for k, v in customer_dict.items()}

    try:
        result = predictor.predict(customer_dict)
        return PredictionResult(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """
    Score a batch of up to 1,000 customers in one request.
    Returns a list of prediction results in the same order.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` first.",
        )

    results = []
    for customer in request.customers:
        customer_dict = customer.model_dump(exclude_none=False)
        customer_dict = {k: (v if v is not None else 0) for k, v in customer_dict.items()}
        results.append(predictor.predict(customer_dict))

    return {
        "count":   len(results),
        "results": results,
    }
