"""
api/main.py
-----------
FastAPI REST API for the Telecom Churn Prediction Engine.
It also serves the website (ui/index.html), so one command runs everything.

HOW TO RUN
==========
Easiest, from the project root:

    ./run.sh

Or by hand:

    source myenv/bin/activate              # activate your virtualenv
    pip install -r requirements.txt        # first time only
    uvicorn api.main:app --reload --port 8000

Then open:
    http://localhost:8000/ui/    ← the website (churn scoring UI)
    http://localhost:8000/docs   ← interactive Swagger API docs (has a link to the website)
    http://localhost:8000/       ← health check (JSON)

The model files must exist in models/. If they don't, train first:
    python -m src.train --client data/Client.csv --record data/Record.csv --output models/

Endpoints:
    GET  /              → health check
    GET  /ui/           → the website
    GET  /model-info    → model metadata and performance metrics
    POST /predict       → score a single customer
    POST /predict/batch → score a list of customers (max 1000)
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import ChurnPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────

API_DESCRIPTION = """
### 👉 [Open the website](/ui/)

Predicts how likely a telecom customer is to **cancel within 60 days**, puts them in a
**risk tier**, and suggests a **retention action**.

**Model:** LightGBM · AUC-ROC 0.692 · F1 0.636 · accuracy 63.5%
(measured on 20,000 customers the model never saw).

---

### What you can ask this API

| Question | Endpoint |
|---|---|
| Is the service running? | `GET /` |
| How good is the model, and what was it trained on? | `GET /model-info` |
| How likely is *this* customer to leave, and what should we do? | `POST /predict` |
| Which of these customers (up to 1,000) are most at risk? | `POST /predict/batch` |

### How to try it on this page

1. Click an endpoint below (for example **POST /predict**).
2. Click **Try it out**: an example customer is already filled in.
3. Change any numbers you like, then click **Execute**. The answer appears under **Responses**.

### The inputs that matter most

| Field | Meaning |
|---|---|
| `months` | How long they've been a customer (months) |
| `eqpdays` | Days since their last phone upgrade |
| `rev_Mean` | Average monthly bill ($) |
| `change_mou` | Change in minutes used vs their recent average (negative = using less) |
| `change_rev` | Change in monthly bill vs their recent average ($) |
| `totmrc_Mean` | Fixed monthly plan charge ($) |

Every field is optional. **Fields you leave out are treated as 0**, so sending more of them gives
more realistic answers.

### Reading the answer

| Field | Meaning |
|---|---|
| `churn_probability` | Estimated chance (0–1) they leave within 60 days |
| `churn_prediction` | 1 = likely to leave (probability ≥ 0.5), 0 = likely to stay |
| `risk_tier` | **High Risk** ≥ 0.60 · **Medium Risk** 0.40–0.60 · **Low Risk** < 0.40 |
| `recommended_action` | What the retention team should do for that tier |

*Built from the University of Tokyo Matsuo-Iwasawa Lab GCI World course (2026) final assignment.*
"""

TAGS_METADATA = [
    {"name": "Prediction", "description": "Score customers: churn probability, risk tier and recommended action."},
    {"name": "Model", "description": "What the model is, what data it learned from, and how well it performs."},
    {"name": "Health", "description": "Check the service is up."},
]

app = FastAPI(
    title="Telecom Churn Prediction Engine",
    description=API_DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=TAGS_METADATA,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Website (static UI) ─────────────────────────────────────────────────────────
# Serves ui/index.html at http://localhost:8000/ui/ so you don't need a
# separate web server. Path is resolved from this file, so it works no matter
# which directory you start uvicorn from.

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

    @app.get("/website", include_in_schema=False)
    def website_redirect():
        """Short alias: /website → /ui/"""
        return RedirectResponse(url="/ui/")


# ── Load model once at startup ─────────────────────────────────────────────────

MODEL_DIR = Path("models/")
MODEL_NOT_LOADED = (
    "Model not loaded. Check the server log: either the model files are "
    "missing (run `python -m src.train`) or LightGBM couldn't start "
    "(on macOS: `brew install libomp`)."
)
predictor: Optional[ChurnPredictor] = None


@app.on_event("startup")
def load_model() -> None:
    global predictor
    try:
        predictor = ChurnPredictor(MODEL_DIR)
        logger.info("ChurnPredictor loaded successfully")
        logger.info("Website:  http://localhost:8000/ui/")
        logger.info("API docs: http://localhost:8000/docs")
    except FileNotFoundError:
        logger.warning(
            "Model files not found in models/. "
            "Run `python -m src.train` first."
        )
    except OSError as exc:
        # Typical on macOS: LightGBM needs the OpenMP runtime (libomp).
        # Keep the server up so the website and /docs still load;
        # /predict will return 503 until this is fixed.
        hint = (
            " On macOS install it with `brew install libomp`."
            if "libomp" in str(exc) else ""
        )
        logger.error("Could not load the model: %s.%s", exc, hint)


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

@app.get("/", tags=["Health"], summary="Is the service running?")
def root():
    """Returns `status: ok` when the API is up, plus links to the docs and the website."""
    return {
        "status":  "ok",
        "service": "Telecom Churn Prediction Engine",
        "version": "1.0.0",
        "docs":    "/docs",
        "website": "/ui/",
    }


@app.get("/model-info", tags=["Model"], summary="How good is the model, and what was it trained on?")
def model_info():
    """
    Describes the model: algorithm, prediction target, training data, number of
    features, test-set scores for the final model and the baseline, and the
    risk-tier cut-offs.
    """
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
        "features":        (
            f"{results['feature_count']} after preprocessing (raw columns + 4 engineered)"
            if "feature_count" in results else "raw columns + 4 engineered"
        ),
        "performance":     results.get("lightgbm", {}),
        "baseline":        results.get("logistic_regression", {}),
        "risk_tiers": {
            "High Risk":   "churn_probability >= 0.60",
            "Medium Risk": "churn_probability 0.40–0.60",
            "Low Risk":    "churn_probability < 0.40",
        },
    }


@app.post(
    "/predict",
    response_model=PredictionResult,
    tags=["Prediction"],
    summary="How likely is this customer to leave, and what should we do?",
)
def predict(customer: CustomerFeatures):
    """
    Send one customer's numbers; get back their churn probability, a 0/1
    prediction, their risk tier and a recommended retention action.

    All fields are optional and missing ones count as 0. The six most useful are
    `months`, `eqpdays`, `rev_Mean`, `change_mou`, `change_rev` and `totmrc_Mean`.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=MODEL_NOT_LOADED,
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


@app.post(
    "/predict/batch",
    tags=["Prediction"],
    summary="Which of these customers are most at risk?",
)
def predict_batch(request: BatchRequest):
    """
    Score up to 1,000 customers in one request, for example a daily export of
    all active customers. Send `{"customers": [ {...}, {...} ]}`; results come
    back in the same order, one per customer.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=MODEL_NOT_LOADED,
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
