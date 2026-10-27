> [!WARNING]
> **This project is on hold.** It is not actively developed or maintained right now, and there is no hosted deployment. The code runs locally as described below.

# Telecom Churn Prediction Engine

> End-to-end ML system that predicts 60-day customer churn for a 100K-customer telecom dataset, segments customers into risk tiers, and serves real-time predictions via a production REST API.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-F7931E)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.692-4CAF50)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-96%20passed-brightgreen)](#running-tests)

---

## Overview

Telecom operators lose significant recurring revenue to churn without knowing *which* customers are at risk until it is too late. This project delivers a complete predictive solution: from raw usage records to a containerised API that scores any customer in milliseconds.

**What it does:**

- **Bring your own data:** upload any labelled customer CSV, train a churn model on it in seconds, and score your current customers — no code, no fixed schema ([details](#use-it-on-your-own-data))
- Trains a LightGBM classifier on 100,000 customers with ~50% observed churn rate
- Engineers 10+ behavioural features from raw call records and billing data
- Classifies customers into High / Medium / Low risk tiers with prescriptive retention actions
- Serves predictions through a FastAPI REST interface with single and batch endpoints
- Projects **$9.9M net uplift** (14× ROI) at a 30% retention success rate

---

## Results

### Model Performance

| Model | AUC-ROC | F1-Score | Accuracy |
|---|---|---|---|
| Logistic Regression (baseline) | 0.604 | 0.562 | 57.7% |
| **LightGBM (final)** | **0.692** | **0.639** | **63.7%** |

Evaluated on a stratified 80/20 train-test split (20,000 held-out customers).

### Business Impact

| Risk Tier | Score Threshold | Customers | Churn Rate | Revenue at Risk |
|---|---|---|---|---|
| High Risk | > 60% | 29,290 | 82.9% | $16.7M / yr |
| Medium Risk | 40 – 60% | 41,023 | 49.5% | $14.0M / yr |
| Low Risk | < 40% | 29,687 | 16.8% | $3.7M / yr |

Projected outcome at 30% retention success and $15 / customer intervention cost:
- Revenue saved: **$10.6M**
- Intervention cost: **$760K**
- **Net uplift: $9.9M — 14× ROI**

---

## Key Findings

EDA across 100+ raw features surfaced five dominant signals:

| Feature | Insight |
|---|---|
| `change_mou` — usage trend | Churners' call volume declining sharply; strongest single predictor |
| `eqpdays` — device age | 45% of churners on devices ≥ 1 year vs 30% of stayers |
| `change_rev` — revenue trend | Revenue decay tracks usage drop; customers drift before they cancel |
| `totmrc_Mean` — monthly recurring charge | High recurring fees without perceived value accelerate exit |
| `rev_per_min` — revenue efficiency (engineered) | Captures value-for-money mismatch; churners pay more per minute of use |

**Critical insight:** Churn is not a pricing problem. Churners and stayers pay nearly identical monthly amounts ($58.21 vs $59.22). Effective retention must target experience — device upgrades and service quality — not blanket discounts.

---

## Architecture

```
Raw Data (Client.csv + Record.csv)
        │
        ▼
  src/preprocess.py       ← merge, clean, engineer features
        │
        ▼
  src/train.py            ← baseline → LightGBM, artefact serialisation
        │
        ▼
  models/                 ← lgbm_churn.pkl, encoders.pkl, feature_cols.pkl
        │
        ▼
  src/predict.py          ← ChurnPredictor class, risk tier logic
        │
        ▼
  api/main.py             ← FastAPI app (3 endpoints, Pydantic v2 validation)
        │
        ▼
  Docker container        ← portable, production-ready deployment
```

---

## Use it on your own data

The engine isn't tied to the telecom dataset. Any business with a history of customers and who left can train its own model:

1. **Upload** a CSV — one row per customer, plus a column saying whether they churned (`Yes/No`, `1/0`, `True/False`, `Churned/Stayed`, …).
2. **Check** how each column was read. The engine detects numeric and categorical columns, the customer ID, and the churn label, and sets aside columns it can't use (constant, mostly empty, free text) with a reason for each.
3. **Train.** It splits the data three ways — train / validation / test — and trains both LightGBM and a logistic regression (each with the encoding that suits it). Whichever scores better **on validation** is served; the test set is used once, for the numbers you see.
4. **Read the report:** AUC on customers the model never saw, how many test customers land in each risk tier and how many of them *actually* churned, the columns driving predictions, and warnings — e.g. a column that leaks the answer (a cancellation date) or a model barely better than guessing.
5. **Score** current customers — upload a CSV, download it with `churn_probability`, `risk_tier` and `recommended_action` appended.

**In the browser:** open `ui/engine.html` (with the API running) and click *Use the sample dataset* to try it with 3,000 synthetic subscription customers.

**From code** — see the [Developer API](#developer-api) below.

Requirements: at least 200 labelled rows, with at least 50 churned and 50 retained. Uploaded training data is deleted once a model trains successfully (pass `keep_dataset: true` to keep it); it's kept after a failure so you can retry. Don't upload names, emails or phone numbers — they don't help the model.

---

## Developer API

A versioned REST API at `/v1`, with interactive docs at `/docs` and the OpenAPI spec at `/openapi.json`.

### 1. Get an API key

Ask the operator for one, or — if the server runs with `CHURN_OPEN_SIGNUP=true` — create a free-tier key yourself:

```bash
curl -X POST https://<host>/v1/signup -H 'Content-Type: application/json' -d '{"name": "Ada"}'
# → {"api_key": "ck_…", "account": {...}}   (the key is shown once)
```

Send it on every request as `Authorization: Bearer ck_…` (or `X-API-Key: ck_…`). Each key's account sees only its own datasets and models.

### 2. Python

```bash
pip install "./clients/python[pandas]"
```

```python
from churn_engine import ChurnEngine

client = ChurnEngine(api_key="ck_…", base_url="https://<host>")   # or env CHURN_ENGINE_API_KEY / CHURN_ENGINE_URL

model = client.train("customers.csv")              # upload + train + wait; churn column auto-detected
print(model.auc, model.report["tiers"])

# Send the columns in model.input_columns — at least half must be present; the rest count as unknown
client.predict(model.id, [{"customer_id": "C1", "tenure_months": 3, "contract": "Month-to-month", ...}])
scored = client.score(model.id, "current_customers.csv", output="scored.csv")   # → DataFrame
```

Errors raise typed exceptions (`AuthenticationError`, `NotFoundError`, `ValidationError`, `RateLimitError`, `TrainingFailedError`); 429s are retried automatically, honouring `Retry-After`. See [`clients/python/README.md`](clients/python/README.md).

### 3. Any language (curl)

```bash
KEY=ck_…; API=https://<host>/v1
curl -H "Authorization: Bearer $KEY" -F file=@customers.csv $API/datasets                # → dataset id + schema
curl -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     -d '{"dataset_id": "<id>", "target": "churned"}' $API/models                        # → 202, status "queued"
curl -H "Authorization: Bearer $KEY" $API/models/<model_id>                              # poll until "ready"
curl -H "Authorization: Bearer $KEY" -F file=@current.csv $API/models/<model_id>/predict/csv -o scored.csv
```

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/health` | Liveness; says whether signup is open (no key needed) |
| `POST /v1/signup` | Self-serve free key, if enabled (no key needed) |
| `GET /v1/account` | Your tier, limits, and usage |
| `POST /v1/datasets` | Upload a training CSV; returns inferred column roles |
| `GET`/`DELETE /v1/datasets/{id}` | Inspect / remove an upload |
| `POST /v1/models` | Train (async); optional `id_column`, `positive_label`, `exclude_columns`, `tier_thresholds` |
| `GET /v1/models`, `GET /v1/models/{id}` | List models / status + full training report |
| `DELETE /v1/models/{id}` | Remove a model |
| `POST /v1/models/{id}/predict` | Score ≤ 1,000 customers as JSON |
| `POST /v1/models/{id}/predict/csv` | Upload a CSV, download it scored |

### Errors

Every error has the same shape; branch on `code`, which is stable:

```json
{"error": {"code": "model_not_ready", "message": "Model is training, not ready to score.", "details": {"status": "training"}}}
```

Common codes: `missing_api_key`, `invalid_api_key` (401) · `dataset_not_found`, `model_not_found` (404) · `model_not_ready`, `model_limit_reached` (409) · `file_too_large` (413) · `validation_error`, `invalid_target`, `insufficient_columns` (422) · `rate_limited`, `training_quota_exceeded` (429).

### Limits

| Tier | Requests / min | Models | Trainings / 24h | Upload |
|---|---|---|---|---|
| `free` (self-serve) | 60 | 5 | 20 | 20 MB |
| `standard` (issued) | 300 | 50 | 200 | 50 MB |

Responses carry `X-RateLimit-Limit` and `X-RateLimit-Remaining`; a 429 adds `Retry-After` (seconds).

### Running it yourself (operators)

```bash
python -m src.keys create --name "Ada" --email ada@example.com    # prints a key once (standard tier)
python -m src.keys list                                           # accounts, key prefixes, last used
python -m src.keys revoke ck_AbC1234                              # by prefix, full key, or key_ id
```

| Env var | Default | |
|---|---|---|
| `CHURN_STORAGE_DIR` | `storage/` | Uploads, models, and `accounts.db` (keys stored hashed) |
| `CHURN_OPEN_SIGNUP` | `false` | Allow `POST /v1/signup` (free tier, 5 signups/hour per IP) |
| `CHURN_MAX_CONCURRENT_TRAINING` | `1` | Training jobs run at once; the rest queue |

Rate limits are kept in memory, so they apply per server process — run a single instance, or move the limiter to Redis before scaling out.

---

## Project Structure

```
telecom-churn-engine/
├── data/
│   ├── README.md           ← dataset instructions
│   └── sample_input.json   ← API test payload
├── src/
│   ├── preprocess.py       ← telecom data loading and feature engineering
│   ├── train.py            ← telecom model training with artefact persistence
│   ├── predict.py          ← ChurnPredictor class with risk tier logic
│   ├── evaluate.py         ← metrics and visualisation utilities
│   ├── schema.py           ← infers column roles in any uploaded CSV
│   ├── engine.py           ← bring-your-own-data training + scoring pipeline
│   ├── registry.py         ← file-based storage for datasets and models
│   ├── accounts.py         ← developer accounts + hashed API keys (SQLite)
│   ├── keys.py             ← admin CLI: create / list / revoke keys
│   └── sample_data.py      ← synthetic example data generator
├── api/
│   ├── main.py             ← FastAPI app + telecom demo endpoints
│   ├── v1.py               ← public developer API (/v1)
│   ├── schemas.py          ← request/response contract (drives /docs)
│   ├── auth.py             ← API-key auth + rate limiting
│   └── errors.py           ← uniform {"error": {code, message}} responses
├── clients/python/         ← `churn-engine` Python client library
├── ui/
│   ├── index.html          ← project showcase page
│   ├── engine.html         ← upload → train → score web app
│   └── examples/           ← sample CSVs for trying the engine
├── models/                 ← serialised artefacts (excluded from git)
├── tests/
│   ├── test_api.py         ← telecom endpoint tests (model mocked)
│   ├── test_preprocess.py  ← feature engineering unit tests
│   ├── test_engine.py      ← schema inference + training engine (real models)
│   ├── test_v1_api.py      ← /v1: auth, isolation, limits, errors, full flow
│   ├── test_accounts.py    ← key storage + admin CLI
│   └── test_client.py      ← Python client against the real API
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/telecom-churn-engine.git
cd telecom-churn-engine
pip install -r requirements.txt
```

### 2. Add data

Place `Client.csv` and `Record.csv` in the `data/` folder.
See [`data/README.md`](data/README.md) for the full dataset specification.

### 3. Train the model

```bash
python -m src.train \
  --client data/Client.csv \
  --record data/Record.csv \
  --output models/
```

Expected output:
```
Loading data...
Loaded merged dataset: 100,000 rows × 100 columns
Churn rate: 49.56%

──────────────────────────────────────────────
  Logistic Regression (baseline)
──────────────────────────────────────────────
  AUC-ROC  : 0.6043
  F1 Score : 0.5622
  Accuracy : 0.5765
──────────────────────────────────────────────
  LightGBM (final model)
──────────────────────────────────────────────
  AUC-ROC  : 0.6927
  F1 Score : 0.6387
  Accuracy : 0.6370

Model saved → models/lgbm_churn.pkl
```

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs available at **http://localhost:8000/docs**.

---

## API Reference

### `POST /predict` — Score a single customer

**Request:**
```json
{
  "months": 18,
  "eqpdays": 480,
  "rev_Mean": 55.2,
  "mou_Mean": 320.0,
  "change_mou": -120.0,
  "change_rev": -8.5,
  "totmrc_Mean": 42.0,
  "drop_vce_Mean": 3.2,
  "plcd_vce_Mean": 85.0,
  "comp_vce_Mean": 80.5,
  "custcare_Mean": 2.1
}
```

**Response:**
```json
{
  "churn_probability": 0.74,
  "churn_prediction": 1,
  "risk_tier": "High Risk",
  "recommended_action": "Priority outreach: dedicated account manager call, device upgrade offer, customised plan review."
}
```

### `POST /predict/batch` — Score up to 1,000 customers

```json
{
  "customers": [ { ...customer1 }, { ...customer2 } ]
}
```

### `GET /model-info` — Model metadata and performance metrics

```json
{
  "model": "LightGBM (gradient boosted trees)",
  "performance": { "auc_roc": 0.6927, "f1": 0.6387, "accuracy": 0.637 },
  "risk_tiers": {
    "High Risk":   "churn_probability > 0.60",
    "Medium Risk": "churn_probability 0.40–0.60",
    "Low Risk":    "churn_probability < 0.40"
  }
}
```

**Test with the sample payload:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/sample_input.json
```

---

## Running Tests

```bash
pytest tests/ -v
```

```
tests/test_accounts.py     11 passed   (key storage + admin CLI)
tests/test_api.py           9 passed   (telecom endpoints, model mocked)
tests/test_client.py       12 passed   (Python client against the real API)
tests/test_engine.py       24 passed   (schema inference + engine, real models)
tests/test_preprocess.py    7 passed
tests/test_v1_api.py       33 passed   (auth, isolation, limits, errors, full flow)
96 passed
```

---

## Docker

```bash
# Build
docker build -t churn-engine .

# Run (mount the telecom models, and a folder to keep uploaded datasets / trained models)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/storage:/app/storage \
  churn-engine
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | LightGBM 4.x (gradient boosted trees) |
| Baseline | scikit-learn LogisticRegression |
| Data processing | pandas, NumPy |
| API framework | FastAPI + Uvicorn |
| Schema validation | Pydantic v2 |
| Model persistence | joblib |
| Testing | pytest + httpx |
| Containerisation | Docker |

---

## References

1. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
2. [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)
3. [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

*Dataset: GCI World 2026 Final Assignment, Matsuo-Iwasawa Laboratory, The University of Tokyo. Raw data not included in this repository.*
