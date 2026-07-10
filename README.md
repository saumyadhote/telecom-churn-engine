# Telecom Churn Prediction Engine

> End-to-end ML system that predicts 60-day customer churn for a 100K-customer telecom dataset, segments customers into risk tiers, and serves real-time predictions via a production REST API.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-F7931E)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.692-4CAF50)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-11%20passed-brightgreen)](#running-tests)

---

## Overview

Telecom operators lose significant recurring revenue to churn without knowing *which* customers are at risk until it is too late. This project delivers a complete predictive solution: from raw usage records to a containerised API that scores any customer in milliseconds.

**What it does:**

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

## Project Structure

```
telecom-churn-engine/
├── data/
│   ├── README.md           ← dataset instructions
│   └── sample_input.json   ← API test payload
├── notebooks/
│   ├── 01_eda.ipynb        ← exploratory data analysis
│   └── 02_modelling.ipynb  ← training pipeline and evaluation
├── src/
│   ├── preprocess.py       ← data loading and feature engineering
│   ├── train.py            ← model training with artefact persistence
│   ├── predict.py          ← ChurnPredictor class with risk tier logic
│   └── evaluate.py         ← metrics and visualisation utilities
├── api/
│   └── main.py             ← FastAPI application
├── models/                 ← serialised artefacts (excluded from git)
├── tests/
│   ├── test_api.py         ← endpoint integration tests
│   └── test_preprocess.py  ← feature engineering unit tests
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
tests/test_api.py::test_health_check                PASSED
tests/test_api.py::test_predict_returns_200         PASSED
tests/test_api.py::test_predict_response_schema     PASSED
tests/test_api.py::test_predict_probability_range   PASSED
tests/test_api.py::test_predict_binary_prediction   PASSED
tests/test_api.py::test_predict_valid_risk_tier     PASSED
tests/test_api.py::test_batch_predict               PASSED
tests/test_api.py::test_predict_with_minimal_fields PASSED
tests/test_preprocess.py::test_engineer_features_adds_columns PASSED
tests/test_preprocess.py::test_call_completion_rate_range     PASSED
tests/test_preprocess.py::test_overage_flag_binary            PASSED
11 passed in 0.84s
```

---

## Docker

```bash
# Build
docker build -t churn-engine .

# Run (mount the trained models directory)
docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-engine
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
