# Telecom Churn Prediction Engine

> Predicts 60-day customer churn on a 100,000-customer telecom dataset, groups customers into High / Medium / Low risk tiers, and serves predictions through a REST API.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-F7931E)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-16-brightgreen)](#running-tests)

---

## Overview

Telecom operators lose recurring revenue when customers leave. This project predicts which customers are likely to churn within 60 days, so retention effort can be directed at the highest-risk customers first.

Built from the final assignment of the University of Tokyo Matsuo-Iwasawa Lab's GCI World data science course (April 2026) and extended into an end-to-end prediction service.

**What it does:**

- Trains a LightGBM classifier on a 100,000-customer dataset (about 50% churn rate in the data)
- Adds 4 engineered features: call completion rate, drop rate, revenue per minute, and an overage flag
- Groups customers into High / Medium / Low risk tiers using probability thresholds, each with a suggested retention action
- Serves predictions through a FastAPI REST interface with single and batch endpoints

---

## Results

| Model | AUC-ROC | F1-Score | Accuracy |
|---|---|---|---|
| Logistic Regression (baseline) | 0.6045 | 0.5564 | 57.6% |
| **LightGBM (final)** | **0.6915** | **0.6363** | **63.5%** |

Evaluated on a stratified 80/20 train-test split (20,000 held-out customers). Figures are from `models/results.json`, written by the latest training run. LightGBM beats the baseline, but an AUC of about 0.69 is modest, not strong.

### Risk tiers

| Tier | Churn probability |
|---|---|
| High Risk | 0.60 or above |
| Medium Risk | 0.40 to 0.60 |
| Low Risk | below 0.40 |

Thresholds are fixed choices and have not been tuned to a cost model.

---

## Features

The model uses about 100 raw columns plus 4 engineered features (102 in total after preprocessing). Columns of particular interest:

| Feature | Meaning |
|---|---|
| `change_mou` | Change in minutes of use compared with the recent average |
| `eqpdays` | Days on the current handset |
| `change_rev` | Change in revenue compared with the recent average |
| `totmrc_Mean` | Mean total monthly recurring charge |
| `rev_per_min` (engineered) | Revenue divided by minutes of use |
| `call_completion_rate` (engineered) | Completed calls divided by placed calls |
| `drop_rate` (engineered) | Dropped calls divided by placed calls |
| `overage_flag` (engineered) | 1 if the customer had any overage minutes, otherwise 0 |

---

## Architecture

```
Raw Data (Client.csv + Record.csv)
        │
        ▼
  src/preprocess.py       ← merge, engineer features, encode categoricals
        │
        ▼
  src/train.py            ← baseline → LightGBM, save model files
        │
        ▼
  models/                 ← lgbm_churn.pkl, encoders.pkl, feature_cols.pkl, results.json
        │
        ▼
  src/predict.py          ← ChurnPredictor class, risk tier logic
        │
        ▼
  api/main.py             ← FastAPI app (4 endpoints, Pydantic v2 validation)
```

---

## Project Structure

```
telecom-churn-engine/
├── data/
│   ├── README.md           ← dataset instructions
│   └── sample_input.json   ← API test payload
├── src/
│   ├── preprocess.py       ← data loading and feature engineering
│   ├── train.py            ← model training and saving
│   ├── predict.py          ← ChurnPredictor class with risk tier logic
│   └── evaluate.py         ← metrics and plotting utilities
├── api/
│   └── main.py             ← FastAPI application
├── tests/
│   ├── test_api.py         ← API endpoint tests (9)
│   └── test_preprocess.py  ← feature engineering tests (7)
├── ui/                     ← static demo page
├── reports/figures/
├── models/                 ← saved model files (not tracked in git)
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/saumyadhote/telecom-churn-engine.git
cd telecom-churn-engine
pip install -r requirements.txt
```

### 2. Add data

Place `Client.csv` and `Record.csv` in the `data/` folder. See [`data/README.md`](data/README.md) for the dataset description. The raw data is not included in this repository.

### 3. Train the model

```bash
python -m src.train \
  --client data/Client.csv \
  --record data/Record.csv \
  --output models/
```

This prints the metrics for both models and saves the model files to `models/`.

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs are available at http://localhost:8000/docs.

---

## API Reference

| Endpoint | Purpose |
|---|---|
| `GET /` | Health check |
| `GET /model-info` | Model description and metrics from `results.json` |
| `POST /predict` | Score one customer |
| `POST /predict/batch` | Score up to 1,000 customers |

Fields left out of a request are treated as 0.

### `POST /predict`

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

**Response (example shape):**
```json
{
  "churn_probability": 0.74,
  "churn_prediction": 1,
  "risk_tier": "High Risk",
  "recommended_action": "Priority outreach: dedicated account manager call, device upgrade offer, customised plan review."
}
```

### `POST /predict/batch`

```json
{
  "customers": [ { ...customer1 }, { ...customer2 } ]
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

There are 16 tests:

- `tests/test_api.py` (9): health check, response schema, probability range, binary prediction, valid risk tier, batch prediction, minimal fields, model-info endpoint, status code. These use a mocked predictor, so they check how the API behaves, not the quality of predictions.
- `tests/test_preprocess.py` (7): engineered columns are added, call completion rate is in range, overage flag is binary and correct, categorical encoding has no missing values and returns numbers, no infinite values after feature engineering.

---

## Docker

A Dockerfile is included to package the API (Python 3.11 slim image, installs the requirements, runs the app with uvicorn on port 8000).

```bash
docker build -t churn-engine .
docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-engine
```

It has not been deployed to a hosted environment.

---

## Limitations

- **Modest accuracy.** AUC of about 0.69 means the model ranks customers better than the baseline, but it is far from perfect.
- **Missing values are handled differently at training and prediction time.** Training fills gaps with each column's median. The API fills omitted fields with 0, which can shift predictions for sparse requests.
- **Test scores are slightly optimistic.** Early stopping during training watches the same test split used for the final scores. A separate validation split would be cleaner.
- **API tests use a mocked model**, so they do not check prediction quality.
- **CORS allows all origins.** This is a demo setting and should be restricted in a real deployment.
- **Risk-tier thresholds are fixed** and not tuned to retention cost.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | LightGBM 4.x (gradient boosted trees) |
| Baseline | scikit-learn LogisticRegression |
| Data processing | pandas, NumPy |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Model persistence | joblib |
| Testing | pytest + httpx |
| Packaging | Docker (Dockerfile included) |

---

## References

1. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
2. [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)
3. [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

*Dataset: GCI World 2026 Final Assignment, Matsuo-Iwasawa Laboratory, The University of Tokyo. Raw data not included in this repository.*
