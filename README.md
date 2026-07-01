# 📡 Telecom Churn Prediction Engine

> **ML-powered churn scoring system** — predicts which telecom customers will cancel within 60 days, segments them by risk tier, and recommends targeted retention actions via a REST API.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.692-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 Problem

A US telecom company with 100,000 customers has a **~50% churn rate** within 60 days but no system to predict *which* customers will leave before they do. Without prediction, every customer gets the same retention offer — wasting budget on low-risk customers and missing high-risk ones entirely.

**This project solves that** by training a LightGBM classifier that:
1. Scores every customer with a churn probability (0–1)
2. Segments them into three actionable risk tiers
3. Exposes predictions via a production-ready REST API

---

## 📊 Model Performance

| Model | AUC-ROC | F1-Score | Accuracy |
|---|---|---|---|
| Logistic Regression (baseline) | 0.604 | 0.562 | 57.7% |
| **LightGBM (final)** | **0.692** | **0.639** | **63.7%** |

Evaluated on a stratified 80/20 train-test split (20,000 held-out customers).

---

## 🔍 Key Findings from EDA

| Signal | Insight |
|---|---|
| **Usage trend** (`change_mou`) | Churners' usage declining faster — #1 predictor |
| **Device age** (`eqpdays`) | 45% of churners on devices 1–2+ years old vs 30% of stayers |
| **Revenue trend** (`change_rev`) | Revenue drop tracks usage — customers drifting before they leave |
| **Monthly bill** (`totmrc_Mean`) | High recurring charges without perceived value accelerate churn |
| **Revenue efficiency** (`rev_per_min`) | Engineered feature — captures value-for-money mismatch |

> **Key insight:** Churn is *not* a pricing problem — churners and stayers pay nearly identical amounts ($58.21 vs $59.22/month). Retention must target experience (device upgrades, service quality), not discounts.

---

## 🏗️ Project Structure

```
telecom-churn-engine/
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
│
├── data/
│   ├── README.md           ← how to get the dataset
│   └── sample_input.json   ← test the API without full data
│
├── notebooks/
│   ├── 01_eda.ipynb        ← exploratory data analysis
│   └── 02_modelling.ipynb  ← model training & evaluation
│
├── src/
│   ├── preprocess.py       ← data loading + feature engineering
│   ├── train.py            ← model training + artefact saving
│   ├── predict.py          ← ChurnPredictor class
│   └── evaluate.py         ← metrics + visualisation utilities
│
├── api/
│   └── main.py             ← FastAPI app (3 endpoints)
│
├── models/                 ← saved model artefacts (not in git)
│   ├── lgbm_churn.pkl
│   ├── encoders.pkl
│   ├── feature_cols.pkl
│   └── results.json
│
└── tests/
    ├── test_api.py         ← API endpoint tests
    └── test_preprocess.py  ← feature engineering unit tests
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/telecom-churn-engine.git
cd telecom-churn-engine
pip install -r requirements.txt
```

### 2. Add data

Place `Client.csv` and `Record.csv` in the `data/` folder.
See [`data/README.md`](data/README.md) for details.

### 3. Train the model

```bash
python -m src.train \
  --client data/Client.csv \
  --record data/Record.csv \
  --output models/
```

Output:
```
2024-xx-xx  Loading data...
2024-xx-xx  Loaded merged dataset: 100,000 rows × 100 columns
2024-xx-xx  Churn rate: 49.56%
2024-xx-xx  Training Logistic Regression baseline...
──────────────────────────────────────────────────
  Logistic Regression (baseline)
──────────────────────────────────────────────────
  AUC-ROC  : 0.6043
  F1 Score : 0.5622
  Accuracy : 0.5765
──────────────────────────────────────────────────
  LightGBM (final model)
──────────────────────────────────────────────────
  AUC-ROC  : 0.6927
  F1 Score : 0.6387
  Accuracy : 0.6370
2024-xx-xx  Model saved → models/lgbm_churn.pkl
```

### 4. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 🔌 API Reference

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

### `GET /model-info` — Model metadata and performance

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

**Test with the sample input:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/sample_input.json
```

---

## 🎯 Risk Tier → Retention Action

| Tier | Score | Customers | Churn Rate | Revenue at Risk | Recommended Action |
|---|---|---|---|---|---|
| 🔴 High Risk | > 60% | 29,290 | 82.9% | $16.7M/yr | Dedicated outreach, device upgrade, plan review |
| 🟡 Medium Risk | 40–60% | 41,023 | 49.5% | $14.0M/yr | Loyalty SMS, plan optimisation, satisfaction survey |
| 🟢 Low Risk | < 40% | 29,687 | 16.8% | $3.7M/yr | Automated email, referral programme |

**Projected business impact** (30% retention success rate, $15/customer intervention cost):
- Revenue saved: **$10.6M**
- Intervention cost: **$760K**
- Net uplift: **$9.9M — 14× ROI**

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_api.py::test_health_check              PASSED
tests/test_api.py::test_predict_returns_200       PASSED
tests/test_api.py::test_predict_response_schema   PASSED
tests/test_api.py::test_predict_probability_range PASSED
tests/test_api.py::test_predict_binary_prediction PASSED
tests/test_api.py::test_predict_valid_risk_tier   PASSED
tests/test_api.py::test_batch_predict             PASSED
tests/test_api.py::test_predict_with_minimal_fields PASSED
tests/test_preprocess.py::test_engineer_features_adds_columns PASSED
tests/test_preprocess.py::test_call_completion_rate_range     PASSED
tests/test_preprocess.py::test_overage_flag_binary            PASSED
...
```

---

## 🐳 Docker

```bash
# Build
docker build -t churn-engine .

# Run (after training — mount the models directory)
docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-engine
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Model | LightGBM 4.x |
| Baseline | scikit-learn LogisticRegression |
| Data processing | pandas, numpy |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Persistence | joblib |
| Testing | pytest + httpx |
| Containerisation | Docker |

---

## 📚 References

1. Statista (2024). Global Telecommunications Market Revenue. https://www.statista.com/statistics/1170056/global-telecom-market-revenue/
2. Bain & Company (2023). Customer Loyalty in Telecom. https://www.bain.com/insights/topics/customer-loyalty/
3. McKinsey & Company (2023). Telecom Churn and the AI Opportunity. https://www.mckinsey.com/industries/technology-media-and-telecommunications/
4. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017. https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
5. LightGBM Documentation. https://lightgbm.readthedocs.io/en/latest/
6. FastAPI Documentation. https://fastapi.tiangolo.com/

---
 
GCI World 2026 Spring 

---

*Dataset: Company A — GCI World 2026 Final Assignment, Matsuo-Iwasawa Laboratory, The University of Tokyo (2024). Raw data not included in this repository.*
