# churn-engine (Python client)

Train a churn model on your own customer data and score customers, from Python.

```bash
pip install "./clients/python[pandas]"    # from a checkout of this repo (not on PyPI yet)
```

```python
from churn_engine import ChurnEngine

client = ChurnEngine(api_key="ck_…", base_url="https://<host>")
```

`api_key` and `base_url` fall back to the `CHURN_ENGINE_API_KEY` and `CHURN_ENGINE_URL` environment variables.

## Train

```python
model = client.train("customers.csv")          # path, DataFrame, bytes, or file object
model.status        # "ready"
model.auc           # AUC-ROC on held-out customers
model.report        # full report: tiers vs actual churn, top drivers, warnings, input schema
model.input_columns # columns to send when scoring
```

`train()` uploads, starts training, and waits. Useful options: `target=` (otherwise auto-detected), `id_column=`, `positive_label=`, `exclude_columns=[...]`, `tier_thresholds={"high": 0.7, "medium": 0.4}`, `wait=False` (then `client.wait_for(model.id)`), `dataset_id=` to reuse an upload.

## Score

```python
# Send the columns in model.input_columns — at least half must be present; the rest count as unknown
client.predict(model.id, [{"customer_id": "C1", "tenure_months": 3, "contract": "Month-to-month", ...}])
# → [{"id": "C1", "churn_probability": 0.87, "churn_prediction": 1, "risk_tier": "High Risk", "recommended_action": "…"}]

client.predict(model.id, df)                                     # DataFrame; >1,000 rows are batched for you
scored = client.score(model.id, "current.csv", output="scored.csv")   # server-side CSV scoring → DataFrame
```

## Manage

```python
client.list_models()
client.get_model(model_id)
client.delete_model(model_id)
client.account()      # tier, limits, usage
```

## Errors

All exceptions subclass `ChurnEngineError` and carry `.status`, `.code`, `.message`, `.details`:

| Exception | When |
|---|---|
| `AuthenticationError` | Missing, invalid, or revoked key (401) |
| `NotFoundError` | Dataset/model doesn't exist or isn't yours (404) |
| `ValidationError` | Bad request or unusable data (422) — e.g. `code == "invalid_target"` |
| `RateLimitError` | Still rate-limited after `max_retries` automatic retries; see `.retry_after` |
| `TrainingFailedError` | Training finished with status `failed`; `.message` says why |
| `TrainingTimeoutError` | `train(wait=True)` gave up waiting (the model may still finish) |
