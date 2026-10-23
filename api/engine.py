"""
api/engine.py
-------------
"Bring your own data" endpoints: upload a labelled customer CSV, train a churn
model on it, then score new customers with that model.

Endpoints:
    POST   /datasets                 → upload a training CSV, get its inferred schema
    GET    /datasets/{id}            → schema of an uploaded dataset
    DELETE /datasets/{id}
    POST   /models                   → train a model on a dataset (runs in background)
    GET    /models                   → list trained models
    GET    /models/{id}              → status, metrics, tiers, drivers, input schema
    DELETE /models/{id}
    POST   /models/{id}/predict      → score up to 1,000 customers sent as JSON
    POST   /models/{id}/predict/csv  → upload a CSV, download it scored
"""

import io
import logging
import os
import re
import threading
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from src.engine import MIN_INPUT_COVERAGE, TrainingError, prepare_labels, train_pipeline
from src.registry import NotFound, Registry
from src.schema import infer_schema, target_values

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Bring your own data"])

STORAGE_DIR    = os.environ.get("CHURN_STORAGE_DIR", "storage/")
MAX_UPLOAD_MB  = float(os.environ.get("CHURN_MAX_UPLOAD_MB", "50"))
MAX_BATCH      = 1000
MAX_CONCURRENT_TRAINING = int(os.environ.get("CHURN_MAX_CONCURRENT_TRAINING", "1"))

_registry: Registry | None = None
_training_slots = threading.Semaphore(MAX_CONCURRENT_TRAINING)


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry(STORAGE_DIR)
    return _registry


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    """Read an uploaded CSV with a size cap; turn parse failures into 400s."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")

    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB:g} MB.")

    try:
        df = pd.read_csv(io.BytesIO(data), low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise HTTPException(400, f"Couldn't read that CSV: {e}")
    if df.empty or df.shape[1] < 2:
        raise HTTPException(400, "The CSV needs a header row, at least 2 columns, and at least 1 data row.")
    return df


def _model_or_404(registry: Registry, model_id: str) -> dict:
    try:
        return registry.get_model(model_id)
    except NotFound:
        raise HTTPException(404, f"No model with id {model_id!r}.")


def _ready_pipeline(registry: Registry, model_id: str):
    meta = _model_or_404(registry, model_id)
    if meta["status"] != "ready":
        raise HTTPException(409, f"Model is {meta['status']}, not ready to score.")
    return registry.load_pipeline(model_id)


def _score(pipeline, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    coverage, missing = pipeline.coverage(df)
    if coverage < MIN_INPUT_COVERAGE:
        raise HTTPException(
            422,
            f"This data has only {coverage:.0%} of the columns the model was trained on. "
            f"Missing: {', '.join(missing[:20])}{'…' if len(missing) > 20 else ''}",
        )
    return pipeline.score(df), missing


# ── Datasets ───────────────────────────────────────────────────────────────────

@router.post("/datasets", status_code=201)
def upload_dataset(file: UploadFile = File(...), registry: Registry = Depends(get_registry)):
    """
    Upload a labelled customer CSV (one row per customer, including a column
    that says whether they churned). Returns how each column was interpreted
    and a guess at the churn and ID columns.
    """
    df = _read_csv_upload(file)
    schema = infer_schema(df)
    meta = registry.save_dataset(df, file.filename, schema)
    if schema["suggested_target"]:
        meta["target_values"] = target_values(df[schema["suggested_target"]])
    return meta


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, registry: Registry = Depends(get_registry)):
    try:
        return registry.get_dataset(dataset_id)
    except NotFound:
        raise HTTPException(404, f"No dataset with id {dataset_id!r}.")


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, registry: Registry = Depends(get_registry)):
    try:
        registry.delete_dataset(dataset_id)
    except NotFound:
        raise HTTPException(404, f"No dataset with id {dataset_id!r}.")


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
    target:          str = Field(..., description="Column that says whether the customer churned")
    id_column:       Optional[str] = Field(None, description="Customer ID column (guessed if omitted)")
    positive_label:  Optional[str] = Field(None, description="Value of the target meaning 'churned' (guessed if omitted)")
    exclude_columns: list[str] = Field(default_factory=list, description="Columns to leave out of training")
    tier_thresholds: TierThresholds = Field(default_factory=TierThresholds)
    name:            Optional[str] = Field(None, max_length=100)
    keep_dataset:    bool = Field(False, description="Keep the uploaded data after training (deleted by default)")


def run_training(registry: Registry, model_id: str, request: TrainRequest, df: pd.DataFrame) -> None:
    """
    Background job: train, save the pipeline, record the report. The uploaded
    data is deleted after a successful run (unless keep_dataset) and kept after
    a failure, so the user can retry with different settings.
    """
    with _training_slots:
        registry.update_model(model_id, status="training")
        try:
            pipeline, report = train_pipeline(
                df,
                target=request.target,
                id_col=request.id_column,
                positive_label=request.positive_label,
                exclude_cols=request.exclude_columns,
                tier_thresholds={"High Risk": request.tier_thresholds.high,
                                 "Medium Risk": request.tier_thresholds.medium},
            )
            registry.save_pipeline(model_id, pipeline)
            registry.update_model(model_id, status="ready", report=report)
        except TrainingError as e:
            registry.update_model(model_id, status="failed", error=str(e))
            return
        except Exception:
            logger.exception(f"Training {model_id} crashed")
            registry.update_model(model_id, status="failed", error="Unexpected error while training.")
            return

    if not request.keep_dataset:
        try:
            registry.delete_dataset(request.dataset_id)
        except NotFound:
            pass


@router.post("/models", status_code=202)
def train_model(
    request: TrainRequest,
    background: BackgroundTasks,
    registry: Registry = Depends(get_registry),
):
    """
    Start training a churn model on an uploaded dataset. Returns immediately
    with status "queued"; poll GET /models/{id} until it is "ready" or "failed".
    """
    try:
        dataset = registry.get_dataset(request.dataset_id)
        df = registry.load_dataset(request.dataset_id)
    except NotFound:
        raise HTTPException(404, f"No dataset with id {request.dataset_id!r}.")

    # Fail fast on problems we can spot without training
    try:
        _, _, positive_label = prepare_labels(df, request.target, request.positive_label)
    except TrainingError as e:
        raise HTTPException(422, str(e))
    if request.id_column and request.id_column not in df.columns:
        raise HTTPException(422, f"ID column {request.id_column!r} is not in the dataset.")

    meta = registry.create_model(
        name=request.name or dataset["filename"],
        dataset_filename=dataset["filename"],
        target=request.target,
        positive_label=positive_label,
    )
    background.add_task(run_training, registry, meta["id"], request, df)
    return meta


# ── Models ─────────────────────────────────────────────────────────────────────

@router.get("/models")
def list_models(registry: Registry = Depends(get_registry)):
    """All models, newest first (summary only — fetch one for the full report)."""
    summary = []
    for meta in registry.list_models():
        report = meta.get("report") or {}
        summary.append({
            "id":         meta["id"],
            "name":       meta.get("name"),
            "status":     meta["status"],
            "created_at": meta["created_at"],
            "auc_roc":    report.get("metrics", {}).get("auc_roc"),
            "rows":       report.get("rows"),
        })
    return summary


@router.get("/models/{model_id}")
def get_model(model_id: str, registry: Registry = Depends(get_registry)):
    return _model_or_404(registry, model_id)


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: str, registry: Registry = Depends(get_registry)):
    _model_or_404(registry, model_id)
    registry.delete_model(model_id)


class ScoreRequest(BaseModel):
    customers: list[dict[str, Any]] = Field(
        ..., min_length=1, max_length=MAX_BATCH,
        description="Customer rows, keyed by the column names the model was trained on",
    )


@router.post("/models/{model_id}/predict")
def predict(model_id: str, request: ScoreRequest, registry: Registry = Depends(get_registry)):
    """Score up to 1,000 customers with a trained model, in one vectorized call."""
    pipeline = _ready_pipeline(registry, model_id)
    scored, missing = _score(pipeline, pd.DataFrame(request.customers))
    return {
        "count":           len(scored),
        "missing_columns": missing,
        "results":         scored.to_dict(orient="records"),
    }


@router.post("/models/{model_id}/predict/csv")
def predict_csv(model_id: str, file: UploadFile = File(...), registry: Registry = Depends(get_registry)):
    """
    Upload a CSV of current customers; download the same CSV with
    churn_probability, churn_prediction, risk_tier, and recommended_action appended.
    """
    pipeline = _ready_pipeline(registry, model_id)
    df = _read_csv_upload(file)
    scored, _ = _score(pipeline, df)
    out = pd.concat([df, scored.drop(columns=[pipeline.id_col], errors="ignore")], axis=1)

    buffer = io.StringIO()
    out.to_csv(buffer, index=False)
    buffer.seek(0)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(file.filename))[0])
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="scored_{name}.csv"'},
    )
