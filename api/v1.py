"""
api/v1.py
---------
Public, versioned developer API: upload a labelled customer CSV, train a churn
model on it, then score customers with that model. Every route except
/v1/health and /v1/signup needs an API key, and accounts only ever see their
own datasets and models.

Endpoints:
    GET    /v1/health
    POST   /v1/signup                    → self-serve API key (if CHURN_OPEN_SIGNUP=true)
    GET    /v1/account                   → your tier, limits, and usage
    POST   /v1/datasets                  → upload a training CSV, get its inferred schema
    GET    /v1/datasets/{id}
    DELETE /v1/datasets/{id}
    POST   /v1/models                    → train a model (async: poll until ready)
    GET    /v1/models
    GET    /v1/models/{id}               → status + full training report
    DELETE /v1/models/{id}
    POST   /v1/models/{id}/predict       → score up to 1,000 customers as JSON
    POST   /v1/models/{id}/predict/csv   → upload a CSV, download it scored
"""

import io
import logging
import os
import re
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from api.auth import STORAGE_DIR, check_signup_rate, get_accounts, require_account
from api.errors import ApiError
from api.schemas import (
    Account, Dataset, ErrorResponse, Model, ModelList, ModelSummary, PredictRequest,
    PredictResponse, SignupRequest, SignupResponse, TrainRequest,
)
from src.accounts import AccountStore
from src.engine import MIN_INPUT_COVERAGE, TrainingError, prepare_labels, train_pipeline
from src.registry import NotFound, Registry
from src.schema import infer_schema, target_values

logger = logging.getLogger(__name__)

OPEN_SIGNUP = os.environ.get("CHURN_OPEN_SIGNUP", "false").lower() in ("1", "true", "yes")
MAX_CONCURRENT_TRAINING = int(os.environ.get("CHURN_MAX_CONCURRENT_TRAINING", "1"))

_registry: Registry | None = None
_training_slots = threading.Semaphore(MAX_CONCURRENT_TRAINING)


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry(STORAGE_DIR)
    return _registry


ERRORS = {code: {"model": ErrorResponse} for code in (400, 401, 404, 409, 413, 422, 429)}

public = APIRouter(prefix="/v1", tags=["Account"])
router = APIRouter(prefix="/v1", dependencies=[Depends(require_account)], responses=ERRORS)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_csv_upload(file: UploadFile, max_mb: float) -> pd.DataFrame:
    """Read an uploaded CSV with a size cap; turn parse failures into 400s."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise ApiError(400, "invalid_file_type", "Please upload a .csv file.")

    limit = int(max_mb * 1024 * 1024)
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise ApiError(413, "file_too_large", f"File is larger than your {max_mb:g} MB upload limit.")

    try:
        df = pd.read_csv(io.BytesIO(data), low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise ApiError(400, "invalid_csv", f"Couldn't read that CSV: {e}")
    if df.empty or df.shape[1] < 2:
        raise ApiError(400, "invalid_csv", "The CSV needs a header row, at least 2 columns, and at least 1 data row.")
    return df


def _dataset_or_404(registry: Registry, dataset_id: str, owner: str) -> dict:
    try:
        return registry.get_dataset(dataset_id, owner)
    except NotFound:
        raise ApiError(404, "dataset_not_found", f"No dataset with id {dataset_id!r}.")


def _model_or_404(registry: Registry, model_id: str, owner: str) -> dict:
    try:
        return registry.get_model(model_id, owner)
    except NotFound:
        raise ApiError(404, "model_not_found", f"No model with id {model_id!r}.")


def _ready_pipeline(registry: Registry, model_id: str, owner: str):
    meta = _model_or_404(registry, model_id, owner)
    if meta["status"] != "ready":
        raise ApiError(409, "model_not_ready", f"Model is {meta['status']}, not ready to score.",
                       details={"status": meta["status"]})
    return registry.load_pipeline(model_id)


def _score(pipeline, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    coverage, missing = pipeline.coverage(df)
    if coverage < MIN_INPUT_COVERAGE:
        raise ApiError(
            422, "insufficient_columns",
            f"This data has only {coverage:.0%} of the columns the model was trained on.",
            details={"missing_columns": missing},
        )
    return pipeline.score(df), missing


def _trainings_today(registry: Registry, owner: str) -> int:
    since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(timespec="seconds")
    return sum(m["created_at"] >= since for m in registry.list_models(owner))


def _account_view(account: dict, registry: Registry) -> dict:
    return {
        **account,
        "usage": {
            "models":          len(registry.list_models(account["id"])),
            "trainings_today": _trainings_today(registry, account["id"]),
        },
    }


# ── Public ─────────────────────────────────────────────────────────────────────

@public.get("/health")
def health():
    """Liveness check. No API key needed."""
    return {"status": "ok", "version": "v1", "signup": "open" if OPEN_SIGNUP else "invite-only"}


@public.post("/signup", status_code=201, response_model=SignupResponse, responses=ERRORS)
def signup(
    body: SignupRequest,
    request: Request,
    accounts: AccountStore = Depends(get_accounts),
    registry: Registry = Depends(get_registry),
):
    """
    Create a free-tier account and API key. Only available when the server
    runs with CHURN_OPEN_SIGNUP=true; otherwise ask the operator for a key.
    The key is shown once — store it.
    """
    if not OPEN_SIGNUP:
        raise ApiError(403, "signup_disabled", "Self-serve signup is turned off. Ask the operator for an API key.")
    check_signup_rate(request)
    account, key = accounts.create_account(body.name, body.email, tier="free")
    return {"api_key": key, "account": _account_view(account, registry)}


@router.get("/account", response_model=Account, tags=["Account"])
def get_account(account: dict = Depends(require_account), registry: Registry = Depends(get_registry)):
    """Your tier, limits, and current usage."""
    return _account_view(account, registry)


# ── Datasets ───────────────────────────────────────────────────────────────────

@router.post("/datasets", status_code=201, response_model=Dataset, tags=["Datasets"])
def upload_dataset(
    file: UploadFile = File(..., description="CSV, one row per customer, including a churn column"),
    account: dict = Depends(require_account),
    registry: Registry = Depends(get_registry),
):
    """
    Upload a labelled customer CSV. Returns how each column was interpreted and
    a guess at the churn and ID columns — check these before training.
    """
    df = _read_csv_upload(file, account["limits"]["max_upload_mb"])
    schema = infer_schema(df)
    meta = registry.save_dataset(df, file.filename, schema, owner=account["id"])
    if schema["suggested_target"]:
        meta["target_values"] = target_values(df[schema["suggested_target"]])
    return meta


@router.get("/datasets/{dataset_id}", response_model=Dataset, tags=["Datasets"])
def get_dataset(dataset_id: str, account: dict = Depends(require_account),
                registry: Registry = Depends(get_registry)):
    return _dataset_or_404(registry, dataset_id, account["id"])


@router.delete("/datasets/{dataset_id}", status_code=204, response_class=Response, tags=["Datasets"])
def delete_dataset(dataset_id: str, account: dict = Depends(require_account),
                   registry: Registry = Depends(get_registry)):
    _dataset_or_404(registry, dataset_id, account["id"])
    registry.delete_dataset(dataset_id, account["id"])


# ── Training ───────────────────────────────────────────────────────────────────

def run_training(registry: Registry, model_id: str, request: TrainRequest, df: pd.DataFrame, owner: str) -> None:
    """
    Background job: train, save the pipeline, record the report. The uploaded
    data is deleted after a successful run (unless keep_dataset) and kept after
    a failure, so the caller can retry with different settings.
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
            registry.delete_dataset(request.dataset_id, owner)
        except NotFound:
            pass


@router.post("/models", status_code=202, response_model=Model, tags=["Models"])
def train_model(
    request: TrainRequest,
    background: BackgroundTasks,
    account: dict = Depends(require_account),
    registry: Registry = Depends(get_registry),
):
    """
    Start training a churn model on one of your datasets. Returns immediately
    with status `queued`; poll `GET /v1/models/{id}` until it is `ready` or
    `failed` (training usually takes seconds).
    """
    owner, limits = account["id"], account["limits"]
    dataset = _dataset_or_404(registry, request.dataset_id, owner)

    if len(registry.list_models(owner)) >= limits["max_models"]:
        raise ApiError(409, "model_limit_reached",
                       f"Your tier allows {limits['max_models']} models. Delete one to train another.")
    if _trainings_today(registry, owner) >= limits["trainings_per_day"]:
        raise ApiError(429, "training_quota_exceeded",
                       f"Your tier allows {limits['trainings_per_day']} trainings per 24 hours.")

    df = registry.load_dataset(request.dataset_id, owner)
    try:   # fail fast on problems we can spot without training
        _, _, positive_label = prepare_labels(df, request.target, request.positive_label)
    except TrainingError as e:
        raise ApiError(422, "invalid_target", str(e))
    if request.id_column and request.id_column not in df.columns:
        raise ApiError(422, "invalid_id_column", f"ID column {request.id_column!r} is not in the dataset.")

    meta = registry.create_model(
        owner=owner,
        name=request.name or dataset["filename"],
        dataset_filename=dataset["filename"],
        target=request.target,
        positive_label=positive_label,
    )
    background.add_task(run_training, registry, meta["id"], request, df, owner)
    return meta


# ── Models ─────────────────────────────────────────────────────────────────────

@router.get("/models", response_model=ModelList, tags=["Models"])
def list_models(account: dict = Depends(require_account), registry: Registry = Depends(get_registry)):
    """Your models, newest first (summary only — fetch one for its full report)."""
    data = []
    for meta in registry.list_models(account["id"]):
        report = meta.get("report") or {}
        data.append(ModelSummary(
            id=meta["id"], name=meta.get("name"), status=meta["status"], created_at=meta["created_at"],
            auc_roc=report.get("metrics", {}).get("auc_roc"), rows=report.get("rows"),
        ))
    return {"data": data}


@router.get("/models/{model_id}", response_model=Model, tags=["Models"])
def get_model(model_id: str, account: dict = Depends(require_account),
              registry: Registry = Depends(get_registry)):
    """Status, and once ready, the full training report and input schema."""
    return _model_or_404(registry, model_id, account["id"])


@router.delete("/models/{model_id}", status_code=204, response_class=Response, tags=["Models"])
def delete_model(model_id: str, account: dict = Depends(require_account),
                 registry: Registry = Depends(get_registry)):
    _model_or_404(registry, model_id, account["id"])
    registry.delete_model(model_id, account["id"])


@router.post("/models/{model_id}/predict", response_model=PredictResponse, tags=["Scoring"])
def predict(model_id: str, request: PredictRequest, account: dict = Depends(require_account),
            registry: Registry = Depends(get_registry)):
    """
    Score up to 1,000 customers in one vectorized call. Send the columns listed
    in the model's `report.input_schema`; missing ones are treated as unknown
    and reported back in `missing_columns`.
    """
    pipeline = _ready_pipeline(registry, model_id, account["id"])
    scored, missing = _score(pipeline, pd.DataFrame(request.customers))
    if pipeline.id_col and pipeline.id_col in scored.columns:
        scored = scored.rename(columns={pipeline.id_col: "id"})
    scored = scored.astype(object).where(scored.notna(), None)
    return {"count": len(scored), "missing_columns": missing, "results": scored.to_dict(orient="records")}


@router.post(
    "/models/{model_id}/predict/csv",
    tags=["Scoring"],
    response_class=StreamingResponse,
    responses={200: {"content": {"text/csv": {}}, "description": "The uploaded CSV with score columns appended"}},
)
def predict_csv(
    model_id: str,
    file: UploadFile = File(..., description="CSV of customers to score (churn column not needed)"),
    account: dict = Depends(require_account),
    registry: Registry = Depends(get_registry),
):
    """
    Upload a CSV of customers; download the same CSV with churn_probability,
    churn_prediction, risk_tier, and recommended_action appended.
    """
    pipeline = _ready_pipeline(registry, model_id, account["id"])
    df = _read_csv_upload(file, account["limits"]["max_upload_mb"])
    scored, _ = _score(pipeline, df)
    out = pd.concat([df, scored.drop(columns=[pipeline.id_col], errors="ignore")], axis=1)

    buffer = io.StringIO()
    out.to_csv(buffer, index=False)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(file.filename))[0])
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="scored_{name}.csv"'},
    )
