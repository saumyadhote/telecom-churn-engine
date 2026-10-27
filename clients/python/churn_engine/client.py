"""
The Churn Engine Python client.

    from churn_engine import ChurnEngine

    client = ChurnEngine(api_key="ck_...", base_url="https://your-host")
    model = client.train("customers.csv", target="churned")      # waits until ready
    print(model.auc, model.report["tiers"])
    scored = client.score(model.id, "current_customers.csv")     # → pandas DataFrame
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx

from churn_engine.errors import (
    ChurnEngineError, RateLimitError, TrainingFailedError, TrainingTimeoutError, from_response,
)

DEFAULT_BASE_URL = "http://localhost:8000"
MAX_BATCH = 1000


@dataclass
class Model:
    """A trained (or training) model. `report` is the full training report once ready."""
    id: str
    status: str
    name: str | None = None
    target: str | None = None
    error: str | None = None
    report: dict | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, data: dict) -> "Model":
        return cls(id=data["id"], status=data["status"], name=data.get("name"), target=data.get("target"),
                   error=data.get("error"), report=data.get("report"), raw=data)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def auc(self) -> float | None:
        """AUC-ROC on the held-out test set."""
        return self.report["metrics"]["auc_roc"] if self.report else None

    @property
    def input_columns(self) -> list[str]:
        """Columns to send when scoring."""
        return [c["name"] for c in self.report["input_schema"]] if self.report else []


def _file_payload(data, default_name: str) -> tuple[str, bytes]:
    """Accept a path, a pandas DataFrame, raw bytes, or a file object."""
    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        return path.name, path.read_bytes()
    if hasattr(data, "to_csv"):            # pandas DataFrame
        return default_name, data.to_csv(index=False).encode()
    if isinstance(data, bytes):
        return default_name, data
    if hasattr(data, "read"):
        name = Path(getattr(data, "name", default_name)).name
        content = data.read()
        return name, content.encode() if isinstance(content, str) else content
    raise TypeError("Expected a file path, pandas DataFrame, bytes, or file object.")


class ChurnEngine:
    """
    Client for the Churn Engine /v1 API.

    Parameters
    ----------
    api_key     : your key (defaults to the CHURN_ENGINE_API_KEY environment variable)
    base_url    : API root (defaults to CHURN_ENGINE_URL, then http://localhost:8000)
    timeout     : per-request timeout in seconds
    max_retries : how many times to retry after a 429, honouring Retry-After
    http_client : bring your own httpx.Client (e.g. for proxies, or testing)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        http_client: httpx.Client | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("CHURN_ENGINE_API_KEY")
        if not api_key:
            raise ValueError("Pass api_key= or set CHURN_ENGINE_API_KEY.")
        self.max_retries = max_retries
        self._http = http_client or httpx.Client(
            base_url=(base_url or os.environ.get("CHURN_ENGINE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            timeout=timeout,
        )
        self._http.headers["Authorization"] = f"Bearer {api_key}"
        self._http.headers["User-Agent"] = "churn-engine-python/0.1"

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Transport ──

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            response = self._http.request(method, f"/v1{path}", **kwargs)
            if response.status_code < 400:
                return response
            error = from_response(response)
            if isinstance(error, RateLimitError) and attempt < self.max_retries:
                time.sleep(error.retry_after or 2 ** attempt)   # payloads are bytes, so resending is safe
                continue
            raise error
        raise AssertionError("unreachable")

    # ── Account ──

    def account(self) -> dict:
        """Your tier, limits, and usage."""
        return self._request("GET", "/account").json()

    # ── Datasets ──

    def upload_dataset(self, data, filename: str = "dataset.csv") -> dict:
        """Upload a labelled CSV (path, DataFrame, bytes, or file). Returns the inferred schema."""
        name, content = _file_payload(data, filename)
        return self._request("POST", "/datasets", files={"file": (name, content, "text/csv")}).json()

    def get_dataset(self, dataset_id: str) -> dict:
        return self._request("GET", f"/datasets/{dataset_id}").json()

    def delete_dataset(self, dataset_id: str) -> None:
        self._request("DELETE", f"/datasets/{dataset_id}")

    # ── Models ──

    def train(
        self,
        data=None,
        target: str | None = None,
        *,
        dataset_id: str | None = None,
        id_column: str | None = None,
        positive_label: str | None = None,
        exclude_columns: Iterable[str] = (),
        tier_thresholds: dict | None = None,
        name: str | None = None,
        keep_dataset: bool = False,
        wait: bool = True,
        timeout: float = 600,
        poll_interval: float = 1.0,
    ) -> Model:
        """
        Train a model. Pass `data` (path / DataFrame / bytes) to upload and train
        in one call, or `dataset_id` for a dataset you already uploaded.

        `target` defaults to the column the API detects as the churn label.
        With wait=True (default) this blocks until training finishes and raises
        TrainingFailedError if it fails.
        """
        if (data is None) == (dataset_id is None):
            raise ValueError("Pass exactly one of data or dataset_id.")
        if data is not None:
            dataset = self.upload_dataset(data)
            dataset_id = dataset["id"]
            target = target or dataset.get("suggested_target")
        if not target:
            raise ValueError("Couldn't detect the churn column — pass target=.")

        body: dict[str, Any] = {
            "dataset_id": dataset_id, "target": target, "id_column": id_column,
            "positive_label": positive_label, "exclude_columns": list(exclude_columns),
            "name": name, "keep_dataset": keep_dataset,
        }
        if tier_thresholds:
            body["tier_thresholds"] = tier_thresholds
        model = Model.from_json(self._request("POST", "/models", json=body).json())
        return self.wait_for(model.id, timeout=timeout, poll_interval=poll_interval) if wait else model

    def wait_for(self, model_id: str, timeout: float = 600, poll_interval: float = 1.0) -> Model:
        """Poll until a model is ready (returns it) or failed (raises)."""
        deadline = time.monotonic() + timeout
        while True:
            model = self.get_model(model_id)
            if model.status == "ready":
                return model
            if model.status == "failed":
                raise TrainingFailedError(200, "training_failed", model.error or "Training failed.")
            if time.monotonic() > deadline:
                raise TrainingTimeoutError(0, "training_timeout",
                                           f"Model {model_id} still {model.status} after {timeout:g}s.")
            time.sleep(poll_interval)

    def get_model(self, model_id: str) -> Model:
        return Model.from_json(self._request("GET", f"/models/{model_id}").json())

    def list_models(self) -> list[dict]:
        return self._request("GET", "/models").json()["data"]

    def delete_model(self, model_id: str) -> None:
        self._request("DELETE", f"/models/{model_id}")

    # ── Scoring ──

    def predict(self, model_id: str, customers) -> list[dict]:
        """
        Score customers given as a list of dicts or a DataFrame. Batches larger
        than 1,000 are split automatically. Returns one result dict per customer.
        """
        if hasattr(customers, "to_dict"):
            customers = customers.astype(object).where(customers.notna(), None).to_dict(orient="records")
        customers = list(customers)
        results = []
        for start in range(0, len(customers), MAX_BATCH):
            chunk = customers[start:start + MAX_BATCH]
            results += self._request("POST", f"/models/{model_id}/predict", json={"customers": chunk}).json()["results"]
        return results

    def score(self, model_id: str, data, output: str | os.PathLike | None = None):
        """
        Score a whole CSV (path / DataFrame / bytes) server-side. Returns a pandas
        DataFrame with score columns appended; also writes it to `output` if given.
        """
        name, content = _file_payload(data, "customers.csv")
        response = self._request("POST", f"/models/{model_id}/predict/csv",
                                 files={"file": (name, content, "text/csv")})
        if output:
            Path(output).write_bytes(response.content)
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise ChurnEngineError(0, "pandas_required", "score() returns a DataFrame — pip install pandas") from e
        return pd.read_csv(io.BytesIO(response.content))
