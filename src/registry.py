"""
registry.py
-----------
File-based storage for uploaded datasets and trained models.

    <root>/datasets/<id>/data.csv + meta.json
    <root>/models/<id>/pipeline.joblib + meta.json

Small and dependency-free on purpose; swapping it for S3 + a database later
only means re-implementing this class.

Usage:
    from src.registry import Registry
    registry = Registry("storage/")
"""

import json
import os
import re
import shutil
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

ID_RE = re.compile(r"^[0-9a-f]{12}$")
CACHE_SIZE = 8     # trained pipelines kept in memory


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class NotFound(KeyError):
    pass


class Registry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "datasets").mkdir(parents=True, exist_ok=True)
        (self.root / "models").mkdir(parents=True, exist_ok=True)
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()

    # ── Helpers ──

    def _dir(self, kind: str, item_id: str) -> Path:
        # IDs come from URLs — only accept our own format (blocks path traversal)
        if not ID_RE.match(item_id or ""):
            raise NotFound(item_id)
        path = self.root / kind / item_id
        if not (path / "meta.json").exists():
            raise NotFound(item_id)
        return path

    @staticmethod
    def _write_meta(path: Path, meta: dict) -> None:
        tmp = path / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2, default=str))
        os.replace(tmp, path / "meta.json")

    @staticmethod
    def _read_meta(path: Path) -> dict:
        return json.loads((path / "meta.json").read_text())

    # ── Datasets ──

    def save_dataset(self, df: pd.DataFrame, filename: str, schema: dict) -> dict:
        dataset_id = _new_id()
        path = self.root / "datasets" / dataset_id
        path.mkdir()
        df.to_csv(path / "data.csv", index=False)
        meta = {
            "id":         dataset_id,
            "filename":   filename,
            "rows":       int(len(df)),
            "columns":    int(df.shape[1]),
            "created_at": _now(),
            **schema,
        }
        self._write_meta(path, meta)
        return meta

    def get_dataset(self, dataset_id: str) -> dict:
        return self._read_meta(self._dir("datasets", dataset_id))

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        path = self._dir("datasets", dataset_id) / "data.csv"
        return pd.read_csv(path, keep_default_na=True, low_memory=False)

    def delete_dataset(self, dataset_id: str) -> None:
        shutil.rmtree(self._dir("datasets", dataset_id))

    # ── Models ──

    def create_model(self, **fields) -> dict:
        model_id = _new_id()
        path = self.root / "models" / model_id
        path.mkdir()
        meta = {"id": model_id, "status": "queued", "created_at": _now(), **fields}
        self._write_meta(path, meta)
        return meta

    def update_model(self, model_id: str, **fields) -> dict:
        path = self._dir("models", model_id)
        with self._lock:
            meta = {**self._read_meta(path), **fields, "updated_at": _now()}
            self._write_meta(path, meta)
        return meta

    def get_model(self, model_id: str) -> dict:
        return self._read_meta(self._dir("models", model_id))

    def list_models(self) -> list[dict]:
        metas = []
        for path in (self.root / "models").iterdir():
            if (path / "meta.json").exists():
                metas.append(self._read_meta(path))
        return sorted(metas, key=lambda m: m["created_at"], reverse=True)

    def save_pipeline(self, model_id: str, pipeline) -> None:
        joblib.dump(pipeline, self._dir("models", model_id) / "pipeline.joblib")

    def load_pipeline(self, model_id: str):
        with self._lock:
            if model_id in self._cache:
                self._cache.move_to_end(model_id)
                return self._cache[model_id]
        pipeline = joblib.load(self._dir("models", model_id) / "pipeline.joblib")
        with self._lock:
            self._cache[model_id] = pipeline
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return pipeline

    def delete_model(self, model_id: str) -> None:
        path = self._dir("models", model_id)
        with self._lock:
            self._cache.pop(model_id, None)
        shutil.rmtree(path)

    def recover_interrupted(self) -> int:
        """Mark models left 'queued'/'training' by a server restart as failed."""
        count = 0
        for meta in self.list_models():
            if meta["status"] in ("queued", "training"):
                self.update_model(meta["id"], status="failed",
                                  error="Training was interrupted by a server restart — please retrain.")
                count += 1
        return count
