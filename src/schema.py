"""
schema.py
---------
Infer what each column of an arbitrary customer CSV is, so the engine can
train on data it has never seen before.

Every column gets one role:
    numeric      — used as a numeric feature
    categorical  — used as a categorical feature
    id           — customer identifier, carried through but not used to train
    ignored      — not usable (constant, mostly empty, free text, ...)

Usage:
    from src.schema import infer_schema, encode_target
"""

import re
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Heuristics ────────────────────────────────────────────────────────────────

TARGET_HINTS = (
    "churn", "churned", "exited", "attrition", "attrited", "left",
    "cancelled", "canceled", "is_churn", "target", "label",
)

ID_PATTERN = re.compile(r"(^|_)(id|uuid|key)$|^(customer|client|account|user|subscriber)(_?(id|no|number|code))?$", re.I)

# Values that mean "this customer churned" when the label is text/boolean
POSITIVE_LABELS = {
    "1", "1.0", "true", "yes", "y", "t", "churn", "churned", "exited",
    "left", "cancelled", "canceled", "attrited", "attrited customer",
}

NUMERIC_PARSE_RATIO = 0.95   # text column becomes numeric if ≥95% of values parse
MAX_MISSING_RATIO   = 0.95   # columns emptier than this are ignored
MAX_CATEGORIES      = 200    # more distinct text values than this → free text


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def coerce_numeric(series: pd.Series) -> pd.Series | None:
    """
    Return the series as numbers if it is numeric (or text that is almost
    entirely numbers, e.g. " " blanks in a charges column); otherwise None.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    non_null = series.dropna()
    if non_null.empty:
        return None
    parsed = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
    if parsed.notna().mean() >= NUMERIC_PARSE_RATIO:
        return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
    return None


def _column_role(series: pd.Series, name: str, n_rows: int) -> tuple[str, str]:
    """Decide the role of one (non-target) column. Returns (role, reason)."""
    missing = series.isna().mean()
    n_unique = series.nunique(dropna=True)

    if missing >= MAX_MISSING_RATIO:
        return "ignored", f"{missing:.0%} of values are empty"
    if n_unique <= 1:
        return "ignored", "same value in every row"

    looks_like_id = bool(ID_PATTERN.search(_normalise(name)))
    if looks_like_id and n_unique >= 0.9 * series.notna().sum():
        return "id", "unique per row and named like an identifier"

    numeric = coerce_numeric(series)
    if numeric is not None:
        return "numeric", "numbers"

    if n_unique > MAX_CATEGORIES and n_unique > 0.05 * n_rows:
        return "ignored", f"{n_unique:,} distinct text values — looks like free text or an identifier"
    return "categorical", f"{n_unique} categories"


def suggest_target(df: pd.DataFrame) -> str | None:
    """Pick the column most likely to be the churn label, if any."""
    binary = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    for hint in TARGET_HINTS:
        for col in binary:
            if hint in _normalise(col).split("_") or _normalise(col) == hint:
                return col
    return None


def infer_schema(df: pd.DataFrame, target: str | None = None) -> dict:
    """
    Describe every column of an uploaded dataframe.

    Parameters
    ----------
    df     : raw uploaded dataframe
    target : the label column, if already known (otherwise it is guessed)

    Returns
    -------
    dict with keys:
        columns          : list of per-column dicts (name, role, reason, ...)
        suggested_target : best guess at the churn label column, or None
        suggested_id     : best guess at the customer ID column, or None
    """
    target = target or suggest_target(df)
    n_rows = len(df)

    columns = []
    for col in df.columns:
        series = df[col]
        if col == target:
            role, reason = "target", "churn label"
        else:
            role, reason = _column_role(series, col, n_rows)

        samples = series.dropna().astype(str).unique()[:5].tolist()
        columns.append({
            "name":        str(col),
            "role":        role,
            "reason":      reason,
            "dtype":       str(series.dtype),
            "missing_pct": round(float(series.isna().mean()) * 100, 1),
            "n_unique":    int(series.nunique(dropna=True)),
            "samples":     samples,
        })

    ids = [c["name"] for c in columns if c["role"] == "id"]
    return {
        "columns":          columns,
        "suggested_target": target,
        "suggested_id":     ids[0] if ids else None,
    }


# ── Target ────────────────────────────────────────────────────────────────────

def target_values(series: pd.Series) -> list[str]:
    """The distinct label values, as strings, in a stable order."""
    return sorted(series.dropna().astype(str).str.strip().unique().tolist())


def encode_target(series: pd.Series, positive_label: str | None = None) -> tuple[pd.Series, str]:
    """
    Turn a binary label column (0/1, yes/no, True/False, "Churned"/"Stayed", ...)
    into a 0/1 series where 1 means "churned".

    Rows with a missing label are returned as NaN so the caller can drop them.

    Raises
    ------
    ValueError if the column does not have exactly two distinct values, or the
    positive label cannot be determined.
    """
    values = target_values(series)
    if len(values) != 2:
        shown = ", ".join(values[:6]) + ("…" if len(values) > 6 else "")
        raise ValueError(
            f"The churn column must have exactly 2 distinct values; found {len(values)} ({shown})."
        )

    if positive_label is None:
        matches = [v for v in values if v.lower() in POSITIVE_LABELS]
        if len(matches) != 1:
            raise ValueError(
                f"Can't tell which of {values} means 'churned' — pass positive_label."
            )
        positive_label = matches[0]
    elif str(positive_label) not in values:
        raise ValueError(f"positive_label {positive_label!r} is not one of {values}.")

    text = series.astype(str).str.strip()
    y = (text == str(positive_label)).astype(float)
    y[series.isna()] = np.nan
    return y, str(positive_label)
