"""Exceptions raised by the Churn Engine client. Branch on `.code`, not `.message`."""

from __future__ import annotations


class ChurnEngineError(Exception):
    """Any error returned by the API."""

    def __init__(self, status: int, code: str, message: str, details=None):
        super().__init__(f"[{status} {code}] {message}")
        self.status, self.code, self.message, self.details = status, code, message, details


class AuthenticationError(ChurnEngineError):
    """401 — missing, invalid, or revoked API key."""


class NotFoundError(ChurnEngineError):
    """404 — the dataset or model doesn't exist (or isn't yours)."""


class ValidationError(ChurnEngineError):
    """422 — the request or data was rejected; see `.details`."""


class RateLimitError(ChurnEngineError):
    """429 — slow down; `.retry_after` says how many seconds to wait."""

    def __init__(self, *args, retry_after: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_after = retry_after


class TrainingFailedError(ChurnEngineError):
    """Training finished with status 'failed'; `.message` says why."""


class TrainingTimeoutError(ChurnEngineError):
    """`train(wait=True)` gave up waiting; the model may still finish."""


_BY_STATUS = {401: AuthenticationError, 404: NotFoundError, 422: ValidationError, 429: RateLimitError}


def from_response(response) -> ChurnEngineError:
    try:
        error = response.json()["error"]
        code, message, details = error["code"], error["message"], error.get("details")
    except Exception:
        code, message, details = "http_error", response.text[:500] or response.reason_phrase, None

    cls = _BY_STATUS.get(response.status_code, ChurnEngineError)
    if cls is RateLimitError:
        retry = response.headers.get("retry-after")
        return RateLimitError(response.status_code, code, message, details,
                              retry_after=float(retry) if retry else None)
    return cls(response.status_code, code, message, details)
