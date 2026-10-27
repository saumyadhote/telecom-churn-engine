"""Python client for the Churn Engine API."""

from churn_engine.client import ChurnEngine, Model
from churn_engine.errors import (
    AuthenticationError, ChurnEngineError, NotFoundError, RateLimitError,
    TrainingFailedError, TrainingTimeoutError, ValidationError,
)

__version__ = "0.1.0"

__all__ = [
    "ChurnEngine", "Model", "ChurnEngineError", "AuthenticationError", "NotFoundError",
    "RateLimitError", "TrainingFailedError", "TrainingTimeoutError", "ValidationError",
]
