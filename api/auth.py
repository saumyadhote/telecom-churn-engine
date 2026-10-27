"""
api/auth.py
-----------
API-key authentication and per-account rate limiting for /v1.

Send the key as either header:
    Authorization: Bearer ck_...
    X-API-Key: ck_...

Every authenticated response carries X-RateLimit-Limit / X-RateLimit-Remaining;
a 429 also carries Retry-After (seconds).
"""

import math
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, Request

from api.errors import ApiError
from src.accounts import AccountStore

STORAGE_DIR = Path(os.environ.get("CHURN_STORAGE_DIR", "storage/"))

_accounts: AccountStore | None = None


def get_accounts() -> AccountStore:
    global _accounts
    if _accounts is None:
        _accounts = AccountStore(STORAGE_DIR / "accounts.db")
    return _accounts


class SlidingWindowLimiter:
    """
    In-memory sliding-window limiter. Per process — fine for one server; move
    to Redis if you run several replicas.
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int) -> tuple[bool, int, int]:
        """Record a request. Returns (allowed, remaining, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= limit:
                return False, 0, max(1, math.ceil(hits[0] + self.window - now))
            hits.append(now)
            return True, limit - len(hits), 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


request_limiter = SlidingWindowLimiter(60)
signup_limiter = SlidingWindowLimiter(3600)
SIGNUPS_PER_IP_PER_HOUR = 5


def _extract_key(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def require_account(request: Request, accounts: AccountStore = Depends(get_accounts)) -> dict:
    """FastAPI dependency: the calling account, or a 401/429."""
    key = _extract_key(request)
    if not key:
        raise ApiError(401, "missing_api_key",
                       "Send your API key as 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
                       headers={"WWW-Authenticate": "Bearer"})
    account = accounts.authenticate(key)
    if account is None:
        raise ApiError(401, "invalid_api_key", "This API key is invalid or has been revoked.",
                       headers={"WWW-Authenticate": "Bearer"})

    limit = account["limits"]["requests_per_minute"]
    allowed, remaining, retry_after = request_limiter.hit(account["id"], limit)
    headers = {"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": str(remaining)}
    if not allowed:
        raise ApiError(429, "rate_limited",
                       f"Rate limit of {limit} requests per minute exceeded. Retry in {retry_after}s.",
                       headers={**headers, "Retry-After": str(retry_after)})
    request.state.rate_limit_headers = headers     # added to the response by middleware
    return account


async def rate_limit_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers.update(getattr(request.state, "rate_limit_headers", {}))
    return response


def check_signup_rate(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    allowed, _, retry_after = signup_limiter.hit(ip, SIGNUPS_PER_IP_PER_HOUR)
    if not allowed:
        raise ApiError(429, "rate_limited", "Too many signups from this address. Try again later.",
                       headers={"Retry-After": str(retry_after)})
