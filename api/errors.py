"""
api/errors.py
-------------
One error shape for the whole /v1 API:

    {"error": {"code": "model_not_found", "message": "...", "details": ...}}

`code` is stable and safe to branch on; `message` is for humans and may change.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

V1_PREFIX = "/v1"

# Default codes for errors raised by the framework rather than by us
STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "unavailable",
}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None, headers: dict | None = None):
        self.status, self.code, self.message = status, code, message
        self.details, self.headers = details, headers


def _body(code: str, message: str, details=None) -> dict:
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return {"error": error}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return JSONResponse(_body(exc.code, exc.message, exc.details), exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if not request.url.path.startswith(V1_PREFIX):
            return JSONResponse({"detail": exc.errors()}, 422)   # demo endpoints keep FastAPI's shape
        details = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(_body("validation_error", "The request body or parameters are invalid.", details), 422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        if not request.url.path.startswith(V1_PREFIX):
            return JSONResponse({"detail": exc.detail}, exc.status_code, headers=exc.headers)
        code = STATUS_CODES.get(exc.status_code, "error")
        return JSONResponse(_body(code, str(exc.detail)), exc.status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        logger.exception(f"Unhandled error on {request.method} {request.url.path}")
        return JSONResponse(_body("internal_error", "Something went wrong on our side."), 500)
