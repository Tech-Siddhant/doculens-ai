import contextvars
import json
import logging
import re
import uuid
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Context variable to hold the request ID for the current async task
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

SENSITIVE_FIELD_NAMES = {
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "secret_str",
    "gemini_api_key",
    "openai_api_key",
    "qdrant_api_key",
    "llm_api_key",
}

# Common API key / token patterns (OpenAI, Google API key, Bearer tokens, hex secrets)
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"AIza[a-zA-Z0-9_\-]{30,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{15,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?([^'\"\s,]+)['\"]?", re.IGNORECASE),
]


def redact_text(text: str) -> str:
    """Scrub known secret formats from log strings."""
    if not isinstance(text, str):
        return str(text)
    
    redacted = text
    # Mask regex matches
    for pattern in SECRET_PATTERNS[:3]:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    
    # Mask key-value patterns (e.g. api_key="secret123")
    def _sub_kv(match: re.Match[str]) -> str:
        k = match.group(1)
        return f"{k}=[REDACTED]"
    
    redacted = SECRET_PATTERNS[3].sub(_sub_kv, redacted)
    return redacted


def sanitize_structured_data(data: Any) -> Any:
    """Recursively redact sensitive keys and values in log data."""
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if str(k).lower() in SENSITIVE_FIELD_NAMES:
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_structured_data(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_structured_data(x) for x in data]
    elif isinstance(data, str):
        return redact_text(data)
    return data


class JSONFormatter(logging.Formatter):
    """
    ponytail: Custom JSON formatter without external dependencies.
    Extracts explicit structured fields if provided in 'extra' and redacts secrets.
    """
    def format(self, record: logging.LogRecord) -> str:
        # Standard record message formatting with secret redaction
        raw_message = record.getMessage()
        clean_message = redact_text(raw_message)

        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": clean_message,
            "logger": record.name,
        }
        
        req_id = request_id_var.get()
        if req_id:
            log_obj["request_id"] = req_id
            
        # Extract common structured logging fields passed via `extra=...`
        for key in (
            "request_id",
            "operation",
            "document_id",
            "query_id",
            "stage",
            "duration_ms",
            "status",
            "error_type",
        ):
            if hasattr(record, key):
                val = getattr(record, key)
                if val is not None:
                    log_obj[key] = sanitize_structured_data(val)

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            log_obj["exception"] = redact_text(record.exc_text)

        return json.dumps(log_obj)


def setup_logging(level: str | int = logging.INFO) -> None:
    """Configure the root logger with the JSON formatter."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    root = logging.getLogger()
    root.setLevel(level)
    
    # Remove existing handlers to avoid duplicates (e.g. from basicConfig or uvicorn)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to ensure every request has an X-Request-ID, propagate it via contextvars,
    and record request metrics.
    """
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(req_id)
        
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            from app.core.metrics import metrics_collector
            metrics_collector.record_request(request.method, request.url.path, response.status_code)
            return response
        except Exception:
            from app.core.metrics import metrics_collector
            metrics_collector.record_request(request.method, request.url.path, 500)
            raise
        finally:
            request_id_var.reset(token)


