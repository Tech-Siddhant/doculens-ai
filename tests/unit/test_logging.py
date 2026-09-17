import json
import logging
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.logger import (
    JSONFormatter,
    RequestIDMiddleware,
    redact_text,
    request_id_var,
    sanitize_structured_data,
    setup_logging,
)


def _make_record(
    logger_name: str = "test",
    level: int = logging.INFO,
    msg: str = "test message",
    extra: dict | None = None,
) -> logging.LogRecord:
    """Helper: build a LogRecord with optional extra fields."""
    lg = logging.getLogger(logger_name)
    return lg.makeRecord(
        name=logger_name, level=level, fn="test_file.py", lno=1,
        msg=msg, args=(), exc_info=None, extra=extra or {},
    )


# ── Structured log fields ──────────────────────────────────────────────


def test_json_formatter_standard_and_structured_fields() -> None:
    formatter = JSONFormatter()
    token = request_id_var.set("req_test_123")
    try:
        record = _make_record(
            msg="Processing test stage",
            extra={
                "document_id": "doc_abc123",
                "stage": "retrieval",
                "duration_ms": 45.67,
                "status": "success",
                "error_type": None,
            },
        )
        log_json = json.loads(formatter.format(record))
        assert log_json["level"] == "INFO"
        assert log_json["message"] == "Processing test stage"
        assert log_json["request_id"] == "req_test_123"
        assert log_json["document_id"] == "doc_abc123"
        assert log_json["stage"] == "retrieval"
        assert log_json["duration_ms"] == 45.67
        assert log_json["status"] == "success"
        assert "error_type" not in log_json  # None values omitted
    finally:
        request_id_var.reset(token)


def test_operation_and_query_id_fields() -> None:
    formatter = JSONFormatter()
    record = _make_record(
        msg="pipeline running",
        extra={"operation": "pipeline", "query_id": "q_abc123def456",
               "stage": "hybrid_retrieval", "document_id": "doc_42",
               "duration_ms": 12.5, "status": "success"},
    )
    log_json = json.loads(formatter.format(record))
    assert log_json["operation"] == "pipeline"
    assert log_json["query_id"] == "q_abc123def456"
    assert log_json["stage"] == "hybrid_retrieval"
    assert log_json["document_id"] == "doc_42"


def test_all_phase91_fields_present_when_set() -> None:
    formatter = JSONFormatter()
    token = request_id_var.set("req_full")
    try:
        record = _make_record(
            msg="full trace",
            extra={"operation": "generation", "document_id": "doc_99",
                   "query_id": "q_full", "stage": "answer_generation",
                   "duration_ms": 320.1, "status": "success", "error_type": None},
        )
        log_json = json.loads(formatter.format(record))
        assert log_json["request_id"] == "req_full"
        assert log_json["operation"] == "generation"
        assert log_json["document_id"] == "doc_99"
        assert log_json["query_id"] == "q_full"
        assert log_json["stage"] == "answer_generation"
        assert log_json["duration_ms"] == 320.1
        assert log_json["status"] == "success"
        assert "error_type" not in log_json
    finally:
        request_id_var.reset(token)


def test_secret_redaction_in_messages_and_extra() -> None:
    formatter = JSONFormatter()
    msg = "Call failed with key sk-abcdef1234567890abcdef12 and AIzaSyD98765432101234567890123456789 and Bearer tok_1234567890abcdef"
    record = _make_record(
        logger_name="test_redaction", level=logging.ERROR, msg=msg,
        extra={"document_id": "doc_999"},
    )
    log_json = json.loads(formatter.format(record))
    assert "sk-" not in log_json["message"]
    assert "AIza" not in log_json["message"]
    assert "[REDACTED_SECRET]" in log_json["message"]
    assert log_json["document_id"] == "doc_999"


def test_sanitize_structured_data_recursive() -> None:
    data = {
        "user": "alice",
        "api_key": "sk-12345678901234567890",
        "nested": {"token": "secret_token_val", "safe_field": "ok"},
        "list_items": [{"secret": "pass123"}, "normal_text"],
    }
    sanitized = sanitize_structured_data(data)
    assert sanitized["user"] == "alice"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["token"] == "[REDACTED]"
    assert sanitized["nested"]["safe_field"] == "ok"
    assert sanitized["list_items"][0]["secret"] == "[REDACTED]"
    assert sanitized["list_items"][1] == "normal_text"


def test_sensitive_keys_never_leak() -> None:
    """All known-sensitive field names are redacted recursively."""
    data = {
        "gemini_api_key": "AIzaSyABCDEF123456",
        "openai_api_key": "sk-proj-abc123",
        "authorization": "Bearer xyz",
        "password": "hunter2",
        "qdrant_api_key": "qdr_secret",
    }
    sanitized = sanitize_structured_data(data)
    for key in data:
        assert sanitized[key] == "[REDACTED]"


def test_request_id_middleware_generates_and_propagates_header() -> None:
    app = create_app()
    client = TestClient(app)
    res1 = client.get("/api/v1/health")
    assert res1.status_code == 200
    assert "X-Request-ID" in res1.headers
    assert len(res1.headers["X-Request-ID"]) > 0
    res2 = client.get("/api/v1/health", headers={"X-Request-ID": "custom-trace-id-999"})
    assert res2.status_code == 200
    assert res2.headers.get("X-Request-ID") == "custom-trace-id-999"


def test_request_id_unique_per_request() -> None:
    app = create_app()
    client = TestClient(app)
    r1 = client.get("/api/v1/health")
    r2 = client.get("/api/v1/health")
    assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]


def test_orchestrator_logs_pipeline_stages(caplog) -> None:
    """Orchestrator emits structured log lines for each pipeline stage."""
    from app.services.orchestrator import orchestrator
    orch_logger = logging.getLogger("app.services.orchestrator")
    orch_logger.propagate = True
    with caplog.at_level(logging.DEBUG, logger="app.services.orchestrator"):
        try:
            orchestrator.orchestrate_query(question="What is DocuLens?", document_id=None, top_k=3)
        except Exception:
            pass
    stage_logs = [r for r in caplog.records if hasattr(r, "stage")]
    assert len(stage_logs) >= 1
    first = stage_logs[0]
    assert first.stage == "question_understanding"
    assert first.operation == "pipeline"
    assert first.query_id.startswith("q_")


def test_error_log_structured_fields() -> None:
    formatter = JSONFormatter()
    record = _make_record(
        level=logging.ERROR, msg="Something broke",
        extra={"operation": "indexing", "document_id": "doc_err",
               "stage": "indexing", "status": "error", "error_type": "ValueError"},
    )
    log_json = json.loads(formatter.format(record))
    assert log_json["level"] == "ERROR"
    assert log_json["error_type"] == "ValueError"
    assert log_json["operation"] == "indexing"
    assert log_json["status"] == "error"


def test_exception_log_includes_traceback() -> None:
    import sys
    formatter = JSONFormatter()
    try:
        raise RuntimeError("test boom")
    except RuntimeError:
        exc_info = sys.exc_info()
    lg = logging.getLogger("test_exc")
    record = lg.makeRecord(
        name="test_exc", level=logging.ERROR, fn="x.py", lno=1,
        msg="Crash", args=(), exc_info=exc_info,
    )
    log_json = json.loads(formatter.format(record))
    assert "exception" in log_json
    assert "RuntimeError" in log_json["exception"]
    assert "test boom" in log_json["exception"]


def test_setup_logging_configures_root_handler() -> None:
    setup_logging(logging.DEBUG)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) > 0
    assert isinstance(root.handlers[0].formatter, JSONFormatter)
