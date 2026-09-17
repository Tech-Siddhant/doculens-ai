# Phase 9.1 — Structured Logging & Request Tracing: Completion Report

**Date:** 2026-09-17
**Status:** ✅ Complete
**Tests:** 471 passed, 0 failed (12 logging-specific tests)

---

## Summary

Implemented structured JSON logging and request correlation tracing across the DocuLens AI pipeline using Python standard library (`logging`, `contextvars`). No external observability frameworks (OpenTelemetry, ELK, etc.) were added.

## Changes

### 1. `app/core/logger.py` — JSONFormatter field expansion
- **Added fields:** `request_id`, `operation`, `query_id` to the structured log extraction loop (joins existing `document_id`, `stage`, `duration_ms`, `status`, `error_type`).
- **`None` suppression:** Fields with `None` values are now omitted from JSON output instead of being serialized as `null`, keeping logs compact.

### 2. `app/api/main.py` — Exception handler enrichment
- `RequestValidationError` handler now logs `operation`, `error_type`, and `status` via `extra=`.
- `Exception` (unhandled) handler now logs `operation`, `error_type` (class name), and `status`.
- Uses `%s` formatting instead of f-strings for lazy log message evaluation.

### 3. `app/api/routes/documents.py` — Endpoint-level structured logging
- **Upload:** Logs `operation=upload`, `document_id`, `status`, `duration_ms` on success; logs `error_type=PDFValidationError` on validation failure.
- **Extraction:** Logs `operation=extraction`, `document_id`, `stage=extraction`, `status` on success/failure.
- **Chunking:** Logs `operation=chunking`, `document_id`, `stage=chunking`, `status` on success/failure.
- **Indexing:** Logs `operation=indexing`, `document_id`, `stage=indexing`, `status`, `duration_ms` on success; `error_type` on failure.
- **Retrieval:** Failure log enriched with `operation=retrieval`, `document_id`, `error_type`.
- **Ask (document + collection):** Logs `operation=generation`, `document_id`, `stage=generation`, `status`, `duration_ms` on success; `error_type` on failure.

### 4. `app/services/orchestrator.py` — Pipeline stage correlation
- Added `query_id` (format: `q_{hex12}`) generated per `orchestrate_query` call.
- All `add_stage()` log calls now include `operation=pipeline`, `query_id`, `document_id` alongside existing `stage`, `status`, `duration_ms`, `error_type`.
- Removed verbose fields (`input_count`, `output_count`, `fallback_used`, `fallback_reason`, `error_category`) from log `extra=` to keep structured output lean — these remain in the `PipelineStageTrace` response schema.

### 5. `tests/unit/test_logging.py` — Expanded from 5 → 12 tests

| Test | Covers |
|------|--------|
| `test_json_formatter_standard_and_structured_fields` | Core fields + None omission |
| `test_operation_and_query_id_fields` | `operation`, `query_id` extraction |
| `test_all_phase91_fields_present_when_set` | All 8 structured fields end-to-end |
| `test_secret_redaction_in_messages_and_extra` | OpenAI/Gemini/Bearer pattern scrubbing |
| `test_sanitize_structured_data_recursive` | Recursive dict/list secret redaction |
| `test_sensitive_keys_never_leak` | All known sensitive key names |
| `test_request_id_middleware_generates_and_propagates_header` | X-Request-ID generation + passthrough |
| `test_request_id_unique_per_request` | Uniqueness across requests |
| `test_orchestrator_logs_pipeline_stages` | Live orchestrator stage log capture |
| `test_error_log_structured_fields` | Error-level structured fields |
| `test_exception_log_includes_traceback` | `exception` field with traceback |
| `test_setup_logging_configures_root_handler` | Root logger configuration |

## Structured Log Field Reference

| Field | Source | Example |
|-------|--------|---------|
| `request_id` | `contextvars` via `RequestIDMiddleware` | `"a1b2c3d4e5f67890"` |
| `operation` | Endpoint/pipeline log `extra=` | `"upload"`, `"pipeline"`, `"generation"` |
| `document_id` | Endpoint/pipeline log `extra=` | `"doc_abc123"` |
| `query_id` | Orchestrator-generated per query | `"q_2d530f0fc23d"` |
| `stage` | Pipeline stage identifier | `"hybrid_retrieval"`, `"reranking"` |
| `duration_ms` | `time.perf_counter()` delta × 1000 | `45.67` |
| `status` | Outcome enum | `"success"`, `"error"`, `"fallback"` |
| `error_type` | Exception class name (when applicable) | `"ValueError"`, `"TimeoutError"` |

## Architecture Decisions

- **No new dependencies.** stdlib `logging` + `contextvars` only.
- **No abstractions.** No decorator-based tracing, no middleware chain — just `extra={}` dicts at call sites.
- **Lean log lines.** Verbose diagnostic fields stay in the API response (`PipelineStageTrace`), not in logs.
- **None suppression.** Fields set to `None` are excluded from JSON to avoid noisy `null` entries.
