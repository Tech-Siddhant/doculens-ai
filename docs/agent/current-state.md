## Current State

Phase 8.10 (Final MLOps Validation and Audit) is complete, verified, and audited. Phase 8 is 100% complete.


## Completed
- **Phase 8.10 Final MLOps Validation & Audit**:
  - Full system review completed across `app/`, `frontend/`, `tests/`, `scripts/`, `docs/`, `design/`, deployment, and CI.
  - Ponytail review and audit: 0 MUST FIX items, zero dead code, zero unnecessary dependencies, 0 npm vulnerabilities, lean ~180MB RAM and 917MB disk footprint.
  - End-to-end 12-stage pipeline validated from upload through multi-representation indexing, hybrid retrieval, cross-encoder reranking, grounded generation, citation validation, to UI rendering and Show Pipeline telemetry.
  - Failure modes tested and verified: bad PDFs, oversized files, PDF bombs, retrieval dropouts, reranker timeouts, LLM retries/auth errors, and out-of-domain abstention.
  - Evaluation harness benchmarked against all 4 baselines (`dense`, `dense_bm25`, `dense_bm25_visual`, `hybrid_reranked`) with strict regression gates passing.
  - Backend unit test suite: 446 passed, 2 skipped, 0 failures.
  - Frontend test suite: 13 passed, 0 failures, Next.js standalone build verified.

- **Phase 8.9 CI/CD Pipeline**:
  - **Automated CI Workflow (`.github/workflows/ci.yml`)**:
    - Triggers automatically on `push` and `pull_request` to `main`, plus `workflow_dispatch`.
    - Implemented with concurrency groups (`cancel-in-progress: true`) to avoid redundant runs on PR updates.
    - **Backend Job**: Standard `ubuntu-latest` runner with Python 3.11, `pip` caching, `pip install -e ".[dev]"`, full unit test suite execution via `pytest tests/unit` with mock offline environment, and automated offline retrieval evaluation via `python scripts/run_evaluation.py --config all --strict`.
    - **Frontend Job**: `ubuntu-latest` runner with Node 20, `npm` caching, `npm ci`, TypeScript check (`npm run type-check`), linting (`npm run lint`), Node test suite (`npm run test`), and Next.js standalone build (`npm run build`).
  - **Live Evaluation Workflow (`.github/workflows/live-evaluation.yml`)**:
    - Explicit `workflow_dispatch` only workflow with input parameters for configuration and evaluation boundary.
    - Securely binds GitHub secrets (`GEMINI_API_KEY`, `LLM_API_KEY`) without leaking credentials or running on untrusted PRs.
  - **Verification & Testing (`tests/unit/test_ci_config.py`)**:
    - Validates YAML syntax, trigger configurations, job structure, pip/npm dependency caching, command presence, absence of hardcoded secrets, and avoidance of heavy/paid infrastructure (no GPU runners, no Kubernetes).


- **Phase 8.8 Reproducible Deployment**:

  - **Store Lifecycle and Shared Object Identity**:
    - `app/evaluation/harness.py`: Implemented `EvaluationStores` ensuring strict object identity and lifecycle consistency between indexing and retrieval across vector stores, BM25 indices, visual vector stores, and cross-encoder rerankers.
    - Implemented `create_isolated_stores()` for hermetic in-memory execution and `get_shared_stores()` for singleton coordination.
    - Added `index_evaluation_document()` to extract, chunk, embed, and index benchmark PDF documents into matching store instances.
  - **Standard Baseline Configurations**:
    - Maintained 4 named baseline configurations in `app/evaluation/harness.py`:
      1. `dense`: Dense vector retrieval only (`retrieval_types=["dense"]`, weights `{"dense": 1.0}`, reranker disabled)
      2. `dense_bm25`: Dense + BM25 weighted hybrid retrieval (`retrieval_types=["dense", "bm25"]`, weights `{"dense": 0.7, "bm25": 0.3}`, reranker disabled)
      3. `dense_bm25_visual`: Dense + BM25 + Visual weighted retrieval (`retrieval_types=["dense", "bm25", "visual"]`, weights `{"dense": 0.5, "bm25": 0.3, "visual": 0.2}`, reranker disabled)
      4. `hybrid_reranked`: Dense + BM25 + Visual weighted hybrid retrieval + BAAI/bge-reranker-base cross-encoder reranking
  - **Regression Detection Engine**:
    - `app/schemas/evaluation.py`: Added `MetricDelta` and `RegressionReport` data models for structured tracking of deltas, percentage changes, regressions, and improvements.
    - `app/evaluation/regression.py`: Implemented `compare_against_baseline()` with configurable `RegressionThresholds` (e.g. max 5% quality drop allowed on Recall@K, MRR@K, Context Precision/Recall, Faithfulness, Relevancy, Citation Precision/Recall, Abstention Accuracy, and max 25% latency increase).
    - Returns structured `RegressionReport` with status (`PASSED`, `REGRESSION_DETECTED`, `IMPROVED`, `NEUTRAL`), lists of flagged degradations/improvements, and per-metric delta records.
  - **Automated CLI Harness (`scripts/run_evaluation.py`)**:
    - Created executable CLI tool supporting `--dataset`, `--config` (`dense`, `dense_bm25`, `dense_bm25_visual`, `hybrid_reranked`, or `all`), `--corpus-pdf`, `--eval-type` (`retrieval`, `generation`, `end_to_end`), `--baseline`, `--save-baseline`, `--output`, `--markdown-output`, and `--strict`.
    - Produces clean terminal summary tables, machine-readable JSON artifacts, and structured Markdown reports.
    - Complete CI compatibility: retrieval evaluation executes hermetically without live external API keys.
  - **Canonical Evaluation Benchmark Document**:
    - Added `data/samples/doculens-architecture-v1.pdf` providing multi-page ground truth text and layout corresponding to `data/gold_dataset.jsonl`.
  - **Testing**:
    - `tests/unit/test_regression_detection.py`: Added 8 tests covering baseline comparisons, quality improvements, metric regressions, tolerance boundaries, latency regressions/speedups, and serialization.
    - `tests/unit/test_evaluation_automation.py`: Added 7 tests covering store identity lifecycle, store isolation, corpus indexing, baseline definitions, execution across all baselines, and formatter outputs.
    - Full backend test suite: 458 passed, 2 skipped, 0 failures.
    - Full frontend test suite: 13 passed, 0 failures.

- **Phase 8.6 API and Backend Hardening**:
  - **Health Endpoints & True Readiness Evaluation**:
    - `app/schemas/health.py`: Added `ComponentStatus` and `ReadinessResponse` models alongside `HealthResponse`.
    - `app/api/routes/health.py`: Implemented distinct liveness (`GET /health` and `GET /health/live`) and deep readiness endpoints (`GET /health/ready` and `GET /health/readiness`).
    - Dependency inspection checks: local upload storage writeability, text Qdrant vector store collection health, visual Qdrant vector store collection health, and LLM provider credentials configuration.
    - True dependency health reporting: returns HTTP `503 Service Unavailable` on `/health/ready` whenever any critical dependency is unavailable (never falsely reporting healthy).
  - **Centralized Safe Error Handling & Zero-Leakage Exception Handlers**:
    - `app/api/main.py`: Registered global exception handlers for `RequestValidationError` (clean, sanitized 422 JSON), `HTTPException` (standard sanitized error response), and unhandled `Exception` (clean generic 500 JSON response preventing disclosure of stack traces, local filesystem paths, secrets, or provider API keys).
    - `app/api/routes/documents.py`: Scrubbed exception details for provider timeout, rate limiting, authentication, and unavailable errors, returning static safe user-facing error messages.
  - **Request Validation & Resource Protection**:
    - `app/services/validator.py` & `app/ingestion/validator.py`: Added `MAX_PAGE_COUNT = 1000` bounds check and sanitized exception messages to protect against PDF bomb / decompression memory exhaustion attacks.
    - `app/api/routes/documents.py`: Added pagination parameters (`limit: int = 1..200`, default 50; `offset: int >= 0`, default 0) to `GET /documents`, preventing unbounded document list responses.
    - Added upper and lower bounds validation to chunking and indexing parameters (`chunk_size` 1..10000, `chunk_overlap` 0..<chunk_size) and page image requests (`page_number` 1..5000).
    - Schema bounds on query lengths (`1..2000`), retrieval limits (`top_k` 1..50), and fusion parameters (`rrf_k` 1..1000).
  - **Testing**:
    - `tests/unit/test_health.py`: Expanded with tests for `/health`, `/health/live`, `/health/ready`, and simulated dependency failure returning HTTP 503.
    - `tests/unit/test_api_hardening.py`: Added 5 comprehensive tests validating document list pagination, parameter bounds, PDF bomb rejection, global unhandled 500 error sanitization, and structured 422 validation error formatting.
    - Full backend test suite: 443 passed, 2 skipped, 0 failures.
    - Full frontend test suite: 13 passed, 0 failures.

- **Phase 8.5 Security Hardening**:
  - **Upload Validation & Path Traversal Guards**:
    - `app/services/validator.py`: Added strict filename inspection rejecting path separators (`/`, `\\`) and null bytes (`\x00`). Validates extension (`.pdf`), size bounds, magic header (`%PDF-`), and structural readability.
    - `app/api/routes/documents.py`: Implemented streaming chunked upload reader (`_read_upload_limited`) rejecting oversized request bodies early (HTTP 413) without unbounded memory buffering.
    - Path parameter validation: `is_valid_document_id` enforces strict format (`doc_[0-9a-f]{12}`), preventing traversal via malicious IDs.
  - **Prompt Injection & LLM Safety**:
    - `app/services/generator.py`: Document excerpts treated strictly as untrusted evidence data. Structural prompt boundaries (`--- EVIDENCE ---`, `--- END EVIDENCE ---`, `--- USER QUESTION ---`, `--- END USER QUESTION ---`) neutralized inside document content before prompt assembly.
    - System prompt and grounding rules explicitly instruct the LLM that evidence content contains data, not commands.
    - `app/services/citation_validator.py`: Rejects fabricated citations (e.g. references to out-of-bounds indices), marking ungrounded responses accordingly.
  - **Request Schema Validation & Bounds**:
    - `app/schemas/generation.py` & `app/schemas/retrieval.py`: Bound `question` and `query` strings (`min_length=1, max_length=2000`) and `top_k` (`ge=1, le=50`).
    - FastAPI/Pydantic returns HTTP 422 on oversized, malformed, or out-of-range payloads.
  - **Error Handling & Information Leakage**:
    - `app/api/routes/documents.py`: Generic 500 exceptions now log details internally while returning generic user-facing messages (e.g. "Extraction failed."), preventing leakage of internal filesystem paths, stack traces, or credentials.
  - **CORS & Rate Limiting**:
    - `app/core/config.py`: Replaced wildcard CORS with restrictive default (`http://localhost:3000,http://127.0.0.1:3000`).
    - `app/core/ratelimit.py`: Minimal in-process rate limiter (`InMemoryRateLimiter`) with configurable thresholds (`RATE_LIMIT_ENABLED`, `RATE_LIMIT_MAX_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`), returning HTTP 429 when enabled and exceeded.
  - **Secrets Hygiene**:
    - Verified `.env` and `.env.local` are git-ignored; `.env.example` contains only template placeholders.
    - Structured logger masks API keys and credentials automatically.
  - **Security Test Suite**:
    - `tests/unit/test_security.py`: 32 comprehensive tests covering path traversal, upload validation, oversized request bounds, prompt injection neutralization, citation fabrication rejection, 500 error hygiene, CORS origin matching, and rate limiting.
- **Phase 8.1 Configuration & Environment Hardening**:
  - `app/core/config.py`: Wrapped sensitive API keys and secrets (`GEMINI_API_KEY`, `LLM_API_KEY`, `QDRANT_API_KEY`) in Pydantic `SecretStr` to prevent accidental logging or exposure.
  - Added fail-fast `@model_validator(mode="after")` to validate required API keys whenever `LLM_PROVIDER` is `gemini` or `openai`.
  - Added configurable `LOG_LEVEL` and `CORS_ORIGINS` settings.
  - Updated consumers (`app/services/llm_provider.py`, `app/services/vector_store.py`) to access secrets safely via `.get_secret_value()`.
  - Configured dynamic CORS middleware in `app/api/main.py`.
  - Updated `.env.example` and `README.md` with explicit environment variable documentation, required keys, and secure mock-first defaults.
  - Added unit test suite in `tests/unit/test_config.py` validating `SecretStr` masking, required provider keys, and env overrides.
- **Phase 8.2 Structured Logging**:
  - `app/core/logger.py`: Implemented standard library-based `JSONFormatter`, recursive structured secret sanitizer (`sanitize_structured_data`), string secret scrubber (`redact_text`), and async context variable `request_id_var`.
  - `RequestIDMiddleware`: Added lightweight HTTP middleware extracting or generating `X-Request-ID` and correlating API requests through the entire async lifecycle.
  - Structured fields support: `timestamp`, `level`, `logger`, `message`, `request_id`, `document_id`, `stage`, `duration_ms`, `status`, `error_type`.
  - Integrated structured stage logging in `app/services/orchestrator.py` and `app/api/routes/documents.py`.
  - Automated redaction of API keys (OpenAI, Gemini, Bearer tokens) and sensitive dictionary keys.
  - Added test suite `tests/unit/test_logging.py` covering formatting, request ID generation and propagation, field sanitization, and secret redaction.
- **Phase 8.3 Pipeline Telemetry (Observable Stage Metrics)**:
  - `app/schemas/generation.py`: Extended `PipelineStageTrace` with `input_count`, `output_count`, and `error_category` fields. Extended `PipelineSummary` with `citation_validation_duration_ms`.
  - `app/services/orchestrator.py`: Rewrote `add_stage()` to use keyword-only args for clarity. Added `sanitize_structured_data()` pass on every stage's `details` dict before recording. Promoted stage variables to named kwargs: `input_count`, `output_count`, `fallback`, `f_reason`, `err_category`, `err_msg`. Extracted `retrieval_dur`, `rerank_dur`, `gen_dur`, `citation_val_dur` as named floats tracked across all 8 stages. Added generation-side token usage metrics (`prompt_tokens`, `completion_tokens`, `total_tokens`) from LLM response. Fixed all raw `(t1 - t0) * 1000` expressions with `max(0.0, ...)` guard. All `PipelineSummary` latency fields now populated.
  - `frontend/src/types/index.ts`: Synchronized `PipelineStageTrace` and `PipelineSummary` TypeScript interfaces with backend schema additions.
  - `frontend/src/components/qa/PipelineTimeline.tsx`: Added citation validation latency chip to the header badge bar. Added `In: N → Out: M` flow counts panel per stage in the expanded detail view.
  - `tests/unit/test_telemetry_inspection.py`: Expanded from 2 to 5 tests covering: all 8 stages and correct ordering, summary latency fields all populated and non-negative, `input_count`/`output_count`/`error_category` field presence, fallback stage handling for empty evidence, error pipeline failure propagation, and JSON serialization safety (no secrets in trace output).
- **Phase 8.4 Reliability and Failure Handling**:
  - `app/core/config.py`: Added `LLM_TIMEOUT: float = 30.0` and `LLM_MAX_RETRIES: int = 2` settings.
  - `app/services/llm_provider.py`: Implemented timeout handling (`ProviderTimeoutError`), bounded retries with exponential backoff for transient errors (HTTP 429, 502, 503, 504, `ConnectError`), immediate fast failure on 401/400 (preventing retry storms), malformed response parsing (missing choices/content, non-JSON), and token usage normalization.
  - `app/services/hybrid.py`: Added per-channel fail-soft handling. If Dense retrieval (e.g. Qdrant) fails or BM25 index fails, the pipeline logs a warning and proceeds with the available retrieval channels.
  - `app/services/orchestrator.py`: Implemented fail-soft reranker fallback. If cross-encoder reranking fails, the pipeline continues using raw hybrid retrieval results, records stage status as `status="fallback"`, and executes downstream evidence selection cleanly.
  - `app/api/routes/documents.py`: Mapped `ProviderTimeoutError` -> HTTP 504, `ProviderRateLimitError` -> HTTP 429, `ProviderUnavailableError` -> HTTP 503, and `ProviderAuthenticationError` -> HTTP 502 (with scrubbed error detail).
  - `frontend/src/lib/api.ts`: Enhanced `getRecommendedActionForError()` to provide actionable user guidance for HTTP 504 (Gateway Timeout), 429 (Rate Limit), 503 (Service Unavailable), and 502 (Bad Gateway / Auth Error).
  - `tests/unit/test_reliability_failure.py`: Added 15 comprehensive unit tests covering all failure modes, fallbacks, groundedness, retries, and token tracking.
- **Dockerfiles & Deployment**:
  - `Dockerfile` (Backend): Production-ready multi-stage Python 3.11-slim container with fast compilation, non-root directories, health check against `/api/v1/health`, and entrypoint ASGI uvicorn server.
  - `frontend/Dockerfile`: Multi-stage Next.js 16 standalone production container (`deps` -> `builder` -> `runner`) with non-root `nextjs` user, lightweight Alpine base, and HTTP health check.
  - `.dockerignore` for root and frontend ensuring lean, fast build contexts excluding `.venv`, `node_modules`, cache, and local secrets.
  - `docker-compose.yml` & `docker-compose.dev.yml`: Production and dev container orchestration.
- **Validation**:
  - Full backend unit test suite: 402 passed, 2 skipped, 0 failures.
  - Backend integration test suite: 18 passed.
  - Frontend test suite: 13 passed, 0 failures.

## Next Steps
- System complete & reliable. Ready for production deployment or evaluation benchmarks.



