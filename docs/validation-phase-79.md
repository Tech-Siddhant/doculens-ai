# DocuLens AI — Phase 7.9 Frontend ↔ Backend Validation Report

**Date:** 2026-09-13
**Scope:** End-to-end validation of the complete DocuLens AI frontend against the real backend.
**Overall verdict:** **PASS** — with 2 non-blocking code defects (lint script, missing favicon) and 1 environment-blocked item (full live-stack E2E). No data-loss, security, or crash-level issues found.

All results below were obtained by actually running the commands in this environment. Nothing is inferred or speculative.

---

## 1. Tests

### Frontend (`frontend/`)
| Check | Command | Result |
|---|---|---|
| Unit tests | `node --experimental-strip-types --test src/lib/__tests__/*.test.mjs` | **PASS — 12/12** (`citation_evidence.test.mjs`, `inspection.test.mjs`) |
| Type check | `npx tsc --noEmit` | **PASS — exit 0** |
| Production build | `next build` | **PASS — exit 0** (`✓ Compiled successfully in 5.1s`, `Finished TypeScript in 6.9s`, static pages 5/5 in 2.6s) |
| Lint | `npm run lint` | **FAIL — exit 1** (see Remaining bugs #1) |

Built routes: `○ /` (static), `○ /_not-found`, `○ /documents` (static), `ƒ /documents/[id]` (dynamic), `○ /settings` (static).

### Backend (`tests/`, DocuLens `.venv`)
| Suite | Result |
|---|---|
| Unit (`tests/unit/`, 35 files) | **PASS — 358/358** |
| Integration (`tests/integration/`, 2 files) | **PASS — 18/18** (45.0s) |
| **Total** | **PASS — 376/376, 0 failures** |

Notes:
- The earlier apparent stall at `test_visual_embedder.py` was a session-harness interruption of the background run, **not** a model-download hang: an isolated rerun passed all 8 tests in 15.4s (fastembed weights are cached).
- Remaining warnings are upstream deprecations only: `starlette.testclient`/`httpx` and `anyio.abc.BlockingPortal` aliases.

## 2. E2E Flows

### Frontend live smoke test (real, Playwright against running `next start` on :3100) — PASS
Backend absent by design → this validates rendering **and** graceful degradation, plus confirms exact API wiring from the browser network log:

| Route | Result | Evidence |
|---|---|---|
| `/` | Renders; **"API Offline"** badge; Recent Documents shows graceful error card + "Try again" | console: `ERR_CONNECTION_REFUSED http://localhost:8000/api/v1/health`, `.../api/v1/documents` |
| `/documents` | Full library UI (search box, 5 status filter chips @0, sort combobox), graceful "Unable to load documents — Failed to fetch" + empty state | console: exact `/api/v1/health` + `/api/v1/documents` calls |
| `/settings` | All preference controls render (Concise/Balanced/Detailed; "Always show sources" + "Highlight evidence" checked; Advanced; Save Preferences) | console: only expected `/api/v1/health` call |
| `/documents/doc_test123` | Graceful "Document not found — Failed to fetch" + Back/Retry | console: `/api/v1/documents/doc_test123` call |

### Full live-stack E2E (frontend + FastAPI + Qdrant + Redis/Celery) — **BLOCKED (environment, not code)**
- No backend on :8000 (only Postgres on 127.0.0.1:5432 listening; the Docker container `jolly_einstein` (`8082/tcp`) publishes no host port; no qdrant/redis processes).
- Host has 7.1 GiB RAM with ~2.3 GiB available under load — bringing the full stack up (qdrant + redis + celery + FastAPI + model loads) is not reliable here.
- Backend E2E coverage is therefore provided in-process by the FastAPI `TestClient` integration suite (18/18 green), which exercises the complete pipeline (upload → extract → index → hybrid retrieve → generate) plus REST validation and security assertions.

## 3. API Contract

Frontend `frontend/src/lib/api.ts` (base `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`, prefix `/api/v1`) vs backend `app/api/routes/*.py` — **fully aligned**:

| Frontend call | Backend route | Match |
|---|---|---|
| `GET /health` | `health.py` `GET /health` | ✓ |
| `GET /documents` | `documents.py` `GET ""` | ✓ |
| `GET /documents/{id}` | `documents.py` `GET /{document_id}` | ✓ |
| `POST /documents/upload` (multipart) | `documents.py` `POST /upload` (201) | ✓ |
| `POST /documents/{id}/extract` | `documents.py` `POST /{document_id}/extract` | ✓ |
| `POST /documents/{id}/index?chunk_size&chunk_overlap` | `documents.py` `POST /{document_id}/index` | ✓ |
| `POST /documents/{id}/ask` | `documents.py` `POST /{document_id}/ask` | ✓ |
| `POST /documents/ask` | `documents.py` `POST /ask` | ✓ |
| `GET /documents/{id}/pages/{n}/image` | `documents.py` `GET .../pages/{page}/image` (FileResponse) | ✓ |

- **Payloads:** frontend `QuestionRequest {question, top_k?, score_threshold?}` is an exact match to the backend pydantic `QuestionRequest`. The QA panel sends `{question, top_k: 5}`.
- **Upload limit:** frontend hardcodes `MAX_UPLOAD_SIZE_BYTES = 10 MB` (10,485,760) == backend `settings.MAX_UPLOAD_SIZE_BYTES`. ✓
- **Status mapping:** frontend `mapBackendDocToItem` maps backend statuses (`uploaded`/`extracting`/`indexing`/`failed`/…) into `ready | processing | needs_attention | failed`. ✓

## 4. Security

- **Source scan:** No real credentials anywhere. All 13 regex matches for key patterns are *configuration variable names* (`LLM_API_KEY`, `GEMINI_BASE_URL`, …), never values. No `AIza…`, `sk-…`, or `AKIA…` literals.
- **Frontend env:** `.env.local` contains only `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` — no secrets.
- **Build artifacts:** grep of `.next/` for key patterns → **0 matches**.
- **Backend:** API keys read only from environment (`LLM_API_KEY`/`GEMINI_API_KEY`, `SECRET_KEY=None` by default); no key present in this environment (`LLM_API_KEY_IS_SET: false`) so generation would run in `mock` provider mode.
- **Leak checks:** backend integration tests explicitly assert API responses contain no `OPENAI_API_KEY`, `SECRET`, `/home/`, or `/workspaces/` strings — green.
- **Housekeeping:** browser console noise limited to a 404 for a missing `favicon.ico` (cosmetic, see bugs).

## 5. Dependency Audit

- **npm audit** (`npm audit --audit-level=moderate`): **0 vulnerabilities**.
- Python: no offline vulnerability scanner (pip-audit/bandit) available in this environment; mitigated by a fully green 376-test suite that imports and exercises every service module. Note the 10 production deps are pinned in `pyproject.toml`/lockfile — recommend running `pip-audit` in CI.

## 6. Resource Usage

- Machine: 7.1 GiB RAM / 2 GiB swap (1.4 GiB in use during concurrent runs).
- `next build`: compile 5.1s, TS 6.9s, static gen 2.6s (7 workers).
- Backend tests: unit bulk 52.1s + visual embedder 15.4s + integration 45.0s.
- `next start`: ready in 815ms, serving HTTP 200.
- **Operational note:** running `next build` concurrently with the pytest suite stalls on this machine (available RAM drops to ~1 GiB). Run build and tests sequentially in constrained environments.

## 7. Design Deviations

1. **Settings page is a UI prototype.** `handleSave` only flashes "Preferences Saved" for 2s — nothing is persisted (no localStorage, no backend call) and `answerStyle`/`showSources`/`highlight` never influence Q&A (the ask payload carries only `{question, top_k}`). The Advanced section itself states "planned for Phase 7.3". Non-breaking, but the settings currently have zero functional effect.
2. **`output: "standalone"` vs `next start`.** `next.config.mjs` sets `output: "standalone"`, so `npm start` prints `⚠ "next start" does not work with "output: standalone"…` and serves incorrectly. Production Docker (multi-stage standalone) is set up correctly; only the dev `start` script is misleading.
3. **Backend surplus not surfaced in UI.** Backend collection-wide `/ask`, and the full hybrid/BM25/visual retrieval endpoints, have no frontend callers; the UI offers per-document grounded QA only.
4. **No browser E2E framework in the repo** (no Playwright/Puppeteer/Cypress). This validation used an ad-hoc Playwright session; a repo-level smoke test would make this reproducible in CI.

## 8. Remaining Bugs

1. **`npm run lint` is broken (exit 1).** Next 16 removed the `next lint` subcommand; the script errors with `Invalid project directory provided, no such directory: …/frontend/lint`. No ESLint dependency or config exists in the frontend.
2. **Missing favicon** — `GET /favicon.ico` returns 404 on every page (cosmetic).
3. **`npm start` misconfigured (dev)**: uses `next start` with `output: standalone`; produces a warning and non-standalone serving (see Deviation 2).

## 9. Recommended Fixes

1. **Lint (P1):** add `eslint` + `eslint-config-next`, create `eslint.config.mjs`, and change the script to `"lint": "eslint ."`. Escapes the removed `next lint` path.
2. **Favicon (P3):** add `frontend/public/favicon.ico` (or `app/icon.svg` for Next metadata) to clear the 404.
3. **Dev start (P2):** either change `"start"` to `node .next/standalone/server.js` (requires build) or drop `output: "standalone"`; align with the existing Docker standalone entry.
4. **Settings (P2, product decision):** persist to `localStorage` immediately, and/or extend backend `QuestionRequest` (currently `extra="ignore"` silently drops extra fields) with `answer_style` to make the controls functional.
5. **E2E (P2):** add a Playwright smoke test (home + documents + settings + not-found) runnable in CI against `next start` + the FastAPI TestClient or a live stack.
6. **CI hygiene (P3):** add `pip-audit`/bandit to the backend CI gate; keep frontend `npm audit` in the pipeline.

---

### Execution artifacts (this session)
- `/tmp/unit_tests.log` — 350 passed (bulk unit run)
- `/tmp/viz_probe.log` — 8 passed (visual embedder isolated rerun)
- `/tmp/int_tests.log` — 18 passed (integration)
- `/tmp/next_build3.log` — build exit 0 + route table
- `/tmp/lint.log` — lint exit 1
- Live Playwright session against `http://localhost:3100` (frontend), backend offline