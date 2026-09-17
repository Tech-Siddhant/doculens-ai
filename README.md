# DocuLens AI

> Multimodal Document Intelligence Platform

DocuLens AI processes complex documents (research papers, technical PDFs with multi-column layouts, tables, figures, charts) using textual and visual evidence retrieval to produce grounded answers with page-level citations.

---

## Directory Structure

```text
doculens-ai/
├── README.md
├── .gitignore
├── .env.example
├── pyproject.toml
│
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── documents.py
│   │       └── health.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── embedding.py
│   │   ├── generation.py
│   │   ├── health.py
│   │   ├── retrieval.py
│   │   └── vector_store.py
│   │
│   └── services/
│       ├── __init__.py
│       ├── chunker.py
│       ├── embedder.py
│       ├── extractor.py
│       ├── generator.py
│       ├── llm_provider.py
│       ├── retriever.py
│       ├── storage.py
│       ├── validator.py
│       └── vector_store.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── unit/
│       ├── __init__.py
│       ├── test_chunker.py
│       ├── test_config.py
│       ├── test_documents_api.py
│       ├── test_embedder.py
│       ├── test_extractor.py
│       ├── test_generator.py
│       ├── test_health.py
│       ├── test_retriever.py
│       ├── test_storage.py
│       ├── test_validator.py
│       └── test_vector_store.py
│
├── docs/
│   ├── problem-statement.md
│   ├── architecture.md
│   ├── workflows.md
│   ├── project-plan.md
│   ├── engineering-standards.md
│   └── agent/
│       └── master-prompt.md
│
└── data/
    ├── samples/
    └── uploads/
```

---

## Quickstart

### Option A: Docker Compose (Recommended)

Run the full production stack (Frontend UI and FastAPI Backend with embedded vector db) in a single command:

```bash
# 1. Clone and configure environment
cp .env.example .env

# 2. Build and launch containers
docker compose up --build -d

# 3. Verify health
curl -f http://localhost:8000/api/v1/health
```

- **Frontend UI**: [http://localhost:3000](http://localhost:3000)
- **API Docs (Swagger)**: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)

See [Deployment Guide](docs/deployment.md) for hot-reloading dev compose commands, persistent volume locations, and advanced settings.

---

### Option B: Local Bare-Metal Setup

#### 1. Backend Environment Setup

```bash
cd doculens-ai
cp .env.example .env
pip install -e ".[dev]"
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

- API Docs: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- Health Check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

#### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```
The frontend starts on `http://localhost:3000` and automatically connects to the FastAPI backend running on port 8000.

#### 3. Run Tests & CI Verification

```bash
# Run backend test suite
pytest tests/unit

# Run offline retrieval evaluation
python scripts/run_evaluation.py --config all --strict

# Run frontend tests & validation
cd frontend
npm run type-check
npm run test
npm run build
```

---

## Evaluation & Benchmarks

DocuLens AI includes an automated evaluation harness (`scripts/run_evaluation.py`) that measures Recall@K, MRR@K, Context Precision, and Context Recall across named baselines:

```bash
# Offline Retrieval Evaluation (no API keys required, ideal for CI)
python scripts/run_evaluation.py --config all --strict

# Live End-to-End Evaluation (requires GEMINI_API_KEY)
export GEMINI_API_KEY="your_api_key"
export LLM_PROVIDER="gemini"
python scripts/run_evaluation.py --config hybrid_reranked --eval-type end_to_end --strict
```

---

## Continuous Integration (CI/CD)

Automated GitHub Actions workflows are defined in `.github/workflows/`:
- **`ci.yml`**: Runs on pull requests and pushes to `main`. Executes backend unit tests, offline evaluation benchmarks, frontend type checks, tests, and production build without requiring external API keys or heavy GPU runners.
- **`live-evaluation.yml`**: On-demand manual workflow (`workflow_dispatch`) for live LLM benchmarking with injected GitHub secrets.


---

## Environment Variables

DocuLens uses `.env` for configuration. The system validates these at startup to prevent silent failures. Secrets are redacted from logs automatically.

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | Target environment (`development`, `testing`, `production`) |
| `LOG_LEVEL` | `INFO` | Application log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed origins |
| `RATE_LIMIT_ENABLED` | `false` | Enable in-process rate limiting |
| `QDRANT_LOCATION` | `:memory:` | String `:memory:` for ephemeral runs or mapped local path (e.g., `data/qdrant`) |
| `LLM_PROVIDER` | `mock` | `mock` (offline testing), `gemini`, or `openai` |
| `GEMINI_API_KEY` | None | **Required** if `LLM_PROVIDER=gemini` |
| `LLM_API_KEY` | None | **Required** if `LLM_PROVIDER=openai` |

See `.env.example` for the full list of indexing, retrieval, and tuning parameters.

---

## Documentation

- [Problem Statement](docs/problem-statement.md)
- [System Architecture](docs/architecture.md)
- [System Workflows](docs/workflows.md)
- [Project Plan](docs/project-plan.md)
- [Deployment Guide](docs/deployment.md)
- [Engineering Standards](docs/engineering-standards.md)
- [AI Coding Agent Master Prompt](docs/agent/master-prompt.md)

