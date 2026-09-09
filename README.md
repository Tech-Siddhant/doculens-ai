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

### 1. Environment Setup

```bash
cd doculens-ai
cp .env.example .env
pip install -e ".[dev]"
```

### 2. Run API Server

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

- API Docs: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- Health Check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

### 3. Run Tests

```bash
pytest tests/unit
```

---

## Documentation

- [Problem Statement](docs/problem-statement.md)
- [System Architecture](docs/architecture.md)
- [System Workflows](docs/workflows.md)
- [Project Plan](docs/project-plan.md)
- [Engineering Standards](docs/engineering-standards.md)
- [AI Coding Agent Master Prompt](docs/agent/master-prompt.md)
