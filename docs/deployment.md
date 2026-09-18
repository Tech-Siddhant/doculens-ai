# Deployment & Containerization Guide

DocuLens AI provides reproducible deployment workflows via **Docker Compose** as well as local bare-metal configurations. The system is designed to be self-contained and run efficiently on a single host.

---

## 1. Quick Start with Docker Compose

### Prerequisites
- Docker Engine >= 24.0
- Docker Compose v2 (e.g. `docker compose`)

### Production Stack
The production compose configuration starts 2 minimal, self-contained services:
- **`frontend`**: Next.js 16 standalone server running on port `3000`.
- **`backend`**: FastAPI ASGI server running on port `8000`, containing the full pipeline, embedded AI models, and an embedded Qdrant vector database (`QDRANT_LOCATION=":memory:"` by default, or persistent file-storage).

No external databases (PostgreSQL, standalone Qdrant, Redis) are required for standard deployment! 

### Step-by-Step Launch
1. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Edit .env to set your GEMINI_API_KEY or LLM_API_KEY
   ```

2. **Build and Run the Services:**
   ```bash
   docker compose up --build -d
   ```

3. **Verify Service Health:**
   ```bash
   # Backend Health Checks (Live and Ready)
   curl -f http://localhost:8000/api/v1/health
   curl -f http://localhost:8000/api/v1/health/ready
   
   # Frontend Web Interface
   curl -I http://localhost:3000
   ```

4. **Access Applications:**
   - **Frontend Web UI**: [http://localhost:3000](http://localhost:3000)
   - **Backend API Docs (Swagger UI)**: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)

5. **Stop the Stack:**
   ```bash
   docker compose down
   # To remove persisted volumes (clears all uploads, indices, configurations):
   docker compose down -v
   ```

### Storage & Persistence
Document uploads and rendered page images are stored in a Docker volume (`doculens_data`) mounted to the `/app/data` workspace. By default, Qdrant runs embedded in-memory (`QDRANT_LOCATION=":memory:"`) for zero external database dependencies. For persistent on-disk vector storage, configure `QDRANT_PATH=/app/data/qdrant`.

---

## 2. Local Development Stack (Hot-Reloading)

To develop with live source code mounting and hot reload:

```bash
docker compose -f docker-compose.dev.yml up --build
```

---

## 3. Bare-Metal Local Setup

If running directly without Docker:

### Backend
```bash
# 1. Virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Dependencies
pip install -e ".[dev]"

# 3. Environment configuration
cp .env.example .env

# 4. Start backend
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 4. Architecture & Port Reference

| Service | Port | Internal URL | Purpose |
|:---|:---|:---|:---|
| **Frontend** | `3000` | `http://frontend:3000` | Next.js User Interface & Citation Viewer |
| **Backend** | `8000` | `http://backend:8000` | FastAPI REST API, Embedded Qdrant, Orchestrator |

