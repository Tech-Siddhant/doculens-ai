# DocuLens AI

An AI-powered document processing and analysis API built with FastAPI.

## Getting Started

### Prerequisites
- Python >= 3.10

### Installation

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Run the development server:
   ```bash
   uvicorn app.api.main:app --reload
   ```

4. Check health endpoint:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```
