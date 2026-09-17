# DocuLens AI - Backend Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install minimal curl for container health check (no GPU or heavy compiler toolchains)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and data directories
RUN useradd -m -u 1001 -s /bin/bash appuser && \
    mkdir -p /app/data/uploads /app/data/rendered_pages /app/data/qdrant && \
    chown -R appuser:appuser /app

# Install python dependencies first for layer caching
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy application source code
COPY app ./app
RUN chown -R appuser:appuser /app

USER appuser

# Expose FastAPI backend port
EXPOSE 8000

# Health check against authoritative health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


