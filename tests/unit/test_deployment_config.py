"""Unit tests for Phase 8.8 Reproducible Deployment configuration."""

from pathlib import Path


def test_dockerfiles_exist() -> None:
    """Verify backend and frontend Dockerfiles exist and have proper configurations."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    backend_dockerfile = root_dir / "Dockerfile"
    frontend_dockerfile = root_dir / "frontend" / "Dockerfile"

    assert backend_dockerfile.exists(), "Root backend Dockerfile must exist"
    assert frontend_dockerfile.exists(), "Frontend Dockerfile must exist"

    backend_content = backend_dockerfile.read_text()
    assert "python:3.11-slim" in backend_content
    assert "uvicorn" in backend_content
    assert "EXPOSE 8000" in backend_content
    assert "HEALTHCHECK" in backend_content
    assert "USER appuser" in backend_content
    assert "cuda" not in backend_content.lower()
    assert "nvidia" not in backend_content.lower()

    frontend_content = frontend_dockerfile.read_text()
    assert "node:20-alpine" in frontend_content
    assert "standalone" in frontend_content
    assert "EXPOSE 3000" in frontend_content
    assert "HEALTHCHECK" in frontend_content
    assert "USER nextjs" in frontend_content


def test_docker_compose_configs_exist() -> None:
    """Verify docker-compose.yml and docker-compose.dev.yml exist and contain required services."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    compose_prod = root_dir / "docker-compose.yml"
    compose_dev = root_dir / "docker-compose.dev.yml"

    assert compose_prod.exists(), "docker-compose.yml must exist"
    assert compose_dev.exists(), "docker-compose.dev.yml must exist"

    prod_content = compose_prod.read_text()
    for service in ["backend:", "frontend:"]:
        assert service in prod_content, f"Service {service} must be defined in production compose"
    assert "doculens_data:" in prod_content, "Persistent storage volume must be defined"

    dev_content = compose_dev.read_text()
    for service in ["backend:", "frontend:"]:
        assert service in dev_content, f"Service {service} must be defined in development compose"


def test_dockerignore_files_exist() -> None:
    """Verify .dockerignore files are configured to exclude secrets, node_modules, and .venv."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    root_ignore = root_dir / ".dockerignore"
    frontend_ignore = root_dir / "frontend" / ".dockerignore"

    assert root_ignore.exists()
    assert frontend_ignore.exists()

    root_content = root_ignore.read_text()
    assert ".venv" in root_content
    assert ".git" in root_content
    assert ".env" in root_content

    frontend_content = frontend_ignore.read_text()
    assert "node_modules" in frontend_content
    assert ".next" in frontend_content


def test_env_example_template() -> None:
    """Verify .env.example contains necessary operational keys."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    env_example = root_dir / ".env.example"

    assert env_example.exists()
    content = env_example.read_text()

    assert "PROJECT_NAME" in content
    assert "UPLOAD_DIR" in content
    assert "QDRANT_LOCATION" in content
    assert "LLM_PROVIDER" in content
    assert "GEMINI_API_KEY" in content

