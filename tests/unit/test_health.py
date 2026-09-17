from unittest.mock import patch
from fastapi.testclient import TestClient

from app.core.config import settings


def test_health_check(client: TestClient) -> None:
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {
        "status": "healthy",
        "version": settings.VERSION,
    }


def test_health_live_endpoint(client: TestClient) -> None:
    response = client.get(f"{settings.API_V1_STR}/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == settings.VERSION


def test_health_readiness_healthy(client: TestClient) -> None:
    response = client.get(f"{settings.API_V1_STR}/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == settings.VERSION
    assert "components" in data
    assert "storage" in data["components"]
    assert "vector_store" in data["components"]
    assert "visual_vector_store" in data["components"]
    assert "llm_provider" in data["components"]
    assert data["components"]["storage"]["status"] == "healthy"
    assert data["components"]["vector_store"]["status"] == "healthy"


def test_health_readiness_dependency_failure_returns_503(client: TestClient) -> None:
    with patch("app.services.vector_store.vector_store.get_stats", side_effect=RuntimeError("Qdrant unreachable")):
        response = client.get(f"{settings.API_V1_STR}/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["components"]["vector_store"]["status"] == "unhealthy"
        assert "Vector store unavailable" in data["components"]["vector_store"]["details"]


def test_health_readiness_storage_failure_returns_503(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "UPLOAD_DIR", "/nonexistent_root_dir/forbidden")
    response = client.get(f"{settings.API_V1_STR}/health/readiness")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["components"]["storage"]["status"] == "unhealthy"


def test_openapi_available(client: TestClient) -> None:
    response = client.get(f"{settings.API_V1_STR}/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert data["info"]["title"] == settings.PROJECT_NAME
    assert data["info"]["version"] == settings.VERSION




