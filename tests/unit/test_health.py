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


def test_openapi_available(client: TestClient) -> None:
    response = client.get(f"{settings.API_V1_STR}/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert data["info"]["title"] == settings.PROJECT_NAME
    assert data["info"]["version"] == settings.VERSION



