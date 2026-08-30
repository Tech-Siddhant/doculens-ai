from fastapi.testclient import TestClient
from app.api.main import app
from app.core.config import settings

client = TestClient(app)


def test_health_endpoint():
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

