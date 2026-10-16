import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers startup/shutdown events,
    # which is what populates app.state.inference_service.
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_payload(client):
    response = client.get("/")
    data = response.json()
    assert "status" in data
    assert data["status"] == "running"
    assert "model_loaded" in data
    assert data["model_loaded"] is True
    assert "device" in data
    assert "timestamp" in data
