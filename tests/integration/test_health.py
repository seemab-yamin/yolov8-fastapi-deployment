import os

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def test_health_returns_200():
    response = requests.get(f"{BASE_URL}/ping")
    assert response.status_code == 200


def test_health_payload():
    response = requests.get(f"{BASE_URL}/ping")
    data = response.json()
    assert "status" in data
    assert data["status"] == "running"
    assert "model_loaded" in data
    assert data["model_loaded"] is True
    assert "device" in data
    assert "timestamp" in data
