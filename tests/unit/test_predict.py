import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

# ---------- Fixtures ----------


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan triggered (populates app.state.inference_service)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_image_bytes():
    """Create a valid in-memory JPEG image for testing."""
    img = Image.new("RGB", (640, 480), color=(120, 180, 240))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def sample_png_bytes():
    """Create a valid in-memory PNG image."""
    img = Image.new("RGB", (320, 240), color=(200, 100, 50))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------- Happy Path ----------


def test_predict_returns_200(client, sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200


def test_predict_response_schema(client, sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    data = response.json()

    # Top-level keys
    assert "detections" in data
    assert "count" in data
    assert "image_width" in data
    assert "image_height" in data

    assert isinstance(data["detections"], list)
    assert data["count"] == len(data["detections"])
    assert data["image_width"] == 640
    assert data["image_height"] == 480


def test_predict_detection_item_schema(client, sample_image_bytes):
    """Each detection must have bbox, confidence, class_id, class_name."""
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    data = response.json()

    for det in data["detections"]:
        assert "bbox" in det
        assert "confidence" in det
        assert "class_id" in det
        assert "class_name" in det

        bbox = det["bbox"]
        assert set(bbox.keys()) == {"x1", "y1", "x2", "y2"}
        assert all(isinstance(v, (int, float)) for v in bbox.values())
        assert bbox["x2"] > bbox["x1"]
        assert bbox["y2"] > bbox["y1"]

        assert 0.0 <= det["confidence"] <= 1.0
        assert isinstance(det["class_id"], int)
        assert isinstance(det["class_name"], str)


def test_predict_accepts_png(client, sample_png_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.png", sample_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["image_width"] == 320
    assert response.json()["image_height"] == 240


# ---------- Validation / Error Cases ----------


def test_predict_rejects_missing_file(client):
    response = client.post("/predict")
    assert response.status_code == 422  # FastAPI validation error


def test_predict_rejects_unsupported_type(client, sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.gif", sample_image_bytes, "image/gif")},
    )
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_predict_rejects_empty_file(client):
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_rejects_corrupt_image(client):
    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", b"not-a-real-image", "image/jpeg")},
    )
    assert response.status_code == 400
    assert (
        "invalid" in response.json()["detail"].lower()
        or "corrupted" in response.json()["detail"].lower()
    )


def test_predict_rejects_too_small_image(client):
    tiny = Image.new("RGB", (10, 10), color=(0, 0, 0))
    buffer = io.BytesIO()
    tiny.save(buffer, format="JPEG")
    response = client.post(
        "/predict",
        files={"file": ("tiny.jpg", buffer.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 400
    assert "too small" in response.json()["detail"].lower()


def test_predict_rejects_oversized_file(client):
    """Simulate a >10MB payload without actually generating a huge image."""
    big_payload = b"\xff\xd8\xff" + b"0" * (11 * 1024 * 1024)
    response = client.post(
        "/predict",
        files={"file": ("big.jpg", big_payload, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()
