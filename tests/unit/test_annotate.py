import io
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app

# ---------- Fixtures ----------


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan triggered (populates app.state.inference_service)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_image_bytes():
    """A valid 640x480 JPEG as bytes."""
    img = Image.new("RGB", (640, 480), color=(120, 180, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes():
    """A valid 320x240 PNG as bytes."""
    img = Image.new("RGB", (320, 240), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_payload():
    """A valid DetectionResponse-shaped dict matching a 640x480 image."""
    return {
        "detections": [
            {
                "bbox": {"x1": 15.0, "y1": 228.5, "x2": 808.2, "y2": 748.1},
                "confidence": 0.921,
                "class_id": 5,
                "class_name": "bus",
            },
            {
                "bbox": {"x1": 667.3, "y1": 389.6, "x2": 809.7, "y2": 878.5},
                "confidence": 0.893,
                "class_id": 0,
                "class_name": "person",
            },
        ],
        "count": 2,
        "image_width": 640,
        "image_height": 480,
    }


@pytest.fixture
def detections_json(valid_payload):
    return json.dumps(valid_payload)


@pytest.fixture(autouse=True)
def cleanup_saved_files(tmp_path, monkeypatch):
    """
    Redirect SAVE_DIR to a temp directory so tests don't pollute the real folder,
    and clean up afterwards.
    """
    monkeypatch.setattr(settings, "SAVE_DIR", tmp_path)
    yield
    # tmp_path is auto-removed by pytest, nothing else to do


# ---------- Happy Path ----------


def test_annotate_returns_200(client, sample_image_bytes, detections_json):
    response = client.post(
        "/annotate",
        files={"file": ("street.jpg", sample_image_bytes, "image/jpeg")},
        data={"detections_json": detections_json},
    )
    assert response.status_code == 200


def test_annotate_response_schema(client, sample_image_bytes, detections_json):
    response = client.post(
        "/annotate",
        files={"file": ("street.jpg", sample_image_bytes, "image/jpeg")},
        data={"detections_json": detections_json},
    )
    data = response.json()

    assert "result" in data
    assert "annotated_image_path" in data
    assert data["result"] == "Image annotated successfully"
    assert isinstance(data["annotated_image_path"], str)
    assert len(data["annotated_image_path"]) > 0


def test_annotate_saves_file_to_disk(client, sample_image_bytes, detections_json):
    response = client.post(
        "/annotate",
        files={"file": ("street.jpg", sample_image_bytes, "image/jpeg")},
        data={"detections_json": detections_json},
    )
    saved_path = response.json()["annotated_image_path"]

    assert os.path.exists(saved_path), f"Expected file at {saved_path}"
    assert os.path.getsize(saved_path) > 0

    # Verify it's a valid image
    with Image.open(saved_path) as img:
        assert img.size == (640, 480)


def test_annotate_filename_uses_uuid_prefix(
    client, sample_image_bytes, detections_json
):
    response = client.post(
        "/annotate",
        files={"file": ("street.jpg", sample_image_bytes, "image/jpeg")},
        data={"detections_json": detections_json},
    )
    saved_path = Path(response.json()["annotated_image_path"])
    name = saved_path.name

    # Expected: "<uuid>_street.jpg"
    assert name.endswith("_street.jpg")
    uuid_part = name.split("_")[0]
    # UUID4 hex is 32 chars (with or without dashes depending on str(uuid.uuid4()))
    assert len(uuid_part) >= 32


def test_annotate_accepts_png(client, sample_png_bytes):
    payload = {
        "detections": [
            {
                "bbox": {"x1": 10.0, "y1": 10.0, "x2": 100.0, "y2": 100.0},
                "confidence": 0.75,
                "class_id": 0,
                "class_name": "person",
            }
        ],
        "count": 1,
        "image_width": 320,
        "image_height": 240,
    }
    response = client.post(
        "/annotate",
        files={"file": ("test.png", sample_png_bytes, "image/png")},
        data={"detections_json": json.dumps(payload)},
    )
    assert response.status_code == 200


def test_annotate_with_empty_detections(client, sample_image_bytes):
    payload = {
        "detections": [],
        "count": 0,
        "image_width": 640,
        "image_height": 480,
    }
    response = client.post(
        "/annotate",
        files={"file": ("empty_dets.jpg", sample_image_bytes, "image/jpeg")},
        data={"detections_json": json.dumps(payload)},
    )
    # Depends on annotate_image behavior — adjust if you reject empty lists
    assert response.status_code == 200


# ---------- Validation / Error Cases ----------


def test_annotate_rejects_missing_file(client, detections_json):
    response = client.post(
        "/annotate",
        data={"detections_json": detections_json},
    )
    assert response.status_code == 422  # FastAPI required field


def test_annotate_rejects_missing_detections_json(client, sample_image_bytes):
    response = client.post(
        "/annotate",
        files={"file": ("street.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 422


def test_annotate_rejects_invalid_json(client, sample_image_bytes):
    response = client.post(
        "/annotate",
        files={"file": ("street.jpg", sample_image_bytes, "image/jpeg")},
        data={"detections_json": "not-json"},
    )
    # DetectionResponse.model_validate_json raises -> 500 unless you catch it.
    # If you want 422, wrap it in try/except HTTPException (see note below).
    assert response.status_code in (422, 500)


def test_annotate_rejects_missing_required_field_in_json(
    client, sample_image_bytes, valid_payload
):
    del valid_payload["count"]
    response = client.post(
        "/annotate",
        files={"file": ("street.jpg", sample_image_bytes, "image/jpeg")},
        data={"detections_json": json.dumps(valid_payload)},
    )
    assert response.status_code in (422, 500)


def test_annotate_rejects_confidence_out_of_range(
    client, sample_image_bytes, valid_payload
):
    valid_payload["detections"][0]["confidence"] = 1.5  # > 1.0
    response = client.post(
        "/annotate",
        files={"file": ("street.jpg", sample_image_bytes, "image/jpeg")},
        data={"detections_json": json.dumps(valid_payload)},
    )
    assert response.status_code in (422, 500)


def test_annotate_rejects_unsupported_file_type(
    client, sample_image_bytes, detections_json
):
    response = client.post(
        "/annotate",
        files={"file": ("test.gif", sample_image_bytes, "image/gif")},
        data={"detections_json": detections_json},
    )
    assert response.status_code == 415


def test_annotate_rejects_empty_file(client, detections_json):
    response = client.post(
        "/annotate",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        data={"detections_json": detections_json},
    )
    assert response.status_code == 400


def test_annotate_rejects_corrupt_image(client, detections_json):
    response = client.post(
        "/annotate",
        files={"file": ("corrupt.jpg", b"not-a-real-image", "image/jpeg")},
        data={"detections_json": detections_json},
    )
    assert response.status_code == 400
