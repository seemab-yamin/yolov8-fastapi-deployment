import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app

# ---------- Fixtures ----------


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan triggered."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def temp_save_dir(tmp_path, monkeypatch):
    """
    Redirect SAVE_DIR to a temp directory so tests don't touch the real folder.
    Keep it a Path (not str) so `dir_path / filename` works.
    """
    monkeypatch.setattr(settings, "SAVE_DIR", tmp_path)
    yield tmp_path
    # tmp_path is auto-cleaned by pytest


@pytest.fixture
def saved_jpeg(temp_save_dir: Path):
    """Create a valid JPEG file in the temp SAVE_DIR and return (filename, bytes)."""
    filename = "annotated_test.jpg"
    file_path = temp_save_dir / filename
    img = Image.new("RGB", (100, 80), color=(50, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()
    file_path.write_bytes(data)
    return filename, data


# ---------- Happy Path ----------


def test_download_returns_200(client, saved_jpeg):
    filename, _ = saved_jpeg
    response = client.get(f"/downloads/{filename}")
    assert response.status_code == 200


def test_download_returns_correct_content_type(client, saved_jpeg):
    filename, _ = saved_jpeg
    response = client.get(f"/downloads/{filename}")
    assert response.headers["content-type"] == "image/jpeg"


def test_download_returns_exact_bytes(client, saved_jpeg):
    filename, original_bytes = saved_jpeg
    response = client.get(f"/downloads/{filename}")
    assert response.content == original_bytes


def test_download_sets_content_disposition(client, saved_jpeg):
    filename, _ = saved_jpeg
    response = client.get(f"/downloads/{filename}")
    disposition = response.headers.get("content-disposition", "")
    # FileResponse sets: attachment; filename="..."
    assert "attachment" in disposition
    assert filename in disposition


def test_download_content_length_matches(client, saved_jpeg):
    filename, original_bytes = saved_jpeg
    response = client.get(f"/downloads/{filename}")
    assert int(response.headers["content-length"]) == len(original_bytes)


# ---------- Error Cases ----------


def test_download_returns_404_for_missing_file(client):
    response = client.get("/downloads/does_not_exist.jpg")
    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"


def test_download_returns_404_for_directory(client, temp_save_dir: Path):
    # Create a subdirectory whose name looks like a file
    subdir = temp_save_dir / "subdir.jpg"
    subdir.mkdir()
    response = client.get("/downloads/subdir.jpg")
    # os.path.exists() returns True for dirs, but FileResponse will fail to read it.
    # Depending on how FastAPI/Starlette handles this, it may be 200 with empty body,
    # 404, or 500. Assert it does NOT silently return a corrupt image.
    assert response.status_code in (404, 500)


def test_download_rejects_path_traversal(client, temp_save_dir: Path):
    """
    Attempt to escape SAVE_DIR via ../ in the filename.
    FastAPI routes don't match slashes in {filename}, so '/downloads/../secret'
    hits a different route or 404. Verify no file leakage.
    """
    # Create a file OUTSIDE SAVE_DIR
    outside = temp_save_dir.parent / "secret.txt"
    outside.write_text("top-secret")

    response = client.get("/downloads/../secret.txt")
    # Path with ../ won't match the route (404) or will be rejected
    assert response.status_code in (404, 422)
    assert b"top-secret" not in response.content


def test_download_rejects_encoded_path_traversal(client, temp_save_dir: Path):
    """URL-encoded traversal should also be blocked."""
    outside = temp_save_dir.parent / "secret2.txt"
    outside.write_text("secret")

    response = client.get("/downloads/%2e%2e%2fsecret2.txt")
    # Depending on router matching this is either 404 or a safe miss
    assert response.status_code in (404, 422)
    assert b"secret" not in response.content or response.status_code != 200


def test_download_empty_filename(client):
    """Route requires a non-empty filename segment."""
    response = client.get("/downloads/")
    # No filename -> route doesn't match -> 404
    assert response.status_code == 404


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Skip case sensitivity test on macOS"
)
def test_download_case_sensitivity(client, saved_jpeg):
    """Filesystem is case-sensitive on macOS/Linux — different case = 404."""
    filename, _ = saved_jpeg
    wrong_case = filename.upper()
    response = client.get(f"/downloads/{wrong_case}")
    assert response.status_code == 404
