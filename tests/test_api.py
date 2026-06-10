"""
tests/test_api.py
-----------------
Tests the FastAPI /predict endpoint using TestClient (no real HTTP server).
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from src.app import app


def make_png_bytes(mode: str = "L", size: tuple = (28, 28)) -> bytes:
    """Create a random greyscale PNG in memory."""
    arr = np.random.randint(0, 256, size, dtype=np.uint8)
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """TestClient loads the ONNX model at startup via lifespan."""
    with TestClient(app) as c:
        yield c


# ── Test 1: Health endpoint ──────────────────────────────────────────
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_loaded"] is True


# ── Test 2: Status 200 and valid JSON structure ──────────────────────
def test_predict_status_200(client):
    """POST to /predict must return HTTP 200."""
    image_bytes = make_png_bytes()
    resp = client.post(
        "/predict",
        files={"file": ("digit.png", image_bytes, "image/png")},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


# ── Test 3: Valid probability output ────────────────────────────────
def test_predict_valid_probability(client):
    """
    /predict must return:
      - predicted_digit: int in [0, 9]
      - confidence: float in (0, 1]
      - probabilities: list of 10 floats that sum to ~1
    """
    image_bytes = make_png_bytes()
    resp = client.post(
        "/predict",
        files={"file": ("digit.png", image_bytes, "image/png")},
    )
    data = resp.json()

    # Digit range
    assert 0 <= data["predicted_digit"] <= 9, "predicted_digit out of range"

    # Confidence range
    assert 0.0 < data["confidence"] <= 1.0, "confidence out of range"

    # Probabilities
    probs = data["probabilities"]
    assert len(probs) == 10, f"Expected 10 probabilities, got {len(probs)}"
    assert abs(sum(probs) - 1.0) < 1e-4, f"Probabilities sum to {sum(probs)}, not 1"
    assert all(p >= 0 for p in probs), "Negative probability found"


# ── Test 4: Invalid file type returns 400 ───────────────────────────
def test_predict_invalid_file_type(client):
    """Uploading a non-image file must return HTTP 400."""
    resp = client.post(
        "/predict",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400
