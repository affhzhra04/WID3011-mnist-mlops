"""
src/app.py
----------
FastAPI REST API for MNIST digit classification.

Endpoints:
    GET  /          — Health check
    GET  /health    — Model load status
    POST /predict   — Upload image, receive digit + confidence

The ONNX model is loaded once at startup (lifespan event) and reused
for all requests — avoids per-request I/O overhead.

Run locally:
    uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import onnxruntime as ort
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# ── Configuration ────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ── Pydantic Response Schema ─────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: List[float]


# ── Global model state ───────────────────────────────────────────────
# Using a dict to allow mutation inside lifespan context
model_state = {"session": None, "model_loaded": False}


# ── Lifespan: load model once at startup ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX model on startup; clean up on shutdown."""
    cfg = load_config()
    onnx_path = cfg["paths"]["model_save_onnx"]

    if not os.path.exists(onnx_path):
        raise RuntimeError(f"ONNX model not found at {onnx_path}. Run export first.")

    # Use CPU for portability; swap to CUDAExecutionProvider for GPU
    model_state["session"] = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )
    model_state["model_loaded"] = True
    print(f"[app] ONNX model loaded from {onnx_path}")

    yield  # application runs here

    # Shutdown cleanup (optional)
    model_state["session"] = None
    model_state["model_loaded"] = False
    print("[app] Model unloaded.")


# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title="MNIST Digit Classifier API",
    description="WID3011 Lab — Classify a 28×28 greyscale image of a handwritten digit.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helper ────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes into a normalised float32 array.

    Steps:
        1. Open with PIL and convert to greyscale ('L' mode).
        2. Resize to 28×28.
        3. Normalise with MNIST mean=0.1307, std=0.3081.
        4. Shape: [1, 1, 28, 28] as float32.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0       # [28, 28], range [0, 1]
    arr = (arr - 0.1307) / 0.3081                        # normalise
    arr = arr[np.newaxis, np.newaxis, :, :]              # [1, 1, 28, 28]
    return arr


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {"message": "MNIST Classifier API is running", "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok" if model_state["model_loaded"] else "model_not_loaded",
        "model_loaded": model_state["model_loaded"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="PNG/JPEG 28×28 greyscale digit image")):
    """
    Upload a digit image and receive:
    - **predicted_digit** — integer 0–9
    - **confidence** — softmax probability of the predicted class
    - **probabilities** — full softmax distribution over all 10 classes
    """
    if not model_state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate content type
    allowed = {"image/png", "image/jpeg", "image/bmp", "image/tiff"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use: {allowed}"
        )

    try:
        image_bytes = await file.read()
        input_array = preprocess(image_bytes)           # [1, 1, 28, 28]

        session = model_state["session"]
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: input_array})[0][0]  # [10]

        probs = softmax(logits)                         # [10]
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])

        return PredictionResponse(
            predicted_digit=predicted,
            confidence=round(confidence, 6),
            probabilities=[round(float(p), 6) for p in probs.tolist()],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
