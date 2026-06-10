"""
src/export_onnx.py
------------------
Exports the trained MNISTClassifier to ONNX and validates numerical equivalence.

The script:
    1. Loads the .pt checkpoint from paths.model_save_pt
    2. Exports to ONNX using torch.onnx.export()
    3. Validates the ONNX graph structure
    4. Runs a comparison test (PyTorch vs ONNX Runtime outputs)

Usage:
    python -m src.export_onnx
"""

import os
import yaml
import numpy as np
import torch
import onnx
import onnxruntime as ort

from src.model import get_model


def export_to_onnx(config_path: str = "configs/config.yaml"):
    """Export trained model to ONNX and validate."""

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    pt_path   = cfg["paths"]["model_save_pt"]
    onnx_path = cfg["paths"]["model_save_onnx"]

    # ── 1. Load trained checkpoint ──────────────────────────────────
    checkpoint = torch.load(pt_path, map_location="cpu")
    model = get_model(
        num_classes=cfg["model"]["num_classes"],
        dropout_rate=cfg["model"]["dropout_rate"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()   # CRITICAL: must be eval() before export
    print(f"[export] Loaded checkpoint from {pt_path}")

    # ── 2. Create a dummy input (defines the input signature) ────────
    dummy_input = torch.randn(1, 1, 28, 28)   # [batch=1, C=1, H=28, W=28]

    # ── 3. Export to ONNX ───────────────────────────────────────────
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,          # embed weights in the .onnx file
        opset_version=17,            # ONNX opset — 17 is broadly supported
        do_constant_folding=True,    # optimise constant expressions
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch_size"},   # allow variable batch size
            "logits": {0: "batch_size"},
        },
    )
    print(f"[export] ONNX model saved → {onnx_path}")

    # ── 4. Structural validation ─────────────────────────────────────
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)          # raises if graph is invalid
    print("[export] ONNX graph structure: VALID")

    # ── 5. Numerical equivalence check ──────────────────────────────
    torch.manual_seed(42)
    test_input = torch.randn(4, 1, 28, 28)   # batch of 4

    # PyTorch reference output
    with torch.no_grad():
        pt_logits = model(test_input).numpy()

    # ONNX Runtime output
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {sess.get_inputs()[0].name: test_input.numpy()}
    ort_logits = sess.run(None, ort_inputs)[0]

    # Compare — allow floating-point tolerance of 1e-5
    max_diff = np.max(np.abs(pt_logits - ort_logits))
    print(f"[export] Max absolute difference (PyTorch vs ONNX): {max_diff:.2e}")

    TOLERANCE = 1e-4
    if max_diff < TOLERANCE:
        print(f"[export] ✅ PASS — outputs match within tolerance ({TOLERANCE})")
    else:
        raise ValueError(f"[export] ❌ FAIL — outputs diverge: max_diff={max_diff}")

    return onnx_path


if __name__ == "__main__":
    export_to_onnx()
