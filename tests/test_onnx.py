"""
tests/test_onnx.py
------------------
Validates ONNX export correctness.
"""

import pytest
import numpy as np
import torch
import onnx
import onnxruntime as ort
import yaml
from src.model import get_model


ONNX_PATH = "models/mnist_model.onnx"
CONFIG_PATH = "configs/config.yaml"


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_onnx_model_is_valid():
    """onnx.checker.check_model must not raise."""
    model_proto = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model_proto)   # raises onnx.checker.ValidationError on failure


def test_onnx_output_matches_pytorch(cfg):
    """
    PyTorch and ONNX Runtime must produce identical logits within 1e-4 tolerance
    for the same random input.
    """
    import torch
    checkpoint = torch.load(cfg["paths"]["model_save_pt"], map_location="cpu")
    model = get_model(num_classes=cfg["model"]["num_classes"],
                      dropout_rate=cfg["model"]["dropout_rate"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    torch.manual_seed(99)
    x = torch.randn(2, 1, 28, 28)

    with torch.no_grad():
        pt_out = model(x).numpy()

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]

    assert np.allclose(pt_out, ort_out, atol=1e-4), (
        f"Max diff: {np.max(np.abs(pt_out - ort_out)):.2e}"
    )
