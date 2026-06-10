"""
tests/test_model.py
-------------------
Unit tests for the MNISTClassifier.

Tests:
    1. test_output_shape  — model produces [B, 10] output for any batch size.
    2. test_no_nan_output — output contains no NaN values (catches bad initialisations
                            or gradient issues in a quick smoke test).
    3. test_probabilities_sum_to_one — softmax probabilities sum to ~1.

Run with:  pytest tests/ -v
"""

import pytest
import torch
import torch.nn.functional as F
from src.model import MNISTClassifier, get_model


@pytest.fixture
def model():
    """Fresh model in eval mode (no trained weights needed for shape/NaN tests)."""
    m = get_model(num_classes=10, dropout_rate=0.25)
    m.eval()
    return m


@pytest.fixture
def sample_batch():
    """Random batch simulating normalised MNIST images."""
    torch.manual_seed(0)
    return torch.randn(8, 1, 28, 28)   # batch of 8 images


# ───────────────────────────────────────────────
# Test 1: Output Shape
# ───────────────────────────────────────────────
def test_output_shape(model, sample_batch):
    """
    The model must output a tensor of shape [B, 10] regardless of batch size.
    This confirms that Conv2d dimensions and FC layer sizes are compatible.
    """
    with torch.no_grad():
        output = model(sample_batch)

    assert output.shape == (8, 10), (
        f"Expected output shape (8, 10), got {tuple(output.shape)}"
    )


# ───────────────────────────────────────────────
# Test 2: No NaN in Output
# ───────────────────────────────────────────────
def test_no_nan_output(model, sample_batch):
    """
    Model output must not contain any NaN or Inf values.
    NaNs indicate numerical instability (e.g. bad initialisations, overflow).
    """
    with torch.no_grad():
        output = model(sample_batch)

    assert not torch.isnan(output).any(), "Model output contains NaN values"
    assert not torch.isinf(output).any(), "Model output contains Inf values"


# ───────────────────────────────────────────────
# Test 3: Softmax Probabilities Sum to 1
# ───────────────────────────────────────────────
def test_probabilities_sum_to_one(model, sample_batch):
    """
    After softmax, probabilities for each sample must sum to 1 (within float tolerance).
    """
    with torch.no_grad():
        logits = model(sample_batch)
        probs = F.softmax(logits, dim=1)

    row_sums = probs.sum(dim=1)   # [B]
    assert torch.allclose(row_sums, torch.ones(8), atol=1e-5), (
        f"Probability rows do not sum to 1: {row_sums}"
    )
