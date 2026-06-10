"""
src/predict.py
--------------
Inference utilities.
Provides preprocess_image() and predict_pytorch() for use by the FastAPI app
and for standalone testing.
"""

import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Match the exact normalisation used during training
PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convert raw image bytes to a normalised [1, 1, 28, 28] tensor.

    Args:
        image_bytes: Raw bytes of a PNG/JPEG/BMP image.

    Returns:
        Tensor of shape [1, 1, 28, 28] ready for model inference.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # force greyscale
    tensor = PREPROCESS(image)          # [1, 28, 28]
    return tensor.unsqueeze(0)          # [1, 1, 28, 28]


@torch.no_grad()
def predict_pytorch(model, tensor: torch.Tensor, device: torch.device) -> dict:
    """
    Run inference with a PyTorch model.

    Args:
        model: Trained MNISTClassifier in eval mode.
        tensor: [1, 1, 28, 28] input tensor.
        device: Target device.

    Returns:
        dict with keys:
            predicted_digit (int): The argmax class.
            confidence (float): Softmax probability of the predicted class.
            probabilities (list[float]): Softmax over all 10 classes.
    """
    model.eval()
    tensor = tensor.to(device)
    logits = model(tensor)               # [1, 10]
    probs  = F.softmax(logits, dim=1)   # [1, 10]
    predicted = int(probs.argmax(dim=1).item())
    confidence = float(probs[0, predicted].item())
    return {
        "predicted_digit": predicted,
        "confidence": round(confidence, 6),
        "probabilities": [round(float(p), 6) for p in probs[0].tolist()],
    }
