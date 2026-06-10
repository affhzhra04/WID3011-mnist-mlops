"""
src/model.py
------------
Defines the CNN architecture for MNIST digit classification.

Architecture overview:
  Input: [B, 1, 28, 28] (batch of greyscale 28x28 images)
  Conv Block 1: Conv2d(1→32, 3x3) → ReLU → Conv2d(32→64, 3x3) → ReLU → MaxPool(2x2) → Dropout
  Conv Block 2: Conv2d(64→128, 3x3, pad=1) → ReLU → MaxPool(2x2) → Dropout
  FC Layers:    Flatten → Linear(128*6*6→256) → ReLU → Dropout → Linear(256→10)
  Output: [B, 10]  (raw logits; use softmax externally for probabilities)
"""

import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.

    Args:
        num_classes (int): Number of output classes. Default: 10.
        dropout_rate (float): Dropout probability. Default: 0.25.
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25):
        super(MNISTClassifier, self).__init__()

        # --- Convolutional Feature Extractor ---
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
        )

        # --- Fully-Connected Classifier Head ---
        # After conv_block1: 28→26→24→12 (MaxPool2d/2)
        # After conv_block2: 12→12→6  (MaxPool2d/2)
        # Flattened: 128 * 6 * 6 = 4608
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, 1, 28, 28]

        Returns:
            Logits tensor of shape [B, 10]
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x


def get_model(num_classes: int = 10, dropout_rate: float = 0.25) -> MNISTClassifier:
    """Factory function — creates a fresh MNISTClassifier."""
    return MNISTClassifier(num_classes=num_classes, dropout_rate=dropout_rate)
