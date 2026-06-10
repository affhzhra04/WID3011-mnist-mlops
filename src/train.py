"""
src/train.py
------------
Trains the MNISTClassifier using hyperparameters from configs/config.yaml.
Saves the trained model weights to paths.model_save_pt.

Usage (from project root):
    python -m src.train
    python -m src.train --config configs/config.yaml
"""

import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.model import get_model


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration from disk."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_cfg: str) -> torch.device:
    """Resolve device string. 'auto' → CUDA if available, else CPU."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def get_data_loaders(cfg: dict):
    """Download MNIST and return (train_loader, val_loader, test_loader)."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # Standard MNIST normalisation (mean=0.1307, std=0.3081)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = datasets.MNIST(
        root=data_cfg["data_dir"],
        train=True,
        download=data_cfg["download"],
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_cfg["data_dir"],
        train=False,
        download=data_cfg["download"],
        transform=transform,
    )

    # Split training into train + validation
    val_size = int(len(full_train) * data_cfg["val_split"])
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(train_cfg["seed"])
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=train_cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=train_cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimiser, device):
    """One full pass over the training data. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train(config_path: str = "configs/config.yaml"):
    """Main training entry point."""
    cfg = load_config(config_path)
    torch.manual_seed(cfg["training"]["seed"])

    device = get_device(cfg["training"]["device"])
    print(f"[train] Using device: {device}")

    # Build model
    model = get_model(
        num_classes=cfg["model"]["num_classes"],
        dropout_rate=cfg["model"]["dropout_rate"],
    ).to(device)
    print(f"[train] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(cfg)
    print(f"[train] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Optimisation
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=2, gamma=0.5)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{cfg['training']['epochs']}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n[train] Final Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Save model
    save_path = cfg["paths"]["model_save_pt"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "test_accuracy": test_acc,
    }, save_path)
    print(f"[train] Model saved → {save_path}")

    return model, cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
