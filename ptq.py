"""
ptq.py — Post-Training Quantization (PTQ).

Loads the trained full-precision model, calibrates activation ranges on a
small subset of MNIST, and produces a statically quantized model.

Key ideas for the workshop:
  - No retraining required — PTQ is fast and simple.
  - Requires a calibration dataset to measure activation ranges.
  - May lose a small amount of accuracy compared to the baseline.
  - PyTorch static quantization only runs on CPU for Conv2d models.

Usage:
    python ptq.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# Suppress warnings about quantization and deprecation in ONNX export
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from model import CNN, QuantizedCNN
from utils.quantization import setup_quantization_engine


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOAD_PATH = Path("./models/cnn.pth")
SAVE_PATH = Path("./models/cnn_ptq.pth")
DATA_DIR  = Path("./data")
CALIBRATION_BATCHES = 10  # How many batches to use for calibration

# Detect and setup the best engine for this machine
Q_ENGINE = setup_quantization_engine()


def get_calibration_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=False)


def evaluate(model, loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    return correct / len(loader.dataset)


def main():
    # PTQ must run on CPU — PyTorch does not support CUDA quantized Conv2d
    device = torch.device("cpu")

    # Load baseline model
    baseline = CNN()
    baseline.load_state_dict(torch.load(LOAD_PATH, map_location=device))
    baseline.eval()

    # Wrap with quant/dequant stubs
    quant_model = QuantizedCNN(baseline)
    quant_model.eval()

    # Set quantization config using the dynamically selected engine
    quant_model.qconfig = torch.quantization.get_default_qconfig(Q_ENGINE)

    # Fuse Conv+ReLU into a single op before quantizing (improves accuracy and speed)
    # We fuse inside the inner model, targeting the conv and relu sequence
    torch.quantization.fuse_modules(quant_model.model, [["conv", "relu"]], inplace=True)

    # Prepare: insert observers that will collect activation statistics
    torch.quantization.prepare(quant_model, inplace=True)

    # Calibrate: run batches through the model so observers collect range data
    print(f"Calibrating with engine '{Q_ENGINE}' on {CALIBRATION_BATCHES} batches...")
    loader = get_calibration_loader()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= CALIBRATION_BATCHES:
                break
            quant_model(images)

    # Convert: replace float ops with quantized equivalents
    torch.quantization.convert(quant_model, inplace=True)
    print("Quantization complete.")

    # Evaluate
    acc = evaluate(quant_model, loader, device)
    print(f"PTQ model accuracy: {100 * acc:.2f}%")

    # Save the quantized model's state dict
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True) # Make sure save directory exists
    torch.save(quant_model.state_dict(), SAVE_PATH)
    print(f"Saved PTQ model to '{SAVE_PATH}'")

    # Report size reduction
    import os
    baseline_size = os.path.getsize(LOAD_PATH) / 1024
    quant_size = os.path.getsize(SAVE_PATH) / 1024
    print(f"\nBaseline model size: {baseline_size:.1f} KB")
    print(f"PTQ model size:      {quant_size:.1f} KB")


if __name__ == "__main__":
    main()
