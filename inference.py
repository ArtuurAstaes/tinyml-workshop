"""
inference.py — Run and compare inference across all model variants.

Loads the full-precision, PTQ, QAT, unstructured pruned, and structured pruned
models, runs them on a small batch of MNIST test samples, and prints predictions,
accuracy, and timing.

Usage:
    python inference.py
"""

import time
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# Suppress warnings about quantization and deprecation in ONNX export
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from model import CNN, QuantizedCNN, PrunedCNN
from utils.quantization import setup_quantization_engine


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR  = Path("./data")
N_SAMPLES = 1000
N_FILTERS_KEPT = 16   # Must match the value used in structured_pruning.py

# Detect and setup the best engine for this machine
Q_ENGINE = setup_quantization_engine()


def get_test_loader(n_samples):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    subset = torch.utils.data.Subset(dataset, range(n_samples))
    return DataLoader(subset, batch_size=64, shuffle=False)


def run_inference(model, loader, device):
    """Returns (accuracy, inference_time_ms)."""
    model.eval()
    correct = 0
    start = time.perf_counter()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    elapsed_ms = (time.perf_counter() - start) * 1000
    acc = correct / len(loader.dataset)
    return acc, elapsed_ms


def model_size_kb(path):
    return os.path.getsize(path) / 1024


def load_baseline(path, device):
    model = CNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_quantized(path, device):
    """
    Quantized models must be reconstructed with the same wrapper and qconfig
    used during PTQ/QAT before loading the state dict.
    """
    baseline = CNN()
    wrapper = QuantizedCNN(baseline)
    wrapper.qconfig = torch.quantization.get_default_qconfig(Q_ENGINE)
    torch.quantization.fuse_modules(wrapper.model, [["conv", "relu"]], inplace=True)
    torch.quantization.prepare(wrapper, inplace=True)
    torch.quantization.convert(wrapper, inplace=True)
    wrapper.load_state_dict(torch.load(path, map_location=device))
    wrapper.eval()
    return wrapper


def load_structured_pruned(path, device):
    model = PrunedCNN(n_filters=N_FILTERS_KEPT).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main():
    # Quantized models only run on CPU
    device = torch.device("cpu")

    loader = get_test_loader(N_SAMPLES)

    variants = [
        ("Baseline (float32)",          Path("./models/cnn.pth"),                     "float"),
        ("PTQ (int8)",                  Path("./models/cnn_ptq.pth"),                 "quantized"),
        ("QAT (int8)",                  Path("./models/cnn_qat.pth"),                 "quantized"),
        ("Unstructured pruned",         Path("./models/cnn_unstructured_pruned.pth"), "float"),
        ("Structured pruned (float32)", Path("./models/cnn_structured_pruned.pth"),   "structured"),
    ]

    print(f"\nRunning inference on {N_SAMPLES} MNIST test samples (CPU)")
    print(f"Quantization engine: {Q_ENGINE}\n")
    print(f"{'Model':<30} {'Accuracy':>10} {'Time (ms)':>12} {'Size (KB)':>12}")
    print("-" * 67)

    for name, path, kind in variants:
        if not os.path.exists(path):
            print(f"{name:<30} {'(not found)':>10}")
            continue

        if kind == "float":
            model = load_baseline(path, device)
        elif kind == "quantized":
            model = load_quantized(path, device)
        else:
            model = load_structured_pruned(path, device)

        acc, elapsed = run_inference(model, loader, device)
        size = model_size_kb(path)

        print(f"{name:<30} {acc:>10.4f} {elapsed:>11.1f}ms {size:>10.1f} KB")

    print()


if __name__ == "__main__":
    main()
