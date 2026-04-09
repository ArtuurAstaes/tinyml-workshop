"""
qat.py — Quantization-Aware Training (QAT).

Loads the trained full-precision model, inserts fake-quantization nodes, and
fine-tunes so the model learns to compensate for quantization error during
training rather than discovering it only at inference time (as PTQ does).

Key ideas for the workshop:
  - QAT generally outperforms PTQ in accuracy, at the cost of extra training.
  - Fake quantization: weights and activations are quantized and dequantized
    during the forward pass, so the model "feels" quantization error while
    gradients still flow in float.
  - Like PTQ, the final converted model only runs on CPU for Conv2d.

Usage:
    python qat.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# Suppress warnings about quantization and deprecation in ONNX export
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from model import CNN, QuantizedCNN


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOAD_PATH = Path("./models/cnn.pth")
SAVE_PATH = Path("./models/cnn_qat.pth")
DATA_DIR  = Path("./data")
EPOCHS    = 3  # Fine-tuning epochs — fewer needed since we start from a trained model
LEARNING_RATE = 1e-4




def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def main():
    # QAT must also run on CPU for the same reason as PTQ
    device = torch.device("cpu")

    train_loader, test_loader = get_dataloaders()

    # Load baseline
    baseline = CNN()
    baseline.load_state_dict(torch.load(LOAD_PATH, map_location=device))

    # Wrap and configure
    qat_model = QuantizedCNN(baseline)
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

    # Fuse Conv+ReLU
    torch.quantization.fuse_modules(qat_model.model, [["conv", "relu"]], inplace=True)

    # Prepare for QAT: inserts fake-quantization nodes into the graph
    torch.quantization.prepare_qat(qat_model, inplace=True)
    qat_model.train()

    optimizer = optim.Adam(qat_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Fine-tuning with QAT for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        qat_model.train()
        total_loss, correct = 0.0, 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = qat_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch}/{EPOCHS} | Train loss: {total_loss / len(train_loader.dataset):.4f}, acc: {100 * train_acc:.2f}%")

    # Convert fake-quantized model to a real quantized model
    qat_model.eval()
    torch.quantization.convert(qat_model, inplace=True)
    print("\nConversion to quantized model complete.")

    acc = evaluate(qat_model, test_loader)
    print(f"QAT model accuracy: {100 * acc:.2f}%")

    # Save the quantized model's state dict
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True) # Make sure save directory exists
    torch.save(qat_model.state_dict(), SAVE_PATH)
    print(f"Saved QAT model to '{SAVE_PATH}'")


if __name__ == "__main__":
    main()
