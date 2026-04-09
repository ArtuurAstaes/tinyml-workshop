"""
train.py — Full-precision training on MNIST.

Trains CNN and saves weights to models/cnn.pth.
This is the baseline model that all other scripts (PTQ, QAT, pruning) build on.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from model import CNN


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
SAVE_PATH = Path("models/cnn.pth")
DATA_DIR  = Path("./data")


def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, test_loader = get_dataloaders()

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # CrossEntropyLoss expects raw logits (no softmax) — the model's forward()
    # returns exactly that, so this is correct.
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining CNN for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f}, acc: {100 * train_acc:.2f}% | "
              f"Test loss: {test_loss:.4f}, acc: {100 * test_acc:.2f}%")

    SAVE_PATH.parent.mkdir(exist_ok=True)  # Ensure 'models' directory exists
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nModel saved to '{SAVE_PATH}'")


if __name__ == "__main__":
    main()