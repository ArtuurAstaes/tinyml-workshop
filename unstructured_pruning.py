"""
pruning.py — Magnitude-based unstructured pruning.

Loads the trained full-precision model, prunes a fraction of the smallest
weights in the conv and linear layers, fine-tunes briefly to recover accuracy,
and makes the pruning permanent.

Key ideas for the workshop:
  - Unstructured pruning zeroes out individual weights (not entire filters).
  - PyTorch pruning works via masks: it doesn't physically remove weights,
    it multiplies them by a binary mask. The model is the same size in memory
    until you call make_permanent() and re-export.
  - After make_permanent(), zeroed weights are truly gone from the state dict,
    but you need a sparse-aware runtime to actually see a speedup.
  - Sparsity (fraction pruned) vs. accuracy is the key tradeoff to discuss.

Usage:
    python pruning.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

from model import CNN


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOAD_PATH = Path("./models/cnn.pth")
SAVE_PATH = Path("./models/cnn_unstructured_pruned.pth")
DATA_DIR  = Path("./data")
PRUNING_AMOUNT = 0.5   # Fraction of weights to prune (50%)
FINETUNE_EPOCHS = 3
LEARNING_RATE = 1e-4


def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)

    return (DataLoader(train_dataset, batch_size=64, shuffle=True),
            DataLoader(test_dataset, batch_size=64, shuffle=False))


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


def count_sparsity(model):
    """Returns the fraction of zero weights across all pruned parameters."""
    total, zeros = 0, 0
    for _, param in model.named_parameters():
        total += param.numel()
        zeros += (param == 0).sum().item()
    return zeros / total


def make_permanent(model):
    """
    Remove pruning masks and make zeroed weights permanent.
    After this, the module no longer has _mask and _orig buffers —
    just a regular weight tensor with zeros baked in.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass  # Module wasn't pruned


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    train_loader, test_loader = get_dataloaders()

    model = CNN().to(device)
    model.load_state_dict(torch.load(LOAD_PATH, map_location=device))

    # Evaluate before pruning
    acc_before = evaluate(model, test_loader, device)
    print(f"Accuracy before pruning: {100 * acc_before:.2f}%")

    # Apply magnitude-based unstructured pruning to conv and fc layers.
    # L1 unstructured pruning removes weights with the smallest absolute values.
    prune.l1_unstructured(model.conv, name="weight", amount=PRUNING_AMOUNT)
    prune.l1_unstructured(model.fc, name="weight", amount=PRUNING_AMOUNT)

    sparsity = count_sparsity(model)
    acc_after_pruning = evaluate(model, test_loader, device)
    print(f"\nAfter pruning (before fine-tuning):")
    print(f"  Sparsity:  {sparsity:.2%}")
    print(f"  Accuracy:  {100 * acc_after_pruning:.2f}%")

    # Fine-tune to recover accuracy lost from pruning.
    # The masks are kept active during fine-tuning, so pruned weights stay zero.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\nFine-tuning for {FINETUNE_EPOCHS} epochs...")
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        correct, total_loss = 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"  Epoch {epoch}/{FINETUNE_EPOCHS} | loss: {total_loss / len(train_loader.dataset):.4f}, acc: {100 * acc:.2f}%")

    acc_after_finetuning = evaluate(model, test_loader, device)
    print(f"\nAfter fine-tuning:")
    print(f"  Accuracy: {100 * acc_after_finetuning:.2f}%")

    # Make pruning permanent: removes masks, bakes zeros into weights.
    # Important teaching point: the .pth file will be smaller after this,
    # but you need a sparse runtime to get actual inference speedups.
    make_permanent(model)
    print(f"\nPruning made permanent. Final sparsity: {count_sparsity(model):.2%}")

    # Save the pruned model's state dict
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True) # Make sure save directory exists
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved pruned model to '{SAVE_PATH}'")


if __name__ == "__main__":
    main()
