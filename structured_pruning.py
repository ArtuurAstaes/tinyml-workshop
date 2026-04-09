"""
structured_pruning.py — Structured (filter) pruning.

Loads the trained full-precision model, ranks Conv2d filters by their L1 norm
(sum of absolute weight values), removes the weakest ones, rebuilds a smaller
PrunedSimpleCNN with the surviving weights copied in, and fine-tunes to recover
accuracy.

Key ideas for the workshop:
  - Structured pruning removes entire filters, producing a physically smaller
    weight matrix. No sparse runtime needed — any hardware benefits immediately.
  - Contrast with unstructured pruning (pruning.py): that zeroes individual
    weights but keeps the same matrix shape, so the model stays the same size.
  - The tradeoff: removing whole filters is more destructive than zeroing
    scattered weights, so fine-tuning is more important here.
  - Ranking by L1 norm is the simplest heuristic: filters with small weights
    contribute little to the output and are safe to remove.

Usage:
    python structured_pruning.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SimpleCNN, PrunedSimpleCNN


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOAD_PATH = "simple_cnn.pth"
SAVE_PATH = "simple_cnn_structured_pruned.pth"
DATA_DIR = "./data"
N_FILTERS_TO_KEEP = 16   # Keep 16 of the 32 filters (50% reduction)
FINETUNE_EPOCHS = 5
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


def select_filters_by_l1(conv_weight, n_keep):
    """
    Rank filters by their L1 norm and return the indices of the top n_keep.
    conv_weight shape: (out_channels, in_channels, kH, kW)
    """
    l1_norms = conv_weight.abs().sum(dim=(1, 2, 3))  # One norm per filter
    _, top_indices = torch.topk(l1_norms, n_keep)
    return top_indices.sort().values  # Keep indices sorted for reproducibility


def copy_weights(source, pruned, filter_indices):
    """
    Copy the surviving filter weights from the full model into the pruned model.

    - Conv2d: copy selected output filters (and their biases)
    - Linear: the input features to fc correspond to flattened conv output,
      so we select the feature groups that correspond to the kept filters.
    """
    with torch.no_grad():
        # Conv layer: select kept filters along the output dimension
        pruned.conv.weight.copy_(source.conv.weight[filter_indices])
        if source.conv.bias is not None:
            pruned.conv.bias.copy_(source.conv.bias[filter_indices])

        # Linear layer: each kept filter produces 13*13 features after pooling.
        # We need to select the corresponding columns of the fc weight matrix.
        n_keep = len(filter_indices)
        spatial = 13 * 13
        fc_indices = torch.cat([
            torch.arange(i * spatial, (i + 1) * spatial) for i in filter_indices
        ])
        pruned.fc.weight.copy_(source.fc.weight[:, fc_indices])
        pruned.fc.bias.copy_(source.fc.bias)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    train_loader, test_loader = get_dataloaders()

    # Load baseline
    baseline = SimpleCNN().to(device)
    baseline.load_state_dict(torch.load(LOAD_PATH, map_location=device))
    baseline.eval()

    acc_before = evaluate(baseline, test_loader, device)
    print(f"Baseline accuracy:         {acc_before:.4f}")
    print(f"Baseline conv filters:     {baseline.conv.out_channels}")

    # Select the N_FILTERS_TO_KEEP strongest filters by L1 norm
    filter_indices = select_filters_by_l1(baseline.conv.weight.data, N_FILTERS_TO_KEEP)
    print(f"\nKeeping {N_FILTERS_TO_KEEP} of {baseline.conv.out_channels} filters "
          f"({100 * N_FILTERS_TO_KEEP // baseline.conv.out_channels}% of original)")

    # Build smaller model and copy surviving weights into it
    pruned = PrunedSimpleCNN(n_filters=N_FILTERS_TO_KEEP).to(device)
    copy_weights(baseline, pruned, filter_indices)

    acc_after_pruning = evaluate(pruned, test_loader, device)
    print(f"Accuracy after pruning (before fine-tuning): {acc_after_pruning:.4f}")

    # Fine-tune to recover accuracy
    optimizer = optim.Adam(pruned.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\nFine-tuning for {FINETUNE_EPOCHS} epochs...")
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        pruned.train()
        correct, total_loss = 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = pruned(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"  Epoch {epoch}/{FINETUNE_EPOCHS} | "
              f"loss: {total_loss / len(train_loader.dataset):.4f}, acc: {acc:.4f}")

    acc_final = evaluate(pruned, test_loader, device)
    print(f"\nFinal accuracy after fine-tuning: {acc_final:.4f}")

    torch.save(pruned.state_dict(), SAVE_PATH)
    print(f"Saved structured pruned model to '{SAVE_PATH}'")


if __name__ == "__main__":
    main()
