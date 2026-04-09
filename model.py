import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    A minimal CNN for 28x28 grayscale images (e.g. MNIST).

    Architecture:
        Conv2d(1, 32, 3) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(32*13*13, 10)

    Input:  (batch, 1, 28, 28)  -- NCHW format
    Output: (batch, 10)         -- raw logits
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 13 * 13, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

    def predict(self, x):
        """Returns class probabilities. Use this for human-readable inference output."""
        return torch.softmax(self.forward(x), dim=1)


class PrunedCNN(nn.Module):
    """
    A structurally pruned version of CNN with fewer Conv2d filters.

    Structured pruning physically removes entire filters, producing a smaller
    and faster model without needing a sparse-aware runtime.

    Args:
        n_filters: Number of Conv2d output filters to keep (default: 16, half of 32).

    Input:  (batch, 1, 28, 28)  -- NCHW format
    Output: (batch, 10)         -- raw logits
    """

    def __init__(self, n_filters=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n_filters * 13 * 13, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

    def predict(self, x):
        """Returns class probabilities. Use this for human-readable inference output."""
        return torch.softmax(self.forward(x), dim=1)
