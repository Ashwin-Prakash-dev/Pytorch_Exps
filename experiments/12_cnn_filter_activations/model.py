"""The CNN and data pipeline shared by every script in this experiment.

Standard 10-way MNIST digit classification -- not binary, unlike
experiments 04/08. This experiment isn't about classification accuracy at
all; it's about what happens *inside* the network while it classifies, so
the model just needs to be a reasonably accurate, structurally simple CNN
with a small, inspectable number of filters.

Architecture: two conv blocks, chosen specifically so experiment 5's
"early vs. later layer" comparison has exactly two layers to compare.

    Conv2d(1->8, 3x3, pad=1) -> ReLU -> MaxPool(2)   # 28x28 -> 14x14, 8 filters
    Conv2d(8->16, 3x3, pad=1) -> ReLU -> MaxPool(2)  # 14x14 -> 7x7, 16 filters
    Flatten -> Linear(16*7*7 -> 32) -> ReLU -> Linear(32 -> 10)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from utils import get_device, set_seed  # noqa: E402

OUTPUTS = HERE / "outputs"
DATA_ROOT = REPO_ROOT / "data" / "raw"
CHECKPOINT = OUTPUTS / "model.pt"


class CNN(nn.Module):
    """Two named conv layers (`conv1`, `conv2`) so activation hooks can be
    attached by name from every other script in this experiment."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


def make_loaders(batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_set, test_set


def load_trained_model(device: torch.device) -> CNN:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"No checkpoint at {CHECKPOINT}. Run train.py first.")
    model = CNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    return model


def setup(seed: int = 42):
    """Common entry point every experiment script uses: seed, device, model, test set."""
    set_seed(seed)
    device = get_device()
    model = load_trained_model(device)
    _, _, _, test_set = make_loaders()
    return model, device, test_set
