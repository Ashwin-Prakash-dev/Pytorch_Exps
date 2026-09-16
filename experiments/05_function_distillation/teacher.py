"""Loads the frozen CNN from experiment 04 and the MNIST splits it was trained on.

This module does not define a new model or a new data pipeline -- it dynamically
imports ``experiments/04_mnist_cnn_digit8/train.py`` (whose folder name starts
with a digit, so it can't be `import`-ed normally) to reuse its exact ``CNN``
class, ``make_loaders`` split logic, and ``TARGET_DIGIT``/``binary_labels``.
Reusing that module -- rather than copy-pasting the class -- is what guarantees
the "teacher" here is really the same frozen model experiment 04 evaluated,
trained with the same seed and the same 55k/5k/10k split.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_EXP04_DIR = REPO_ROOT / "experiments" / "04_mnist_cnn_digit8"
_CHECKPOINT = _EXP04_DIR / "outputs" / "model.pt"

_spec = importlib.util.spec_from_file_location("exp04_train", _EXP04_DIR / "train.py")
exp04_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp04_train)  # safe: exp04's training code is guarded by `if __name__ == "__main__"`

CNN = exp04_train.CNN
make_loaders = exp04_train.make_loaders
binary_labels = exp04_train.binary_labels
TARGET_DIGIT = exp04_train.TARGET_DIGIT


def load_frozen_teacher(device: torch.device) -> torch.nn.Module:
    """Load experiment 04's trained CNN and freeze it: eval mode, no grad."""
    if not _CHECKPOINT.exists():
        raise FileNotFoundError(
            f"No teacher checkpoint at {_CHECKPOINT}. Run "
            f"experiments/04_mnist_cnn_digit8/train.py first."
        )
    model = CNN().to(device)
    model.load_state_dict(torch.load(_CHECKPOINT, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_split_loaders(batch_size: int = 512, val_size: int = 5000, seed: int = 42):
    """The exact same train/val/test split experiment 04 used, in a fixed
    (non-shuffled) order so images line up 1:1 with cached teacher outputs."""
    from torch.utils.data import DataLoader

    train_loader, val_loader, test_loader = make_loaders(batch_size=batch_size, val_size=val_size, seed=seed)
    return {
        "train": DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=False),
        "val": DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False),
    }
