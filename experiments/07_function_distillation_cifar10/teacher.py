"""Loads the frozen CNN from experiment 06 and the CIFAR-10 splits it was
trained on.

Same pattern as experiment 05's teacher.py: dynamically imports
``experiments/06_cifar10_cnn_airplane/train.py`` (folder name starts with a
digit, so it can't be `import`-ed normally) to reuse its exact ``CNN``
class, ``make_loaders`` split logic, and ``TARGET_CLASS``/``binary_labels``
-- no CNN code duplicated.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_EXP06_DIR = REPO_ROOT / "experiments" / "06_cifar10_cnn_airplane"
_CHECKPOINT = _EXP06_DIR / "outputs" / "model.pt"

_spec = importlib.util.spec_from_file_location("exp06_train", _EXP06_DIR / "train.py")
exp06_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp06_train)  # safe: exp06's training code is guarded by `if __name__ == "__main__"`

CNN = exp06_train.CNN
make_loaders = exp06_train.make_loaders
binary_labels = exp06_train.binary_labels
TARGET_CLASS = exp06_train.TARGET_CLASS
CLASS_NAMES = exp06_train.CLASS_NAMES


def load_frozen_teacher(device: torch.device) -> torch.nn.Module:
    """Load experiment 06's trained CNN and freeze it: eval mode, no grad."""
    if not _CHECKPOINT.exists():
        raise FileNotFoundError(
            f"No teacher checkpoint at {_CHECKPOINT}. Run "
            f"experiments/06_cifar10_cnn_airplane/train.py first."
        )
    model = CNN().to(device)
    model.load_state_dict(torch.load(_CHECKPOINT, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_split_loaders(batch_size: int = 512, val_size: int = 5000, seed: int = 42):
    """The exact same train/val/test split experiment 06 used, in a fixed
    (non-shuffled) order so images line up 1:1 with cached teacher outputs."""
    from torch.utils.data import DataLoader

    train_loader, val_loader, test_loader = make_loaders(batch_size=batch_size, val_size=val_size, seed=seed)
    return {
        "train": DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=False),
        "val": DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False),
    }
