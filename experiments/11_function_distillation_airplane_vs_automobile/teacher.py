"""Loads the frozen CNN from experiment 10 (CIFAR-10 airplane-vs-automobile,
balanced binary classification) and the exact filtered/split data it was
trained on. Same dynamic-import pattern as experiment 07's teacher.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_EXP10_DIR = REPO_ROOT / "experiments" / "10_cifar10_cnn_airplane_vs_automobile"
_CHECKPOINT = _EXP10_DIR / "outputs" / "model.pt"

_spec = importlib.util.spec_from_file_location("exp10_train", _EXP10_DIR / "train.py")
exp10_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp10_train)  # safe: exp10's training code is guarded by `if __name__ == "__main__"`

CNN = exp10_train.CNN
make_loaders = exp10_train.make_loaders
binary_labels = exp10_train.binary_labels
POS_CLASS = exp10_train.POS_CLASS
NEG_CLASS = exp10_train.NEG_CLASS
CLASS_NAMES = exp10_train.CLASS_NAMES


def load_frozen_teacher(device: torch.device) -> torch.nn.Module:
    if not _CHECKPOINT.exists():
        raise FileNotFoundError(
            f"No teacher checkpoint at {_CHECKPOINT}. Run "
            f"experiments/10_cifar10_cnn_airplane_vs_automobile/train.py first."
        )
    model = CNN().to(device)
    model.load_state_dict(torch.load(_CHECKPOINT, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_split_loaders(batch_size: int = 512, val_size: int = 1000, seed: int = 42):
    from torch.utils.data import DataLoader

    train_loader, val_loader, test_loader = make_loaders(batch_size=batch_size, val_size=val_size, seed=seed)
    return {
        "train": DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=False),
        "val": DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False),
    }
