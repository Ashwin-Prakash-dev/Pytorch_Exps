"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed every RNG that can affect an experiment.

    Why this matters: training touches three independent sources of randomness
    -- Python's ``random`` (e.g. data shuffling), NumPy (data generation), and
    PyTorch (weight init, dropout, DataLoader workers). Seeding all of them is
    what makes a run repeatable.

    The cuDNN flags trade a little GPU speed for determinism; harmless on CPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
