"""Device selection helper."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available device.

    Writing device-agnostic code from day one is a good habit: the same script
    runs on a laptop CPU and a CUDA GPU with no changes. You move both the model
    and the data tensors onto this device (``.to(device)``) before the forward
    pass.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
