"""Small shared helpers for PyTorch experiments.

Kept deliberately thin: experiments spell out their own models and training
loops (that's where the learning happens). This package only holds true
boilerplate that would otherwise be copy-pasted everywhere.
"""

from .seed import set_seed
from .device import get_device
from .plotting import save_loss_curve, save_regression_fit, save_image_grid

__all__ = [
    "set_seed",
    "get_device",
    "save_loss_curve",
    "save_regression_fit",
    "save_image_grid",
]
