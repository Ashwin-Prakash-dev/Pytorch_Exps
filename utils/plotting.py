"""Plot helpers that save figures to disk (no interactive display needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: just write PNGs, never pop a window
import matplotlib.pyplot as plt


def _prep(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_loss_curve(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None,
    path: str | Path,
    title: str = "Training loss",
    log_y: bool = True,
) -> None:
    """Plot loss vs. epoch. A log y-axis usually makes the descent readable."""
    path = _prep(path)
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_losses, label="train")
    if val_losses is not None:
        ax.plot(epochs, val_losses, label="val")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss" + (" (log scale)" if log_y else ""))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_regression_fit(
    x: Sequence[float],
    y: Sequence[float],
    x_line: Sequence[float],
    y_line: Sequence[float],
    path: str | Path,
    title: str = "Model fit",
) -> None:
    """Scatter the (x, y) data and overlay the model's prediction curve."""
    path = _prep(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=12, alpha=0.5, label="data")
    ax.plot(x_line, y_line, color="crimson", linewidth=2, label="model")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
