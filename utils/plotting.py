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
    ylabel: str = "loss",
    labels: tuple[str, str] = ("train", "val"),
) -> None:
    """Plot a metric vs. epoch. A log y-axis usually makes loss descent readable."""
    path = _prep(path)
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_losses, label=labels[0])
    if val_losses is not None:
        ax.plot(epochs, val_losses, label=labels[1])
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel + (" (log scale)" if log_y else ""))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_image_grid(
    images: Sequence,
    true_labels: Sequence,
    pred_labels: Sequence,
    path: str | Path,
    title: str = "Predictions",
    n: int = 16,
) -> None:
    """Show a grid of images with true vs. predicted labels (wrong ones in red).

    ``images`` are 2D arrays (H, W) already on CPU; ``true_labels`` /
    ``pred_labels`` are anything ``str()``-able (ints, bools, ...).
    """
    path = _prep(path)
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        correct = true_labels[i] == pred_labels[i]
        ax.imshow(images[i], cmap="gray")
        ax.set_title(
            f"true {true_labels[i]} / pred {pred_labels[i]}",
            fontsize=9,
            color="black" if correct else "crimson",
        )
        ax.axis("off")
    fig.suptitle(title)
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
