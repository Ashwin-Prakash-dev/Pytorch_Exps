"""Experiment 02 -- A single linear layer CANNOT fit a sine wave.

This is the deliberate failure that motivates experiment 03. The data is
``y = sin(x)`` -- a curve. A single ``nn.Linear(1, 1)`` can only produce a
straight line ``w*x + b``, so no matter how long we train, the best it can do is
a nearly-flat line through the middle of the wave. The loss bottoms out far
above zero.

The takeaway: model *capacity* matters. A linear model is fundamentally unable
to represent a nonlinear function. To fit a curve we need nonlinearity -- which
is exactly what a hidden layer + activation provides (see experiment 03).

Run:
    python experiments/02_sine_wave_linear/train.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils import get_device, save_loss_curve, save_regression_fit, set_seed  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


def make_data(n: int, noise_std: float, device: torch.device):
    """``y = sin(x)`` over two periods, [-2pi, 2pi], with a little noise."""
    x = torch.linspace(-2 * torch.pi, 2 * torch.pi, n, device=device).unsqueeze(1)
    y = torch.sin(x) + torch.randn(n, 1, device=device) * noise_std
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    x, y = make_data(args.n, args.noise, device)
    n_val = args.n // 5
    # Shuffle before splitting: the data is sorted by x, so a tail split would
    # put an entire x-region in validation only. A random split keeps both sets
    # representative.
    perm = torch.randperm(args.n, device=device)
    x, y = x[perm], y[perm]
    x_train, y_train = x[:-n_val], y[:-n_val]
    x_val, y_val = x[-n_val:], y[-n_val:]

    # The whole point: one linear layer, no activation.
    model = nn.Linear(1, 1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss = loss_fn(model(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train {loss.item():.4f} | val {val_loss.item():.4f}")

    # var(sin) over a whole number of periods is 0.5, so a flat line at y=0 has
    # MSE ~= 0.5 + noise_var. Landing near that confirms the model gave up.
    print(f"\nfinal train loss: {train_losses[-1]:.4f}  (a flat line scores ~0.5 here)")

    save_loss_curve(
        train_losses, val_losses,
        OUTPUTS / "loss_curve.png",
        title="Experiment 02 -- linear layer on sin(x): loss stays high",
    )
    with torch.no_grad():
        x_line = torch.linspace(-2 * torch.pi, 2 * torch.pi, 300, device=device).unsqueeze(1)
        y_line = model(x_line)
    save_regression_fit(
        x.cpu().squeeze(1), y.cpu().squeeze(1),
        x_line.cpu().squeeze(1), y_line.cpu().squeeze(1),
        OUTPUTS / "fit.png",
        title="Experiment 02 -- a line cannot be a sine wave",
    )
    print(f"saved plots to {OUTPUTS}")


if __name__ == "__main__":
    main()
