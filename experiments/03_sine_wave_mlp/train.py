"""Experiment 03 -- An MLP with one hidden layer DOES fit a sine wave.

Same data as experiment 02 (``y = sin(x)``), but now the model has a hidden
layer and a nonlinear activation:

    Linear(1 -> H) -> ReLU -> Linear(H -> 1)

The ReLU is what makes this work. Stacking two linear layers with nothing
between them collapses back into a single linear map (the product of two
matrices is a matrix) -- it would be no better than experiment 02. The
nonlinearity between them lets the network compose many little "bent" lines into
a smooth curve. With enough hidden units it can approximate any continuous
function (the universal approximation idea).

Two practical details that matter here:
  * Adam (adaptive optimizer) converges far faster than plain SGD on this.
  * We standardize the input x. Inputs spanning [-6.28, 6.28] make optimization
    sluggish; centering to mean 0 / std 1 is a cheap, standard speed-up.

Run:
    python experiments/03_sine_wave_mlp/train.py
    python experiments/03_sine_wave_mlp/train.py --hidden 16   # try a smaller net
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
    x = torch.linspace(-2 * torch.pi, 2 * torch.pi, n, device=device).unsqueeze(1)
    y = torch.sin(x) + torch.randn(n, 1, device=device) * noise_std
    return x, y


class MLP(nn.Module):
    """One hidden layer. The smallest network that can bend."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    x, y = make_data(args.n, args.noise, device)

    # Standardize x using TRAIN statistics only (computed below after the split)
    # -- using the full dataset's stats would leak validation info. We compute
    # the split first, then normalize.
    perm = torch.randperm(args.n, device=device)
    x, y = x[perm], y[perm]
    n_val = args.n // 5
    x_train, y_train = x[:-n_val], y[:-n_val]
    x_val, y_val = x[-n_val:], y[-n_val:]

    x_mean, x_std = x_train.mean(), x_train.std()
    x_train_n = (x_train - x_mean) / x_std
    x_val_n = (x_val - x_mean) / x_std

    model = MLP(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: MLP(hidden={args.hidden}), {n_params} parameters")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss = loss_fn(model(x_train_n), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val_n), y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train {loss.item():.4f} | val {val_loss.item():.4f}")

    # With noise_std=0.1 the irreducible MSE floor is ~0.01. Getting close to
    # that means the model fit the signal and only the noise remains.
    print(f"\nfinal train loss: {train_losses[-1]:.4f}  (noise floor ~{args.noise ** 2:.4f})")

    save_loss_curve(
        train_losses, val_losses,
        OUTPUTS / "loss_curve.png",
        title="Experiment 03 -- MLP on sin(x): loss drops to the noise floor",
    )
    # Plot the fit over a SORTED, normalized grid so the line is smooth.
    with torch.no_grad():
        x_line = torch.linspace(-2 * torch.pi, 2 * torch.pi, 400, device=device).unsqueeze(1)
        y_line = model((x_line - x_mean) / x_std)
    save_regression_fit(
        x.cpu().squeeze(1), y.cpu().squeeze(1),
        x_line.cpu().squeeze(1), y_line.cpu().squeeze(1),
        OUTPUTS / "fit.png",
        title="Experiment 03 -- a hidden layer + ReLU fits the curve",
    )
    print(f"saved plots to {OUTPUTS}")


if __name__ == "__main__":
    main()
