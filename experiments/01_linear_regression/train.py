

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

# --- make the repo-root `utils` package importable regardless of cwd ---------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils import get_device, save_loss_curve, save_regression_fit, set_seed  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

# Ground-truth line we will try to recover from noisy samples.
TRUE_W = 2.5
TRUE_B = -1.0


def make_data(n: int, noise_std: float, device: torch.device):
    """Generate ``y = TRUE_W * x + TRUE_B + noise`` on [-3, 3].

    Returns column tensors of shape (n, 1) -- PyTorch layers expect a feature
    dimension, so even a single feature is shape (batch, 1), not (batch,).
    """
    x = torch.linspace(-3, 3, n, device=device).unsqueeze(1)  # (n, 1)
    noise = torch.randn(n, 1, device=device) * noise_std
    y = TRUE_W * x + TRUE_B + noise
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=200, help="number of samples")
    parser.add_argument("--noise", type=float, default=1.0, help="noise std dev")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    # --- data ----------------------------------------------------------------
    x, y = make_data(args.n, args.noise, device)

    # Hold out the last 20% as a validation set to watch for generalization.
    # (For this tiny linear problem train/val track each other closely; the
    # split is here to build the habit.)
    n_val = args.n // 5
    x_train, y_train = x[:-n_val], y[:-n_val]
    x_val, y_val = x[-n_val:], y[-n_val:]

    # --- model, loss, optimizer ---------------------------------------------
    # nn.Linear(1, 1): one input feature -> one output. Its parameters are a
    # weight (1x1) and a bias (1,) -- exactly the slope and intercept.
    model = nn.Linear(1, 1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []

    # --- training loop -------------------------------------------------------
    # The five canonical steps, every epoch:
    #   1. forward pass        -> predictions
    #   2. compute loss        -> a scalar measuring error
    #   3. zero the gradients  -> they accumulate by default
    #   4. backward pass       -> autograd fills param.grad
    #   5. optimizer.step()    -> nudge params down the gradient
    for epoch in range(1, args.epochs + 1):
        model.train()
        y_pred = model(x_train)              # 1. forward
        loss = loss_fn(y_pred, y_train)      # 2. loss

        optimizer.zero_grad()                # 3. reset grads
        loss.backward()                      # 4. backprop
        optimizer.step()                     # 5. update

        # Evaluate without building the autograd graph (faster, less memory).
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train {loss.item():.4f} | val {val_loss.item():.4f}")

    # --- results -------------------------------------------------------------
    learned_w = model.weight.item()
    learned_b = model.bias.item()
    print("\nlearned vs. true parameters")
    print(f"  w: {learned_w:+.4f}  (true {TRUE_W:+.4f})")
    print(f"  b: {learned_b:+.4f}  (true {TRUE_B:+.4f})")

    # --- plots ---------------------------------------------------------------
    save_loss_curve(
        train_losses, val_losses,
        OUTPUTS / "loss_curve.png",
        title="Experiment 01 -- linear regression loss",
    )

    # Build the fitted line across the full x-range for a clean overlay.
    with torch.no_grad():
        x_line = torch.linspace(-3, 3, 100, device=device).unsqueeze(1)
        y_line = model(x_line)
    save_regression_fit(
        x.cpu().squeeze(1), y.cpu().squeeze(1),
        x_line.cpu().squeeze(1), y_line.cpu().squeeze(1),
        OUTPUTS / "fit.png",
        title=f"Experiment 01 -- fit  (y = {learned_w:.2f}x + {learned_b:.2f})",
    )
    print(f"\nsaved plots to {OUTPUTS}")


if __name__ == "__main__":
    main()
