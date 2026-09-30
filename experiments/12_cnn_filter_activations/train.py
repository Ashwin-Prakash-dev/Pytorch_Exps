"""Experiment 12 -- train the CNN whose internal activations the rest of
this experiment inspects.

Standard 10-way MNIST classification, CrossEntropyLoss. This script's only
job is to produce a reasonably accurate, reproducible checkpoint -- the
actual experiment (does this network activate different filter subsets for
different inputs?) happens in the exp1-exp7 scripts that load this
checkpoint and never touch training again.

Run:
    python experiments/12_cnn_filter_activations/train.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model import CNN, OUTPUTS, make_loaders  # noqa: E402
from utils import get_device, save_loss_curve, set_seed  # noqa: E402


@torch.no_grad()
def evaluate(model: nn.Module, loader, loss_fn, device) -> dict:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return {"loss": total_loss / n, "accuracy": correct / n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    train_loader, test_loader, train_set, test_set = make_loaders(args.batch_size)
    print(f"train: {len(train_set)}  test: {len(test_set)}")

    model = CNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: CNN, {n_params} parameters "
          f"(conv1: {model.conv1.out_channels} filters, conv2: {model.conv2.out_channels} filters)")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, test_losses, test_accuracies = [], [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_loss = running_loss / n_seen
        test_metrics = evaluate(model, test_loader, loss_fn, device)

        train_losses.append(train_loss)
        test_losses.append(test_metrics["loss"])
        test_accuracies.append(test_metrics["accuracy"])

        print(f"epoch {epoch} | train loss {train_loss:.4f} | test loss {test_metrics['loss']:.4f} "
              f"| test acc {test_metrics['accuracy']:.4f}")

    # Per-class accuracy, since the rest of this experiment cares about
    # per-class behavior -- worth knowing the classifier isn't lopsided.
    model.eval()
    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            for c in range(10):
                mask = y == c
                class_total[c] += mask.sum()
                class_correct[c] += (preds[mask] == c).sum()
    print("\nper-class test accuracy:")
    for c in range(10):
        print(f"  {c}: {(class_correct[c] / class_total[c]).item():.4f}")

    save_loss_curve(
        train_losses, test_losses,
        OUTPUTS / "loss_curve.png",
        title="Experiment 12 -- CNN on MNIST (10-way): cross-entropy loss",
        labels=("train", "test"),
    )
    save_loss_curve(
        test_accuracies, None,
        OUTPUTS / "accuracy_curve.png",
        title="Experiment 12 -- test accuracy",
        log_y=False,
        ylabel="test accuracy",
        labels=("test", ""),
    )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUTS / "model.pt")
    print(f"\nsaved plots and model.pt to {OUTPUTS}")


if __name__ == "__main__":
    main()
