"""Experiment 08 -- A CNN that distinguishes 8s from 2s in MNIST.

Experiment 04 asked "is this an 8?" across all of MNIST -- a task where
~90% of images are negatives, so a model that always says "no" already
scores ~90% "accuracy." This experiment removes that shortcut: the dataset
is filtered down to ONLY digits 8 and 2 before training, so the two classes
are genuinely balanced (~49.5% / 50.5%). A model that always guesses one
class now scores ~50%, not ~90% -- accuracy here actually has to reflect
real discriminative ability, and precision/recall stop being the load-
bearing metrics they were in experiment 04.

8 and 2 were picked deliberately, not arbitrarily: both have a closed loop
in some handwriting styles and share a curved lower stroke, making them a
plausibly *harder* pair than "8 vs. everything" (where "everything" includes
easy-to-distinguish digits like 1 or 7).

This CNN also exists to be experiment 09's distillation teacher.

Architecture: identical to experiment 04's (same 2-block CNN) -- the point
of this experiment is the *data*, not the model.

Run:
    python experiments/08_mnist_cnn_8_vs_2/train.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils import get_device, save_image_grid, save_loss_curve, set_seed  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
DATA_ROOT = REPO_ROOT / "data" / "raw"

POS_CLASS = 8  # logit > 0 means "this is an 8"
NEG_CLASS = 2  # the only other class in this filtered dataset


class CNN(nn.Module):
    """Same architecture as experiment 04's CNN."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def _filter_to_two_classes(dataset):
    targets = dataset.targets
    mask = (targets == POS_CLASS) | (targets == NEG_CLASS)
    indices = mask.nonzero().squeeze(1).tolist()
    return Subset(dataset, indices)


def make_loaders(batch_size: int, val_size: int, seed: int):
    transform = transforms.ToTensor()
    train_all = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    test_all = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)

    train_full = _filter_to_two_classes(train_all)
    test_set = _filter_to_two_classes(test_all)

    n_train = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def binary_labels(y: torch.Tensor) -> torch.Tensor:
    return (y == POS_CLASS).float()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    tp = fp = fn = tn = 0
    for x, y in loader:
        x = x.to(device)
        target = binary_labels(y).to(device)

        logits = model(x)
        total_loss += loss_fn(logits, target).item() * x.size(0)
        n += x.size(0)

        pred = (logits > 0).float()
        tp += ((pred == 1) & (target == 1)).sum().item()
        fp += ((pred == 1) & (target == 0)).sum().item()
        fn += ((pred == 0) & (target == 1)).sum().item()
        tn += ((pred == 0) & (target == 0)).sum().item()

    return {
        "loss": total_loss / n,
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    train_loader, val_loader, test_loader = make_loaders(args.batch_size, args.val_size, args.seed)
    print(
        f"train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}  "
        f"test: {len(test_loader.dataset)}  (filtered to digits {NEG_CLASS} and {POS_CLASS} only)"
    )

    model = CNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: CNN, {n_params} parameters")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            target = binary_labels(y).to(device)

            logits = model(x)
            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_loss = running_loss / n_seen
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        val_accuracies.append(val_metrics["accuracy"])

        print(
            f"epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_metrics['loss']:.4f} "
            f"| val acc {val_metrics['accuracy']:.4f} | val precision {val_metrics['precision']:.4f} "
            f"| val recall {val_metrics['recall']:.4f}"
        )

    test_metrics = evaluate(model, test_loader, loss_fn, device)
    test_targets = test_loader.dataset.dataset.targets[test_loader.dataset.indices]
    majority_baseline = max(
        (test_targets == POS_CLASS).float().mean().item(),
        (test_targets == NEG_CLASS).float().mean().item(),
    )

    print(f"\ntest accuracy:  {test_metrics['accuracy']:.4f}  (majority-class baseline: {majority_baseline:.4f})")
    print(f"test precision: {test_metrics['precision']:.4f}")
    print(f"test recall:    {test_metrics['recall']:.4f}")

    save_loss_curve(
        train_losses, val_losses,
        OUTPUTS / "loss_curve.png",
        title="Experiment 08 -- CNN on MNIST 8-vs-2: BCE loss",
    )
    save_loss_curve(
        val_accuracies, None,
        OUTPUTS / "accuracy_curve.png",
        title=f"Experiment 08 -- val accuracy vs. majority-class baseline ({majority_baseline:.3f})",
        log_y=False,
        ylabel="val accuracy",
        labels=("val", ""),
    )

    model.eval()
    with torch.no_grad():
        x_sample, y_sample = next(iter(DataLoader(test_loader.dataset, batch_size=64, shuffle=True)))
        logits = model(x_sample.to(device))
        preds = (logits > 0).cpu().long()
    true_is_8 = (y_sample == POS_CLASS).long()
    wrong = (preds != true_is_8).nonzero().squeeze(1).tolist()
    order = wrong + [i for i in range(len(y_sample)) if i not in wrong]
    idx = order[:16]
    save_image_grid(
        [x_sample[i, 0].numpy() for i in idx],
        [bool(true_is_8[i]) for i in idx],
        [bool(preds[i]) for i in idx],
        OUTPUTS / "predictions.png",
        title="Experiment 08 -- 8 (True) vs 2 (False)? (true / pred, mistakes in red)",
    )

    torch.save(model.state_dict(), OUTPUTS / "model.pt")
    print(f"saved plots and model.pt to {OUTPUTS}")


if __name__ == "__main__":
    main()
