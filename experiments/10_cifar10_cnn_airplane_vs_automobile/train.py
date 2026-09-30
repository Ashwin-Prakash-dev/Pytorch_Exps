"""Experiment 10 -- A CNN that distinguishes airplanes from automobiles in
CIFAR-10.

Experiment 06 asked "is this an airplane?" across all of CIFAR-10 -- a task
where ~90% of images are negatives, so a model that always says "no" scores
~90% "accuracy." This experiment removes that shortcut: the dataset is
filtered down to ONLY airplane and automobile images before training, a
perfectly balanced 5,000/5,000 split (1,000/1,000 in test). A model that
always guesses one class now scores 50%, not 90%.

Airplane and automobile were picked deliberately: both are usually
photographed against a relatively clean, contrasty background (sky / road)
and both are rigid man-made objects with elongated bodies -- a plausibly
*easier* pair to separate than, say, cat vs. dog, but a genuinely balanced
one either way.

This CNN also exists to be experiment 11's distillation teacher.

Architecture: identical to experiment 06's (same 3-block CNN) -- the point
of this experiment is the *data*, not the model.

Run:
    python experiments/10_cifar10_cnn_airplane_vs_automobile/train.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXP06_DIR = REPO_ROOT / "experiments" / "06_cifar10_cnn_airplane"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP06_DIR))

from cifar10_data import CIFAR10HF, CLASS_NAMES  # noqa: E402
from utils import get_device, save_image_grid, save_loss_curve, set_seed  # noqa: E402

OUTPUTS = HERE / "outputs"
DATA_ROOT = REPO_ROOT / "data" / "raw"

POS_CLASS = 0  # airplane -- logit > 0 means "this is an airplane"
NEG_CLASS = 1  # automobile -- the only other class in this filtered dataset


class CNN(nn.Module):
    """Same architecture as experiment 06's CNN."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def _filter_to_two_classes(dataset):
    targets = torch.as_tensor(dataset.targets)
    mask = (targets == POS_CLASS) | (targets == NEG_CLASS)
    indices = mask.nonzero().squeeze(1).tolist()
    return Subset(dataset, indices)


def make_loaders(batch_size: int, val_size: int, seed: int):
    transform = transforms.ToTensor()
    train_all = CIFAR10HF(DATA_ROOT, train=True, download=True, transform=transform)
    test_all = CIFAR10HF(DATA_ROOT, train=False, download=True, transform=transform)

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
    parser.add_argument("--epochs", type=int, default=12)
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
        f"test: {len(test_loader.dataset)}  "
        f"(filtered to {CLASS_NAMES[NEG_CLASS]} and {CLASS_NAMES[POS_CLASS]} only)"
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
    test_targets = torch.as_tensor(test_loader.dataset.dataset.targets)[test_loader.dataset.indices]
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
        title="Experiment 10 -- CNN on CIFAR-10 airplane-vs-automobile: BCE loss",
    )
    save_loss_curve(
        val_accuracies, None,
        OUTPUTS / "accuracy_curve.png",
        title=f"Experiment 10 -- val accuracy vs. majority-class baseline ({majority_baseline:.3f})",
        log_y=False,
        ylabel="val accuracy",
        labels=("val", ""),
    )

    model.eval()
    with torch.no_grad():
        x_sample, y_sample = next(iter(DataLoader(test_loader.dataset, batch_size=64, shuffle=True)))
        logits = model(x_sample.to(device))
        preds = (logits > 0).cpu().long()
    true_is_airplane = (y_sample == POS_CLASS).long()
    wrong = (preds != true_is_airplane).nonzero().squeeze(1).tolist()
    order = wrong + [i for i in range(len(y_sample)) if i not in wrong]
    idx = order[:16]
    save_image_grid(
        [x_sample[i].permute(1, 2, 0).numpy() for i in idx],
        [bool(true_is_airplane[i]) for i in idx],
        [bool(preds[i]) for i in idx],
        OUTPUTS / "predictions.png",
        title="Experiment 10 -- airplane (True) vs automobile (False)? (true / pred, mistakes in red)",
    )

    torch.save(model.state_dict(), OUTPUTS / "model.pt")
    print(f"saved plots and model.pt to {OUTPUTS}")


if __name__ == "__main__":
    main()
