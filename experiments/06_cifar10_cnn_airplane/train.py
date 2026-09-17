"""Experiment 06 -- A CNN that recognizes airplanes in CIFAR-10.

The RGB, harder-dataset sequel to experiment 04 (MNIST digit-8). Same binary
framing -- "is this an airplane?" -- and the same reason for it: airplane is
exactly 1 of 10 balanced CIFAR-10 classes, so ~10% of the data is positive
and a model that always says "no" still scores ~90% accuracy. Precision and
recall are tracked for the same reason as experiment 04.

CIFAR-10 images are 3x32x32 (RGB, 4x the pixels of MNIST) and visually much
harder than MNIST digits -- real photographs of real objects, not clean
handwritten strokes on a black background. Expect noticeably lower accuracy
than experiment 04's 99.1%; that gap is the point, not a bug -- this CNN is
built to be experiment 07's distillation *teacher*, and how good the teacher
itself is matters for interpreting what the student models can learn.

Architecture (one more conv block than experiment 04, since CIFAR-10 needs
more capacity to separate real photographs):

    Conv(3->16, 3x3) -> ReLU -> MaxPool(2)   # 32x32 -> 16x16
    Conv(16->32, 3x3) -> ReLU -> MaxPool(2)  # 16x16 -> 8x8
    Conv(32->64, 3x3) -> ReLU -> MaxPool(2)  # 8x8 -> 4x4
    Flatten -> Linear(64*4*4 -> 64) -> ReLU -> Linear(64 -> 1)

Run:
    python experiments/06_cifar10_cnn_airplane/train.py
    python experiments/06_cifar10_cnn_airplane/train.py --pos-weight 9.0   # counter the 9:1 imbalance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from cifar10_data import CIFAR10HF  # noqa: E402
from utils import get_device, save_image_grid, save_loss_curve, set_seed  # noqa: E402

OUTPUTS = HERE / "outputs"
DATA_ROOT = REPO_ROOT / "data" / "raw"

TARGET_CLASS = 0  # CIFAR-10 class index for "airplane"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class CNN(nn.Module):
    """Three conv blocks feeding a small classifier head. One output logit:
    positive means "this is an airplane"."""

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


def make_loaders(batch_size: int, val_size: int, seed: int):
    transform = transforms.ToTensor()
    train_full = CIFAR10HF(DATA_ROOT, train=True, download=True, transform=transform)
    test_set = CIFAR10HF(DATA_ROOT, train=False, download=True, transform=transform)

    n_train = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def binary_labels(y: torch.Tensor) -> torch.Tensor:
    return (y == TARGET_CLASS).float()


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
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pos-weight", type=float, default=1.0,
        help="weight on the positive (airplane) class in BCE; counters the ~9:1 class "
             "imbalance. 1.0 = unweighted, try 9.0 to see recall improve.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    train_loader, val_loader, test_loader = make_loaders(args.batch_size, args.val_size, args.seed)
    print(
        f"train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}  "
        f"test: {len(test_loader.dataset)}"
    )

    model = CNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: CNN, {n_params} parameters")

    pos_weight = torch.tensor(args.pos_weight, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
    test_targets = torch.as_tensor(test_loader.dataset.targets)
    baseline_accuracy = (test_targets != TARGET_CLASS).float().mean().item()

    print(f"\ntest accuracy:  {test_metrics['accuracy']:.4f}  (always-'not airplane' baseline: {baseline_accuracy:.4f})")
    print(f"test precision: {test_metrics['precision']:.4f}")
    print(f"test recall:    {test_metrics['recall']:.4f}")

    save_loss_curve(
        train_losses, val_losses,
        OUTPUTS / "loss_curve.png",
        title="Experiment 06 -- CNN on CIFAR-10: BCE loss",
    )
    save_loss_curve(
        val_accuracies, None,
        OUTPUTS / "accuracy_curve.png",
        title=f"Experiment 06 -- val accuracy vs. always-'not airplane' baseline ({baseline_accuracy:.3f})",
        log_y=False,
        ylabel="val accuracy",
        labels=("val", ""),
    )

    model.eval()
    with torch.no_grad():
        x_sample, y_sample = next(iter(DataLoader(test_loader.dataset, batch_size=64, shuffle=True)))
        logits = model(x_sample.to(device))
        preds = (logits > 0).cpu().long()
    true_is_airplane = (y_sample == TARGET_CLASS).long()
    wrong = (preds != true_is_airplane).nonzero().squeeze(1).tolist()
    order = wrong + [i for i in range(len(y_sample)) if i not in wrong]
    idx = order[:16]
    save_image_grid(
        [x_sample[i].permute(1, 2, 0).numpy() for i in idx],
        [bool(true_is_airplane[i]) for i in idx],
        [bool(preds[i]) for i in idx],
        OUTPUTS / "predictions.png",
        title="Experiment 06 -- is this an airplane? (true / pred, mistakes in red)",
    )

    torch.save(model.state_dict(), OUTPUTS / "model.pt")
    print(f"saved plots and model.pt to {OUTPUTS}")


if __name__ == "__main__":
    main()
