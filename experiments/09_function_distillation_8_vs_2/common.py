"""Shared training/eval code for every student model: the cached-teacher
dataset, the distillation training loop (MSE against the CNN's logit, never
against the MNIST label), and the function-approximation + task-agreement
metrics used throughout the experiment.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
CACHE = HERE / "outputs" / "cache"


class TeacherLogitDataset(Dataset):
    """Wraps a cached split: flat [0,1] image -> teacher logit.

    `labels` and `is_target_digit` are carried along for evaluation only --
    nothing in the training loop reads them.
    """

    def __init__(self, split: str):
        cache = torch.load(CACHE / f"{split}.pt")
        self.images = cache["images"]            # (N, 1, 28, 28) uint8
        self.logits = cache["logits"].float()     # (N,) -- the training target
        self.probs = cache["probs"].float()
        self.labels = cache["labels"].long()
        self.is_target_digit = cache["is_target_digit"].long()

    def __len__(self) -> int:
        return len(self.logits)

    def __getitem__(self, idx: int):
        x = self.images[idx].float().flatten() / 255.0  # (784,), matches CNN's own [0,1] input scale
        return x, self.logits[idx]


def load_splits(batch_size: int = 256):
    splits = {name: TeacherLogitDataset(name) for name in ("train", "val", "test")}
    loaders = {
        "train": DataLoader(splits["train"], batch_size=batch_size, shuffle=True),
        "val": DataLoader(splits["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(splits["test"], batch_size=batch_size, shuffle=False),
    }
    return splits, loaders


def train_student(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    log: bool = True,
) -> dict:
    """Train `model` to regress the teacher's logit (MSE). Returns per-epoch
    train/val MSE history plus wall-clock training time."""
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_mse": [], "val_mse": []}
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            pred = model(x)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_mse = running / n_seen
        val_mse = function_error(model, val_loader, device)["mse"]
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        if log:
            print(f"  epoch {epoch:2d} | train MSE {train_mse:8.4f} | val MSE {val_mse:8.4f}")

    history["train_time_sec"] = time.perf_counter() - start
    return history


@torch.no_grad()
def function_error(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Core function-approximation metrics: how close is the student's logit
    to the teacher's logit, on whatever split `loader` covers."""
    model.eval()
    se_sum, ae_sum, n = 0.0, 0.0, 0
    preds, targets = [], []
    for x, target in loader:
        x, target = x.to(device), target.to(device)
        pred = model(x)
        se_sum += ((pred - target) ** 2).sum().item()
        ae_sum += (pred - target).abs().sum().item()
        n += x.size(0)
        preds.append(pred.cpu())
        targets.append(target.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    correlation = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()

    return {
        "mse": se_sum / n,
        "mae": ae_sum / n,
        "correlation": correlation,
    }


@torch.no_grad()
def task_agreement(model: nn.Module, dataset: TeacherLogitDataset, device: torch.device, batch_size: int = 512) -> dict:
    """How well does the student's *classification behavior* match the
    teacher's, and separately, the ground-truth MNIST label? Also computes
    KL(teacher_prob || student_prob), symmetrized as JS-style average since
    plain KL blows up wherever either probability saturates near 0 or 1
    (which happens constantly here -- see the +-30 logit range)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    student_logits = []
    for x, _ in loader:
        student_logits.append(model(x.to(device)).cpu())
    student_logits = torch.cat(student_logits)

    teacher_probs = dataset.probs.clamp(1e-6, 1 - 1e-6)
    student_probs = torch.sigmoid(student_logits).clamp(1e-6, 1 - 1e-6)

    teacher_pred = (teacher_probs > 0.5).long()
    student_pred = (student_probs > 0.5).long()
    ground_truth = dataset.is_target_digit

    kl = (teacher_probs * (teacher_probs / student_probs).log()
          + (1 - teacher_probs) * ((1 - teacher_probs) / (1 - student_probs)).log()).mean().item()

    return {
        "teacher_student_agreement": (teacher_pred == student_pred).float().mean().item(),
        "student_vs_ground_truth_accuracy": (student_pred == ground_truth).float().mean().item(),
        "teacher_vs_ground_truth_accuracy": (teacher_pred == ground_truth).float().mean().item(),
        "kl_teacher_student": kl,
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
