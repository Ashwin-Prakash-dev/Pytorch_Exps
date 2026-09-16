"""Run every MNIST image (train/val/test) through the frozen CNN once and cache
the results: image, teacher raw logit, teacher sigmoid probability, and the
original MNIST label (0-9, kept only for evaluation -- never a training target
here).

This produces the dataset the student models actually learn from:

    image -> teacher logit

not:

    image -> ground-truth label

Run once before training any student model:
    python experiments/05_function_distillation/generate_teacher_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from teacher import binary_labels, get_split_loaders, load_frozen_teacher  # noqa: E402

OUTPUTS = HERE / "outputs"
CACHE = OUTPUTS / "cache"


@torch.no_grad()
def build_split_cache(model: torch.nn.Module, loader, device: torch.device) -> dict:
    images, logits, labels = [], [], []
    for x, y in loader:
        x = x.to(device)
        logit = model(x)
        # Images are stored as uint8 (they're already 0-255 grayscale under the
        # hood; ToTensor's [0,1] scaling is reapplied on load) -- ~4x smaller
        # on disk than float32, with no real precision loss for 8-bit sources.
        images.append((x.cpu() * 255).round().byte())
        logits.append(logit.cpu())
        labels.append(y)

    images = torch.cat(images)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    probs = torch.sigmoid(logits)
    return {
        "images": images,          # (N, 1, 28, 28) uint8
        "logits": logits,          # (N,) float32 -- the student's training target
        "probs": probs,            # (N,) float32
        "labels": labels,          # (N,) int64, original MNIST digit 0-9
        "is_target_digit": binary_labels(labels).long(),  # (N,) 0/1, teacher's ground-truth task
    }


def main() -> None:
    device = torch.device("cpu")
    model = load_frozen_teacher(device)
    loaders = get_split_loaders()

    CACHE.mkdir(parents=True, exist_ok=True)
    for split, loader in loaders.items():
        cache = build_split_cache(model, loader, device)
        path = CACHE / f"{split}.pt"
        torch.save(cache, path)
        agreement = (cache["is_target_digit"] == (cache["probs"] > 0.5).long()).float().mean().item()
        print(
            f"{split:5s}: {len(cache['labels']):5d} images | "
            f"logit range [{cache['logits'].min():.2f}, {cache['logits'].max():.2f}] | "
            f"teacher accuracy on its own task: {agreement:.4f} -> {path}"
        )


if __name__ == "__main__":
    main()
