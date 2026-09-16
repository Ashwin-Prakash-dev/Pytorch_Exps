"""Section 9: does a student reproduce the CNN's actual input-output function,
or just the distribution of training images?

Take held-out test images, apply small perturbations the students never saw
during distillation (translation, rotation, Gaussian noise, brightness), and
compare CNN(x_modified) vs student(x_modified) on the SAME modified images.
If a student only memorized the CNN's outputs on the original training
manifold, its agreement with the teacher should degrade sharply off that
manifold; if it learned the actual function, agreement should degrade only
as much as the teacher's own predictions become genuinely ambiguous.

Requires: generate_teacher_dataset.py and sweep.py (or run_one.py) to have
produced student checkpoints in outputs/checkpoints/.

Run:
    python experiments/05_function_distillation/robustness_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import TeacherLogitDataset  # noqa: E402
from models import build_model  # noqa: E402
from sweep import build_configs  # noqa: E402
from teacher import load_frozen_teacher  # noqa: E402

OUTPUTS = HERE / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"

# One representative model per family -- the same families compared
# everywhere else in this experiment, at their largest configured capacity.
REPRESENTATIVE_MODELS = [
    "linear",
    "conventional_mlp_H128",
    "deepplain_H128_N8",
    "compositional_H128_N8",
    "parallel_B32_W16",
]

N_SAMPLES = 1000  # subset of the test set perturbations are applied to


def perturbations(img: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    """img: (1, 28, 28) float in [0, 1]. Returns named perturbed copies."""
    g = torch.Generator().manual_seed(seed)
    out = {
        "translate_2px": TF.affine(img, angle=0, translate=(2, 2), scale=1.0, shear=0),
        "rotate_10deg": TF.rotate(img, angle=10),
        "gaussian_noise": (img + torch.randn(img.shape, generator=g) * 0.15).clamp(0, 1),
        "brightness_1.3x": TF.adjust_brightness(img, brightness_factor=1.3),
    }
    return out


@torch.no_grad()
def evaluate_on_perturbation(teacher, student, images: torch.Tensor) -> dict:
    """images: (N, 1, 28, 28) float in [0, 1]."""
    teacher_logits = teacher(images)
    student_logits = student(images.flatten(1))
    mse = ((teacher_logits - student_logits) ** 2).mean().item()
    mae = (teacher_logits - student_logits).abs().mean().item()
    agreement = (((teacher_logits > 0).long()) == ((student_logits > 0).long())).float().mean().item()
    return {"mse": mse, "mae": mae, "agreement": agreement}


def main() -> None:
    device = torch.device("cpu")
    teacher = load_frozen_teacher(device)
    test_set = TeacherLogitDataset("test")

    configs_by_name = {c["name"]: c for c in build_configs()}

    torch.manual_seed(0)
    idx = torch.randperm(len(test_set))[:N_SAMPLES]
    images = torch.stack([test_set.images[i].float() / 255.0 for i in idx])  # (N,1,28,28)

    results = []
    for name in REPRESENTATIVE_MODELS:
        ckpt_path = CHECKPOINTS / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"skip {name}: no checkpoint at {ckpt_path} (run sweep.py first)")
            continue
        config = configs_by_name[name]
        student = build_model(config["arch"], **config["kwargs"])
        student.load_state_dict(torch.load(ckpt_path, map_location=device))
        student.eval()

        clean = evaluate_on_perturbation(teacher, student, images)
        row = {"model": name, "condition": "clean_test", **clean}
        results.append(row)
        print(f"{name:24s} clean       mse={clean['mse']:8.3f} agreement={clean['agreement']:.4f}")

        for pert_name in ("translate_2px", "rotate_10deg", "gaussian_noise", "brightness_1.3x"):
            perturbed = torch.stack([
                perturbations(images[i], seed=i)[pert_name] for i in range(images.size(0))
            ])
            metrics = evaluate_on_perturbation(teacher, student, perturbed)
            results.append({"model": name, "condition": pert_name, **metrics})
            print(f"{'':24s} {pert_name:16s} mse={metrics['mse']:8.3f} agreement={metrics['agreement']:.4f}")
        print()

    import csv
    out_path = OUTPUTS / "robustness_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "condition", "mse", "mae", "agreement"])
        writer.writeheader()
        writer.writerows(results)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
