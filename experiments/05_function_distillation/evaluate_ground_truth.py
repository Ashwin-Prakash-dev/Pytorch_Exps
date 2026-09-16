"""Ground-truth task evaluation for one student model: how well does it
actually classify "is this an 8?", using the real MNIST labels -- not the
teacher's logit.

This is deliberately separate from the rest of the experiment. Everywhere
else, the student is trained and scored against the CNN's *function*
(logit MSE, teacher agreement); the ground-truth label is off-limits as a
training signal and only used sparingly for sanity checks. This script asks
the other question directly: if you'd deployed this distilled student as a
classifier, how good would it actually be, and how does that compare to
the teacher CNN it was distilled from and to the do-nothing baseline?

Run:
    python experiments/05_function_distillation/evaluate_ground_truth.py --model conventional_mlp_H32
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import TeacherLogitDataset  # noqa: E402
from models import build_model  # noqa: E402
from sweep import build_configs  # noqa: E402
from teacher import load_frozen_teacher  # noqa: E402

OUTPUTS = HERE / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"


@torch.no_grad()
def classification_metrics(pred: torch.Tensor, ground_truth: torch.Tensor) -> dict:
    tp = ((pred == 1) & (ground_truth == 1)).sum().item()
    fp = ((pred == 1) & (ground_truth == 0)).sum().item()
    fn = ((pred == 0) & (ground_truth == 1)).sum().item()
    tn = ((pred == 0) & (ground_truth == 0)).sum().item()
    n = tp + fp + fn + tn
    return {
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


@torch.no_grad()
def evaluate_split(student, teacher, dataset: TeacherLogitDataset) -> dict:
    images = dataset.images.float() / 255.0
    flat = images.flatten(1)
    ground_truth = dataset.is_target_digit

    student_pred = (student(flat) > 0).long()
    teacher_pred = (teacher(images) > 0).long()
    baseline_pred = torch.zeros_like(ground_truth)  # always predicts "not 8"

    return {
        "student": classification_metrics(student_pred, ground_truth),
        "teacher": classification_metrics(teacher_pred, ground_truth),
        "baseline": classification_metrics(baseline_pred, ground_truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="conventional_mlp_H32",
                         help="name from sweep.py's build_configs(); default is the smallest true MLP (26,209 params)")
    args = parser.parse_args()

    device = torch.device("cpu")
    teacher = load_frozen_teacher(device)

    configs_by_name = {c["name"]: c for c in build_configs()}
    config = configs_by_name[args.model]
    student = build_model(config["arch"], **config["kwargs"])
    ckpt_path = CHECKPOINTS / f"{args.model}.pt"
    student.load_state_dict(torch.load(ckpt_path, map_location=device))
    student.eval()

    print(f"model: {args.model}  arch={config['arch']}  kwargs={config['kwargs']}  "
          f"params={sum(p.numel() for p in student.parameters())}\n")

    rows = []
    for split in ("train", "val", "test"):
        dataset = TeacherLogitDataset(split)
        metrics = evaluate_split(student, teacher, dataset)

        print(f"=== {split} (n={len(dataset)}) ===")
        header = f'{"":10s} {"accuracy":>9s} {"precision":>10s} {"recall":>8s} {"tp":>6s} {"fp":>6s} {"fn":>6s} {"tn":>6s}'
        print(header)
        for who in ("baseline", "teacher", "student"):
            m = metrics[who]
            print(f'{who:10s} {m["accuracy"]:9.4f} {m["precision"]:10.4f} {m["recall"]:8.4f} '
                  f'{m["tp"]:6d} {m["fp"]:6d} {m["fn"]:6d} {m["tn"]:6d}')
            rows.append({"split": split, "who": who, "model_name": args.model, **m})
        print()

    out_path = OUTPUTS / f"ground_truth_eval_{args.model}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
