"""Section 11: where does the student disagree most with the teacher?

For a chosen student model, ranks the held-out test set by
|CNN_logit - student_logit|, saves an image grid of the worst offenders
(with both models' logits/probabilities/predictions and the ground-truth
label), and does a light inspection of the ReLU-heavy student's internal
activations on those worst cases vs. a random control batch -- specifically,
the fraction of "dead" (zeroed) ReLU units per block, which is one visible
sign of where representational capacity is or isn't being used.

Run:
    python experiments/05_function_distillation/worst_case.py --model compositional_H128_N8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from common import TeacherLogitDataset  # noqa: E402
from models import CompositionalReLUMLP, build_model  # noqa: E402
from sweep import build_configs  # noqa: E402
from teacher import load_frozen_teacher  # noqa: E402

OUTPUTS = HERE / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
WORST_CASES = OUTPUTS / "worst_cases"


@torch.no_grad()
def rank_by_disagreement(teacher, student, test_set: TeacherLogitDataset, device):
    images = test_set.images.float().to(device) / 255.0     # (N,1,28,28)
    flat = images.flatten(1)
    teacher_logits = teacher(images)
    student_logits = student(flat)
    abs_err = (teacher_logits - student_logits).abs()
    order = torch.argsort(abs_err, descending=True)
    return order, teacher_logits, student_logits, abs_err


def block_dead_relu_fractions(model: CompositionalReLUMLP, x: torch.Tensor) -> list[float]:
    """Fraction of zero activations after each block's *last* ReLU, for the
    given batch. Only meaningful for CompositionalReLUMLP (has `.blocks`)."""
    fractions = []
    activations = {}

    def make_hook(i):
        def hook(_module, _inp, out):
            activations[i] = out
        return hook

    # block.block is Sequential(Linear, ReLU, Linear, ReLU, Linear, ReLU); index -1 is the last ReLU.
    handles = [block.block[-1].register_forward_hook(make_hook(i)) for i, block in enumerate(model.blocks)]
    with torch.no_grad():
        h = model.input_proj(x)
        for i, block in enumerate(model.blocks):
            h = block(h)
    for handle in handles:
        handle.remove()

    for i in sorted(activations):
        act = activations[i]
        fractions.append((act == 0).float().mean().item())
    return fractions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="compositional_H128_N8")
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cpu")
    teacher = load_frozen_teacher(device)
    test_set = TeacherLogitDataset("test")

    configs_by_name = {c["name"]: c for c in build_configs()}
    config = configs_by_name[args.model]
    student = build_model(config["arch"], **config["kwargs"])
    student.load_state_dict(torch.load(CHECKPOINTS / f"{args.model}.pt", map_location=device))
    student.eval()

    order, teacher_logits, student_logits, abs_err = rank_by_disagreement(teacher, student, test_set, device)
    worst_idx = order[: args.top_k]

    WORST_CASES.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, i in enumerate(worst_idx.tolist(), 1):
        t_logit, s_logit = teacher_logits[i].item(), student_logits[i].item()
        rows.append({
            "rank": rank,
            "test_index": i,
            "ground_truth_label": test_set.labels[i].item(),
            "cnn_logit": t_logit,
            "cnn_prob": torch.sigmoid(torch.tensor(t_logit)).item(),
            "cnn_pred_is_8": t_logit > 0,
            "mlp_logit": s_logit,
            "mlp_prob": torch.sigmoid(torch.tensor(s_logit)).item(),
            "mlp_pred_is_8": s_logit > 0,
            "abs_error": abs_err[i].item(),
        })

    csv_path = WORST_CASES / f"{args.model}_worst_cases.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}")
    for r in rows[:5]:
        print(r)

    # A custom grid, not utils.save_image_grid: that helper colors by
    # true_label == pred_label, which doesn't apply here -- these rows can
    # have large |logit| disagreement while still agreeing on the class
    # (is-it-an-8) decision. Color instead by whether the *classifications*
    # actually disagree, which is the more meaningful red flag.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 4
    n = len(rows)
    grid_rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(grid_rows, cols, figsize=(2.4 * cols, 2.6 * grid_rows))
    for ax, i, r in zip(axes.flat, worst_idx.tolist(), rows):
        class_disagree = r["cnn_pred_is_8"] != r["mlp_pred_is_8"]
        ax.imshow(test_set.images[i, 0].numpy(), cmap="gray")
        ax.set_title(
            f"gt{r['ground_truth_label']}  cnn{r['cnn_logit']:.1f}/mlp{r['mlp_logit']:.1f}"
            f"\n|err|={r['abs_error']:.1f}" + (" CLASS DISAGREE" if class_disagree else ""),
            fontsize=8, color="crimson" if class_disagree else "black",
        )
        ax.axis("off")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{args.model}: worst teacher/student disagreements (test set)")
    fig.tight_layout()
    out_path = WORST_CASES / f"{args.model}_worst_cases.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")

    if config["arch"] == "compositional_relu":
        images = test_set.images.float() / 255.0
        flat = images.flatten(1)
        worst_flat = flat[worst_idx]
        control_idx = order[len(order) // 2 - args.top_k // 2: len(order) // 2 + args.top_k // 2]
        control_flat = flat[control_idx]

        worst_fracs = block_dead_relu_fractions(student, worst_flat)
        control_fracs = block_dead_relu_fractions(student, control_flat)
        print("\ndead-ReLU fraction per block (0=all active, 1=all zeroed):")
        print(f"{'block':>6s} {'worst-case batch':>18s} {'median-error batch':>20s}")
        for i, (w, c) in enumerate(zip(worst_fracs, control_fracs)):
            print(f"{i:6d} {w:18.3f} {c:20.3f}")


if __name__ == "__main__":
    main()
