"""Same plot suite as experiment 05's make_plots.py, pointed at this
experiment's outputs/results.csv (the CIFAR-10 airplane teacher's sweep).

Run after sweep.py:
    python experiments/07_function_distillation_cifar10/make_plots.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUTS = HERE / "outputs"
PLOTS = OUTPUTS / "plots"

ARCH_STYLE = {
    "linear": dict(color="#888888", marker="x", label="linear"),
    "conventional_mlp": dict(color="#1f77b4", marker="o", label="conventional MLP"),
    "deep_plain_mlp": dict(color="#ff7f0e", marker="s", label="deep plain MLP (param-matched)"),
    "compositional_relu": dict(color="#d62728", marker="^", label="compositional ReLU (residual)"),
    "parallel_bank": dict(color="#2ca02c", marker="D", label="parallel ReLU bank"),
}


def load_results() -> list[dict]:
    with open(OUTPUTS / "results.csv") as f:
        rows = list(csv.DictReader(f))
    numeric_fields = [k for k in rows[0] if k not in ("name", "arch")]
    for row in rows:
        for k in numeric_fields:
            try:
                row[k] = float(row[k]) if row[k] != "" else None
            except ValueError:
                pass
    return rows


def scatter_by_arch(rows, x_key, y_key, path, title, xlabel, ylabel, logx=False, logy=False):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    seen = set()
    for row in rows:
        style = ARCH_STYLE[row["arch"]]
        label = style["label"] if row["arch"] not in seen else None
        seen.add(row["arch"])
        ax.scatter(row[x_key], row[y_key], color=style["color"], marker=style["marker"],
                   s=60, alpha=0.85, label=label, edgecolors="black" if style["marker"] != "x" else None, linewidths=0.4)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


def compositional_vs_matched_plain(rows, path):
    by_h = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["arch"] in ("compositional_relu", "deep_plain_mlp"):
            h = int(row["cfg_hidden"])
            n = int(row["cfg_n_blocks"]) if row["arch"] == "compositional_relu" else int(row["cfg_n_layers"]) // 3
            by_h[h][n][row["arch"]] = row["test_mse"]

    fig, axes = plt.subplots(1, len(by_h), figsize=(5 * len(by_h), 4.5), sharey=True)
    if len(by_h) == 1:
        axes = [axes]
    for ax, h in zip(axes, sorted(by_h)):
        ns = sorted(by_h[h])
        comp = [by_h[h][n].get("compositional_relu") for n in ns]
        plain = [by_h[h][n].get("deep_plain_mlp") for n in ns]
        ax.plot(ns, comp, "o-", color=ARCH_STYLE["compositional_relu"]["color"], label="compositional (residual)")
        ax.plot(ns, plain, "s-", color=ARCH_STYLE["deep_plain_mlp"]["color"], label="plain (no residual)")
        ax.set_title(f"H={h}")
        ax.set_xlabel("N (blocks) -- same param count at each N")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("test MSE (log scale)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Residual structure vs. depth, at IDENTICAL parameter counts (CIFAR-10 airplane teacher)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = load_results()

    scatter_by_arch(
        rows, "n_params", "test_mse", PLOTS / "test_error_vs_params.png",
        "Test function error (MSE vs. teacher logit) vs. parameter count",
        "parameters (log scale)", "test MSE (log scale)", logx=True, logy=True,
    )
    scatter_by_arch(
        rows, "n_relu_layers", "test_mse", PLOTS / "test_error_vs_relu_layers.png",
        "Test function error vs. number of ReLU transformations",
        "# ReLU layers", "test MSE (log scale)", logy=True,
    )
    scatter_by_arch(
        rows, "flops_per_sample", "test_mse", PLOTS / "test_error_vs_flops.png",
        "Test function error vs. approximate FLOPs per sample",
        "FLOPs/sample (log scale)", "test MSE (log scale)", logx=True, logy=True,
    )
    scatter_by_arch(
        rows, "n_params", "test_teacher_student_agreement", PLOTS / "agreement_vs_params.png",
        "Teacher/student classification agreement vs. parameter count",
        "parameters (log scale)", "agreement (fraction of test set)", logx=True,
    )
    compositional_vs_matched_plain(rows, PLOTS / "compositional_vs_matched_plain.png")


if __name__ == "__main__":
    main()
