"""Experiment 3 -- class-level activation patterns, quantified rather than
eyeballed.

For a large sample of test images per class, computes each filter's mean
activation per class (a class x filter matrix), then quantifies how much
each filter's activation actually separates the 10 classes:
  - one-way ANOVA F-statistic and eta-squared effect size per filter
    (scipy.stats.f_oneway) -- "does this filter's activation depend on
    class at all, and how strongly?"
  - the 10x10 correlation matrix between classes' mean activation
    *patterns* across filters -- "which classes look most/least alike to
    this network, in activation space?"

Run:
    python experiments/12_cnn_filter_activations/exp3_class_activation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from activation_utils import N_FILTERS, batched_filter_scores, filter_label  # noqa: E402
from model import setup  # noqa: E402

OUTPUTS = HERE / "outputs"
N_PER_CLASS = 200


def main() -> None:
    model, device, test_set = setup()

    by_class: dict[int, list[int]] = {c: [] for c in range(10)}
    for i in range(len(test_set)):
        label = test_set[i][1]
        if len(by_class[label]) < N_PER_CLASS:
            by_class[label].append(i)
        if all(len(v) >= N_PER_CLASS for v in by_class.values()):
            break
    n_actual = min(len(v) for v in by_class.values())
    print(f"using {n_actual} images per class (requested {N_PER_CLASS})")

    indices = [i for c in range(10) for i in by_class[c][:n_actual]]
    means, _, _, labels = batched_filter_scores(model, test_set, indices, device)
    means_np = means.numpy()
    labels_np = labels.numpy()

    class_means = np.stack([means_np[labels_np == c].mean(axis=0) for c in range(10)])  # (10, 24)

    # --- class x filter heatmap ---
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(class_means, aspect="auto", cmap="viridis")
    ax.set_xlabel("filter")
    ax.set_ylabel("digit class")
    ax.set_xticks(range(N_FILTERS))
    ax.set_xticklabels([filter_label(i) for i in range(N_FILTERS)], rotation=90, fontsize=7)
    ax.set_yticks(range(10))
    ax.set_title(f"Mean activation per class (averaged over {n_actual} test images/class)")
    fig.colorbar(im, ax=ax, label="mean activation")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "exp3_class_filter_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"saved {OUTPUTS / 'exp3_class_filter_heatmap.png'}")

    # --- per-filter ANOVA across classes ---
    dead_filters = [i for i in range(N_FILTERS) if means_np[:, i].var() < 1e-8]
    rows = []
    for i in range(N_FILTERS):
        if i in dead_filters:
            rows.append((i, float("nan"), float("nan"), float("nan")))
            continue
        groups = [means_np[labels_np == c, i] for c in range(10)]
        f_stat, p_value = stats.f_oneway(*groups)
        # eta-squared: between-group SS / total SS
        grand_mean = means_np[:, i].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((means_np[:, i] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total
        rows.append((i, f_stat, p_value, eta_sq))

    rows.sort(key=lambda r: (-r[3] if not np.isnan(r[3]) else float("-inf")))
    print(f"\n{'filter':12s} {'F-statistic':>12s} {'p-value':>12s} {'eta^2':>8s}")
    for i, f_stat, p_value, eta_sq in rows:
        tag = " (dead filter)" if i in dead_filters else ""
        if np.isnan(f_stat):
            print(f"{filter_label(i):12s} {'--':>12s} {'--':>12s} {'--':>8s}{tag}")
        else:
            print(f"{filter_label(i):12s} {f_stat:12.1f} {p_value:12.2e} {eta_sq:8.3f}{tag}")

    n_significant = sum(1 for _, _, p, _ in rows if not np.isnan(p) and p < 0.001)
    print(f"\n{n_significant} / {N_FILTERS - len(dead_filters)} live filters show class-dependent "
          f"mean activation at p < 0.001 (Bonferroni-uncorrected; with {N_FILTERS - len(dead_filters)} "
          f"tests, a conservative threshold would be p < {0.05 / (N_FILTERS - len(dead_filters)):.4f})")

    # --- class x class correlation of activation PATTERNS ---
    class_corr = np.corrcoef(class_means)
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
    im2 = ax2.imshow(class_corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    ax2.set_xlabel("digit class")
    ax2.set_ylabel("digit class")
    ax2.set_title("Correlation between classes' mean activation patterns\n(across all 24 filters)")
    for r in range(10):
        for c in range(10):
            ax2.text(c, r, f"{class_corr[r, c]:.2f}", ha="center", va="center", fontsize=7)
    fig2.colorbar(im2, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(OUTPUTS / "exp3_class_correlation.png", dpi=120)
    plt.close(fig2)
    print(f"\nsaved {OUTPUTS / 'exp3_class_correlation.png'}")

    off_diag = class_corr[~np.eye(10, dtype=bool)]
    most_similar_idx = np.unravel_index(np.argmax(class_corr - np.eye(10) * 2), class_corr.shape)
    most_different_idx = np.unravel_index(np.argmin(class_corr), class_corr.shape)
    print(f"mean off-diagonal class-pattern correlation: {off_diag.mean():.3f}")
    print(f"most similar pair: digits {most_similar_idx[0]} & {most_similar_idx[1]} "
          f"(r={class_corr[most_similar_idx]:.3f})")
    print(f"most different pair: digits {most_different_idx[0]} & {most_different_idx[1]} "
          f"(r={class_corr[most_different_idx]:.3f})")


if __name__ == "__main__":
    main()
