"""Experiment 2 -- do different images produce different activation
patterns across filters?

Builds an (images x filters) matrix of mean activation scores for a sample
of test images spanning all 10 classes, and renders it as a heatmap with
rows grouped and labeled by class.

Run:
    python experiments/12_cnn_filter_activations/exp2_activation_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from activation_utils import N_FILTERS, batched_filter_scores, filter_label  # noqa: E402
from model import setup  # noqa: E402

OUTPUTS = HERE / "outputs"
IMAGES_PER_CLASS = 8


def main() -> None:
    model, device, test_set = setup()

    # IMAGES_PER_CLASS images per digit, grouped by class for a readable heatmap.
    by_class: dict[int, list[int]] = {c: [] for c in range(10)}
    for i in range(len(test_set)):
        label = test_set[i][1]
        if len(by_class[label]) < IMAGES_PER_CLASS:
            by_class[label].append(i)
        if all(len(v) >= IMAGES_PER_CLASS for v in by_class.values()):
            break

    indices = [i for c in range(10) for i in by_class[c]]
    means, _, _, labels = batched_filter_scores(model, test_set, indices, device)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(means.numpy(), aspect="auto", cmap="viridis")
    ax.set_xlabel("filter (conv1 f0-f7, then conv2 f0-f15)")
    ax.set_ylabel("image (grouped by digit class)")
    ax.set_xticks(range(N_FILTERS))
    ax.set_xticklabels([filter_label(i) for i in range(N_FILTERS)], rotation=90, fontsize=7)

    # Label class-group boundaries on the y-axis.
    boundaries = [i * IMAGES_PER_CLASS for i in range(11)]
    ax.set_yticks([(boundaries[c] + boundaries[c + 1]) / 2 - 0.5 for c in range(10)])
    ax.set_yticklabels([f"digit {c}" for c in range(10)])
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="white", linewidth=1, alpha=0.7)

    ax.set_title(f"Mean activation per filter, {IMAGES_PER_CLASS} images per class (image x filter matrix)")
    fig.colorbar(im, ax=ax, label="mean activation")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "exp2_activation_matrix.png", dpi=120)
    plt.close(fig)
    print(f"saved {OUTPUTS / 'exp2_activation_matrix.png'}")

    # Quantify: is the row-to-row variation systematic (class-related) or just
    # noise? Compare within-class row variance to between-class row variance
    # per filter (this is the same question experiment 3 answers formally --
    # here it's a quick sanity check that the heatmap's visible banding is real).
    means_np = means.numpy()
    class_of_row = np.repeat(np.arange(10), IMAGES_PER_CLASS)
    within_var = np.mean([means_np[class_of_row == c].var(axis=0) for c in range(10)], axis=0)
    overall_var = means_np.var(axis=0)

    dead_filters = [i for i in range(N_FILTERS) if overall_var[i] < 1e-8]
    if dead_filters:
        print(f"\ndead filters (zero activation variance across this whole sample): "
              f"{[filter_label(i) for i in dead_filters]}")
        print("(excluded from the variance-ratio ranking below -- 0/0 there is undefined, not informative)")

    live = [i for i in range(N_FILTERS) if i not in dead_filters]
    between_ratio = 1 - (within_var[live] / overall_var[live])
    print("\nfraction of each live filter's total variance explained by class "
          "(1 - within-class/overall; quick sanity check for experiment 3):")
    for rank_i in np.argsort(-between_ratio)[:8]:
        i = live[rank_i]
        print(f"  {filter_label(i):10s} {between_ratio[rank_i]:.3f}")


if __name__ == "__main__":
    main()
