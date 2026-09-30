"""Experiment 5 -- for representative filters from conv1 (early) and conv2
(later), find and display the test images that activate them most
strongly.

Run:
    python experiments/12_cnn_filter_activations/exp5_top_activating_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from activation_utils import batched_filter_scores, filter_label  # noqa: E402
from model import setup  # noqa: E402

OUTPUTS = HERE / "outputs" / "top_activating"
TOP_K = 9
N_SAMPLE = 3000  # subsample of the 10k test images, for speed

# 4 conv1 filters (skip the known-dead f2) + 4 conv2 filters, spanning the
# range of class-separation strength found in experiment 3.
CONV1_FILTERS = [0, 1, 5, 7]
CONV2_FILTERS = [8 + i for i in (0, 2, 5, 13)]
FILTERS_TO_SHOW = CONV1_FILTERS + CONV2_FILTERS


def main() -> None:
    model, device, test_set = setup()

    torch.manual_seed(0)
    indices = torch.randperm(len(test_set))[:N_SAMPLE].tolist()
    means, _, _, labels = batched_filter_scores(model, test_set, indices, device)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for filt in FILTERS_TO_SHOW:
        scores = means[:, filt]
        top_order = torch.argsort(scores, descending=True)[:TOP_K]
        top_indices = [indices[i] for i in top_order.tolist()]
        top_scores = scores[top_order].tolist()
        top_labels = [test_set[i][1] for i in top_indices]

        fig, axes = plt.subplots(3, 3, figsize=(7, 7.5))
        for ax, idx, score, label in zip(axes.flat, top_indices, top_scores, top_labels):
            img, _ = test_set[idx]
            ax.imshow(img.squeeze(0), cmap="gray")
            ax.set_title(f"label {label}\nscore {score:.2f}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Top-{TOP_K} activating images for {filter_label(filt)}\n"
                     f"(sample of {N_SAMPLE} test images)")
        fig.tight_layout()
        out_path = OUTPUTS / f"{filter_label(filt)}_top{TOP_K}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        label_counts = {}
        for l in top_labels:
            label_counts[l] = label_counts.get(l, 0) + 1
        print(f"{filter_label(filt):10s} top-{TOP_K} labels: {sorted(top_labels)}  "
              f"(saved {out_path.name})")

    print(f"\nall grids saved to {OUTPUTS}")


if __name__ == "__main__":
    main()
