"""Experiment 4 -- are individual images actually using different SUBSETS
of filters, or just different activation magnitudes on the same filters?

This is the sharpest test of the hypothesis. Experiments 2-3 show
activation *magnitude* varies by image/class; this experiment asks whether
the *set* of "active" filters differs, using Jaccard similarity between
images' active-filter sets -- same-class pairs vs. different-class pairs.

"Active" is defined two ways (both reported, since the threshold choice
matters): (1) trivial post-ReLU activity (mean activation > 0), and (2) a
stricter top-K threshold (a filter counts as "active" only if it's among
the top-8 of 24 filters for that image, by mean activation) -- (1) mostly
reflects which filters are dead vs. alive in general; (2) is the more
meaningful "what is this image actually using" question.

Run:
    python experiments/12_cnn_filter_activations/exp4_active_filter_overlap.py
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

from activation_utils import N_FILTERS, batched_filter_scores  # noqa: E402
from model import setup  # noqa: E402

OUTPUTS = HERE / "outputs"
N_PER_CLASS = 100
TOP_K = 8
N_PAIRS_PER_KIND = 3000


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return np.logical_and(a, b).sum() / union


def active_sets(means_np: np.ndarray, mode: str) -> np.ndarray:
    if mode == "nonzero":
        return means_np > 0
    if mode == "top_k":
        active = np.zeros_like(means_np, dtype=bool)
        top_idx = np.argsort(-means_np, axis=1)[:, :TOP_K]
        for row, cols in enumerate(top_idx):
            active[row, cols] = True
        return active
    raise ValueError(mode)


def main() -> None:
    model, device, test_set = setup()
    rng = np.random.default_rng(42)

    by_class: dict[int, list[int]] = {c: [] for c in range(10)}
    for i in range(len(test_set)):
        label = test_set[i][1]
        if len(by_class[label]) < N_PER_CLASS:
            by_class[label].append(i)
        if all(len(v) >= N_PER_CLASS for v in by_class.values()):
            break

    indices = [i for c in range(10) for i in by_class[c]]
    means, _, _, labels = batched_filter_scores(model, test_set, indices, device)
    means_np = means.numpy()
    labels_np = labels.numpy()

    results = {}
    for mode in ("nonzero", "top_k"):
        active = active_sets(means_np, mode)
        n = len(active)

        same_class_scores, diff_class_scores = [], []
        # Sample pairs rather than all C(n,2) -- n is a few thousand.
        attempts = 0
        while len(same_class_scores) < N_PAIRS_PER_KIND and attempts < N_PAIRS_PER_KIND * 20:
            i, j = rng.integers(0, n, size=2)
            attempts += 1
            if i == j:
                continue
            if labels_np[i] == labels_np[j]:
                same_class_scores.append(jaccard(active[i], active[j]))
        attempts = 0
        while len(diff_class_scores) < N_PAIRS_PER_KIND and attempts < N_PAIRS_PER_KIND * 20:
            i, j = rng.integers(0, n, size=2)
            attempts += 1
            if labels_np[i] != labels_np[j]:
                diff_class_scores.append(jaccard(active[i], active[j]))

        same_class_scores = np.array(same_class_scores)
        diff_class_scores = np.array(diff_class_scores)
        t_stat, p_value = stats.mannwhitneyu(same_class_scores, diff_class_scores, alternative="greater")

        n_active_per_image = active.sum(axis=1)
        results[mode] = {
            "same_mean": same_class_scores.mean(), "same_std": same_class_scores.std(),
            "diff_mean": diff_class_scores.mean(), "diff_std": diff_class_scores.std(),
            "u_stat": t_stat, "p_value": p_value,
            "n_active_mean": n_active_per_image.mean(), "n_active_std": n_active_per_image.std(),
            "same_scores": same_class_scores, "diff_scores": diff_class_scores,
        }

        print(f"--- active-filter definition: {mode} ---")
        print(f"active filters per image: {n_active_per_image.mean():.2f} +/- {n_active_per_image.std():.2f} (of {N_FILTERS})")
        print(f"same-class Jaccard similarity:      {same_class_scores.mean():.4f} +/- {same_class_scores.std():.4f}")
        print(f"different-class Jaccard similarity: {diff_class_scores.mean():.4f} +/- {diff_class_scores.std():.4f}")
        print(f"Mann-Whitney U test (same > different): U={t_stat:.0f}, p={p_value:.2e}")
        print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, mode in zip(axes, ("nonzero", "top_k")):
        r = results[mode]
        ax.hist(r["same_scores"], bins=30, alpha=0.6, label="same class", density=True)
        ax.hist(r["diff_scores"], bins=30, alpha=0.6, label="different class", density=True)
        ax.axvline(r["same_mean"], color="C0", linestyle="--")
        ax.axvline(r["diff_mean"], color="C1", linestyle="--")
        ax.set_xlabel("Jaccard similarity of active-filter sets")
        ax.set_ylabel("density")
        ax.set_title(f'"{mode}" active-filter definition')
        ax.legend(fontsize=8)
    fig.suptitle("Same-class vs. different-class active-filter-set overlap")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "exp4_jaccard_similarity.png", dpi=120)
    plt.close(fig)
    print(f"saved {OUTPUTS / 'exp4_jaccard_similarity.png'}")


if __name__ == "__main__":
    main()
