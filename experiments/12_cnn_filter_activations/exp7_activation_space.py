"""Experiment 7 (additional analysis) -- do images from different classes
occupy different regions of "activation space"?

Each image is reduced to its 24-dim activation vector (mean activation per
filter). This is projected to 2D with PCA and with t-SNE and plotted,
colored by class. Separately (and more rigorously than eyeballing a
scatter plot), pairwise cosine similarity between activation vectors is
compared for same-class vs. different-class pairs.

Run:
    python experiments/12_cnn_filter_activations/exp7_activation_space.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from activation_utils import batched_filter_scores  # noqa: E402
from model import setup  # noqa: E402

OUTPUTS = HERE / "outputs" / "activation_space"
N_PER_CLASS = 150
N_PAIRS_PER_KIND = 4000


def main() -> None:
    model, device, test_set = setup()
    rng = np.random.default_rng(42)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    by_class: dict[int, list[int]] = {c: [] for c in range(10)}
    for i in range(len(test_set)):
        label = test_set[i][1]
        if len(by_class[label]) < N_PER_CLASS:
            by_class[label].append(i)
        if all(len(v) >= N_PER_CLASS for v in by_class.values()):
            break

    indices = [i for c in range(10) for i in by_class[c]]
    means, _, _, labels = batched_filter_scores(model, test_set, indices, device)
    X = means.numpy()
    y = labels.numpy()

    # --- PCA ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA: explained variance ratio (PC1, PC2) = {pca.explained_variance_ratio_}")

    # --- t-SNE ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    X_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("tab10")
    for ax, X_2d, title in [(axes[0], X_pca, "PCA"), (axes[1], X_tsne, "t-SNE")]:
        for c in range(10):
            mask = y == c
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, alpha=0.6, color=cmap(c), label=str(c))
        ax.set_title(f"{title} of 24-dim activation vectors")
        ax.set_xlabel(f"{title} dim 1")
        ax.set_ylabel(f"{title} dim 2")
    axes[0].legend(title="digit", fontsize=8, loc="best", ncol=2)
    fig.suptitle(f"Activation space, {N_PER_CLASS} images/class ({len(X)} total)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "exp7_pca_tsne.png", dpi=120)
    plt.close(fig)
    print(f"saved {OUTPUTS / 'exp7_pca_tsne.png'}")

    # --- pairwise cosine similarity, same-class vs different-class ---
    sim_matrix = cosine_similarity(X)
    n = len(X)
    same_scores, diff_scores = [], []
    attempts = 0
    while len(same_scores) < N_PAIRS_PER_KIND and attempts < N_PAIRS_PER_KIND * 20:
        i, j = rng.integers(0, n, size=2)
        attempts += 1
        if i == j:
            continue
        if y[i] == y[j]:
            same_scores.append(sim_matrix[i, j])
    attempts = 0
    while len(diff_scores) < N_PAIRS_PER_KIND and attempts < N_PAIRS_PER_KIND * 20:
        i, j = rng.integers(0, n, size=2)
        attempts += 1
        if y[i] != y[j]:
            diff_scores.append(sim_matrix[i, j])

    same_scores = np.array(same_scores)
    diff_scores = np.array(diff_scores)
    u_stat, p_value = stats.mannwhitneyu(same_scores, diff_scores, alternative="greater")

    print(f"\nsame-class cosine similarity:      {same_scores.mean():.4f} +/- {same_scores.std():.4f}")
    print(f"different-class cosine similarity: {diff_scores.mean():.4f} +/- {diff_scores.std():.4f}")
    print(f"Mann-Whitney U (same > different): U={u_stat:.0f}, p={p_value:.2e}")

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.hist(same_scores, bins=40, alpha=0.6, label="same class", density=True)
    ax2.hist(diff_scores, bins=40, alpha=0.6, label="different class", density=True)
    ax2.axvline(same_scores.mean(), color="C0", linestyle="--")
    ax2.axvline(diff_scores.mean(), color="C1", linestyle="--")
    ax2.set_xlabel("cosine similarity between activation vectors")
    ax2.set_ylabel("density")
    ax2.set_title("Same-class vs. different-class activation-vector cosine similarity")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUTPUTS / "exp7_cosine_similarity.png", dpi=120)
    plt.close(fig2)
    print(f"saved {OUTPUTS / 'exp7_cosine_similarity.png'}")


if __name__ == "__main__":
    main()
