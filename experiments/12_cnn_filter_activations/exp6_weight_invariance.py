"""Experiment 6 -- explicit demonstration that the WEIGHTS do not change
between images, only the ACTIVATIONS do.

This is the premise the rest of the experiment rests on, so it's checked
directly and mechanically rather than assumed: clone conv1/conv2 weights,
run image A, clone again, run image B, clone again, and diff every pair.
Then show the two images' activations are different despite identical
weights throughout.

Run:
    python experiments/12_cnn_filter_activations/exp6_weight_invariance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from activation_utils import ActivationCapture, filter_scores  # noqa: E402
from model import setup  # noqa: E402

OUTPUTS = HERE / "outputs"


def snapshot_weights(model) -> dict[str, torch.Tensor]:
    return {
        "conv1.weight": model.conv1.weight.detach().clone(),
        "conv1.bias": model.conv1.bias.detach().clone(),
        "conv2.weight": model.conv2.weight.detach().clone(),
        "conv2.bias": model.conv2.bias.detach().clone(),
    }


def max_abs_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return max((a[k] - b[k]).abs().max().item() for k in a)


def main() -> None:
    model, device, test_set = setup()

    # Two images from DIFFERENT classes, picked by scanning labels.
    idx_a = next(i for i in range(len(test_set)) if test_set[i][1] == 3)
    idx_b = next(i for i in range(len(test_set)) if test_set[i][1] == 7)
    img_a = test_set[idx_a][0].unsqueeze(0).to(device)
    img_b = test_set[idx_b][0].unsqueeze(0).to(device)

    print(f"image A: test index {idx_a}, label {test_set[idx_a][1]}")
    print(f"image B: test index {idx_b}, label {test_set[idx_b][1]}")
    print()

    weights_before = snapshot_weights(model)

    with ActivationCapture(model) as cap:
        acts_a = cap.run(img_a)
    weights_after_a = snapshot_weights(model)

    with ActivationCapture(model) as cap:
        acts_b = cap.run(img_b)
    weights_after_b = snapshot_weights(model)

    diff_before_vs_after_a = max_abs_diff(weights_before, weights_after_a)
    diff_after_a_vs_after_b = max_abs_diff(weights_after_a, weights_after_b)
    diff_before_vs_after_b = max_abs_diff(weights_before, weights_after_b)

    print(f"max |weight difference| before inference vs. after running image A = {diff_before_vs_after_a}")
    print(f"max |weight difference| after image A vs. after image B          = {diff_after_a_vs_after_b}")
    print(f"max |weight difference| before inference vs. after running image B = {diff_before_vs_after_b}")

    assert diff_before_vs_after_a == 0.0
    assert diff_after_a_vs_after_b == 0.0
    assert diff_before_vs_after_b == 0.0
    print("\n-> confirmed: the weights are bit-for-bit identical across both forward passes.")

    scores_a = filter_scores(acts_a).squeeze(0)
    scores_b = filter_scores(acts_b).squeeze(0)
    activation_diff = (scores_a - scores_b).abs()

    print(f"\nmean |activation difference| across all 24 filters = {activation_diff.mean().item():.4f}")
    print(f"max  |activation difference| (single filter)        = {activation_diff.max().item():.4f}")
    n_filters_differ = (activation_diff > 1e-4).sum().item()
    print(f"filters differing by > 1e-4: {n_filters_differ} / {len(activation_diff)}")

    print(
        "\nconclusion: identical weights, run twice, produced measurably different\n"
        "activations for two different inputs. The weights (what the filters ARE) did\n"
        "not change; the activations (what the filters OUTPUT for this specific image)\n"
        "did. This is the distinction the rest of this experiment investigates."
    )

    with open(OUTPUTS / "exp6_weight_invariance.txt", "w") as f:
        f.write(
            f"image A: test index {idx_a}, label {test_set[idx_a][1]}\n"
            f"image B: test index {idx_b}, label {test_set[idx_b][1]}\n\n"
            f"max weight diff before vs after image A: {diff_before_vs_after_a}\n"
            f"max weight diff after image A vs after image B: {diff_after_a_vs_after_b}\n"
            f"max weight diff before vs after image B: {diff_before_vs_after_b}\n\n"
            f"mean |activation difference| across filters: {activation_diff.mean().item():.4f}\n"
            f"max |activation difference| (single filter): {activation_diff.max().item():.4f}\n"
            f"filters differing by > 1e-4: {n_filters_differ} / {len(activation_diff)}\n"
        )
    print(f"\nsaved {OUTPUTS / 'exp6_weight_invariance.txt'}")


if __name__ == "__main__":
    main()
