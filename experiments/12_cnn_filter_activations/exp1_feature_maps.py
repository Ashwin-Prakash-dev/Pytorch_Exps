"""Experiment 1 -- visualize feature maps for representative images from
each class.

For one image per digit (0-9), shows the input alongside every conv1
feature map (8 channels, 28x28) and every conv2 feature map (16 channels,
14x14), each labeled by layer and filter index.

Run:
    python experiments/12_cnn_filter_activations/exp1_feature_maps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from activation_utils import ActivationCapture  # noqa: E402
from model import setup  # noqa: E402

OUTPUTS = HERE / "outputs" / "feature_maps"


def plot_digit_feature_maps(model, device, image, label, out_path):
    with ActivationCapture(model) as cap:
        acts = cap.run(image.unsqueeze(0).to(device))
    conv1_maps = acts["conv1"].squeeze(0).cpu()   # (8, 28, 28)
    conv2_maps = acts["conv2"].squeeze(0).cpu()   # (16, 14, 14)

    # conv1: input + its 8 feature maps, one row.
    n_cols = 9
    fig1, axes1 = plt.subplots(1, n_cols, figsize=(2 * n_cols, 2.4))
    axes1[0].imshow(image.squeeze(0), cmap="gray")
    axes1[0].set_title(f"input\nlabel={label}", fontsize=9)
    axes1[0].axis("off")
    for i in range(8):
        axes1[i + 1].imshow(conv1_maps[i], cmap="viridis")
        axes1[i + 1].set_title(f"conv1 f{i}\n(28x28)", fontsize=8)
        axes1[i + 1].axis("off")
    fig1.suptitle(f"digit {label}: input + conv1 feature maps (post ReLU)")
    fig1.tight_layout()
    fig1.savefig(out_path, dpi=120)
    plt.close(fig1)

    # conv2: 16 feature maps, its own grid.
    fig2, axes2 = plt.subplots(2, 8, figsize=(16, 4.4))
    for i in range(16):
        r, c = divmod(i, 8)
        axes2[r, c].imshow(conv2_maps[i], cmap="viridis")
        axes2[r, c].set_title(f"conv2 f{i} (14x14)", fontsize=8)
        axes2[r, c].axis("off")
    fig2.suptitle(f"digit {label}: conv2 feature maps (post ReLU)")
    fig2.tight_layout()
    fig2.savefig(str(out_path).replace("_conv1.png", "_conv2.png"), dpi=120)
    plt.close(fig2)


def main() -> None:
    model, device, test_set = setup()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    seen = set()
    for idx in range(len(test_set)):
        image, label = test_set[idx]
        if label in seen:
            continue
        seen.add(label)
        out_path = OUTPUTS / f"digit_{label}_conv1.png"
        plot_digit_feature_maps(model, device, image, label, out_path)
        print(f"saved feature maps for digit {label} (test index {idx}) -> "
              f"{out_path.name} + {out_path.name.replace('_conv1', '_conv2')}")
        if len(seen) == 10:
            break

    print(f"\nall feature-map grids saved to {OUTPUTS}")


if __name__ == "__main__":
    main()
