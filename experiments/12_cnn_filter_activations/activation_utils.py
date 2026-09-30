"""Shared activation-extraction helpers for every exp*.py script in this
experiment.

Terminology used consistently throughout (see the README for the full
distinction): a **filter** (equivalently, a convolutional *channel*) is a
fixed, learned kernel -- `model.conv1.weight[i]` never changes between
images. A **feature map** or **activation map** is what that filter
produces for a *specific* input -- `conv1_activations[i]`, a 2D tensor that
is different for every image. An **activation score** is a single scalar
summarizing one feature map (its mean, max, or the fraction of its
locations that are nonzero).

Every filter in this network has a stable, global index: filters 0-7 are
conv1's 8 filters, filters 8-23 are conv2's 16 filters. `FILTER_LAYER` and
`FILTER_INDEX_IN_LAYER` map a global filter index back to its layer.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

N_CONV1_FILTERS = 8
N_CONV2_FILTERS = 16
N_FILTERS = N_CONV1_FILTERS + N_CONV2_FILTERS  # 24

FILTER_LAYER = ["conv1"] * N_CONV1_FILTERS + ["conv2"] * N_CONV2_FILTERS
FILTER_INDEX_IN_LAYER = list(range(N_CONV1_FILTERS)) + list(range(N_CONV2_FILTERS))


class ActivationCapture:
    """Registers forward hooks on `model.relu1` and `model.relu2` (the
    post-conv, post-ReLU feature maps) and stores whatever the most recent
    forward pass produced. A context manager so hooks are always removed.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: dict[str, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        self._handles.append(self.model.relu1.register_forward_hook(self._make_hook("conv1")))
        self._handles.append(self.model.relu2.register_forward_hook(self._make_hook("conv2")))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self.activations[name] = output.detach()
        return hook

    @torch.no_grad()
    def run(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (batch, 1, 28, 28). Returns {"conv1": (batch,8,28,28), "conv2": (batch,16,14,14)}."""
        self.model(x)
        return dict(self.activations)


@torch.no_grad()
def filter_scores(activations: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reduce each image's feature maps to one scalar per filter (the mean
    activation across spatial locations), concatenated conv1 then conv2.
    Returns (batch, N_FILTERS)."""
    conv1_mean = activations["conv1"].mean(dim=(2, 3))  # (batch, 8)
    conv2_mean = activations["conv2"].mean(dim=(2, 3))  # (batch, 16)
    return torch.cat([conv1_mean, conv2_mean], dim=1)


@torch.no_grad()
def filter_max_scores(activations: dict[str, torch.Tensor]) -> torch.Tensor:
    conv1_max = activations["conv1"].amax(dim=(2, 3))
    conv2_max = activations["conv2"].amax(dim=(2, 3))
    return torch.cat([conv1_max, conv2_max], dim=1)


@torch.no_grad()
def filter_active_fraction(activations: dict[str, torch.Tensor]) -> torch.Tensor:
    """Fraction of spatial locations with activation > 0 (post-ReLU, so
    this is exactly the fraction NOT zeroed by the ReLU)."""
    conv1_frac = (activations["conv1"] > 0).float().mean(dim=(2, 3))
    conv2_frac = (activations["conv2"] > 0).float().mean(dim=(2, 3))
    return torch.cat([conv1_frac, conv2_frac], dim=1)


@torch.no_grad()
def batched_filter_scores(model: nn.Module, dataset: Dataset, indices, device, batch_size: int = 256):
    """Run `dataset[indices]` through the model in batches, returning
    (mean_scores, max_scores, active_fraction, labels) each shaped
    (len(indices), ...) / (len(indices),).
    """
    means, maxes, fracs, labels = [], [], [], []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        images = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
        batch_labels = torch.tensor([dataset[i][1] for i in batch_idx])

        with ActivationCapture(model) as cap:
            acts = cap.run(images)
        means.append(filter_scores(acts).cpu())
        maxes.append(filter_max_scores(acts).cpu())
        fracs.append(filter_active_fraction(acts).cpu())
        labels.append(batch_labels)

    return torch.cat(means), torch.cat(maxes), torch.cat(fracs), torch.cat(labels)


def filter_label(global_idx: int) -> str:
    return f"{FILTER_LAYER[global_idx]}_f{FILTER_INDEX_IN_LAYER[global_idx]}"
