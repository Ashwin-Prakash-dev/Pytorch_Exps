"""Student architectures that try to reproduce the frozen CIFAR-10 airplane
CNN's logit function from a flat 3072-dim input (32x32x3 flattened) -- no
convolution, no locality, no weight sharing, no pooling. Every model here
takes (batch, 3072) and returns (batch,) -- one predicted logit, directly
comparable to the teacher's.

Identical architecture family to experiment 05's models.py (see that file
for the full rationale of each). The only real difference is INPUT_DIM:
3072 instead of 784, since CIFAR-10 is RGB 32x32 rather than grayscale
28x28 -- almost 4x the input dimensionality, which is why every model here
has substantially more parameters than its experiment-05 namesake at the
same H.
"""

from __future__ import annotations

import torch
from torch import nn

INPUT_DIM = 3 * 32 * 32  # 3072


class _StudentBase(nn.Module):
    """Adds capacity-accounting helpers shared by every student model."""

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_linear_layers(self) -> int:
        return sum(1 for m in self.modules() if isinstance(m, nn.Linear))

    @property
    def n_relu_layers(self) -> int:
        return sum(1 for m in self.modules() if isinstance(m, nn.ReLU))


class LinearModel(_StudentBase):
    """Baseline 1: 3072 -> 1, no nonlinearity at all."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(INPUT_DIM, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class ConventionalMLP(_StudentBase):
    """Baseline 2: 3072 -> H -> ReLU -> H -> ReLU -> 1."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DeepPlainMLP(_StudentBase):
    """Baseline 3: a plain stack of `n_layers` H->H linear+ReLU layers, no
    residual connections -- parameter-matched to CompositionalReLUMLP(H, N)
    via n_layers=3*N (same width, same total linear-layer count)."""

    def __init__(self, hidden: int = 64, n_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, hidden)
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.body(h)
        return self.output(h).squeeze(-1)


class ResidualReLUBlock(nn.Module):
    """Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU, then + input."""

    def __init__(self, hidden: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CompositionalReLUMLP(_StudentBase):
    """Model 4: Linear(3072->H), then N residual blocks of 3x[Linear->ReLU],
    then Linear(H->1)."""

    def __init__(self, hidden: int = 64, n_blocks: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, hidden)
        self.blocks = nn.Sequential(*[ResidualReLUBlock(hidden) for _ in range(n_blocks)])
        self.output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.output(h).squeeze(-1)


class ParallelReLUBank(_StudentBase):
    """Model 5: `n_branches` independent Linear(3072->W)->ReLU branches,
    concatenated and linearly combined."""

    def __init__(self, n_branches: int = 8, branch_width: int = 16):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(INPUT_DIM, branch_width), nn.ReLU())
            for _ in range(n_branches)
        ])
        self.output = nn.Linear(n_branches * branch_width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([branch(x) for branch in self.branches], dim=-1)
        return self.output(feats).squeeze(-1)


def estimate_flops(model: nn.Module, batch: int = 1) -> int:
    """Approximate forward-pass FLOPs: 2*in*out (multiply-add) + out (bias)
    per nn.Linear, summed over the model."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += 2 * m.in_features * m.out_features + m.out_features
    return total * batch


def build_model(name: str, **kwargs) -> _StudentBase:
    registry = {
        "linear": LinearModel,
        "conventional_mlp": ConventionalMLP,
        "deep_plain_mlp": DeepPlainMLP,
        "compositional_relu": CompositionalReLUMLP,
        "parallel_bank": ParallelReLUBank,
    }
    if name not in registry:
        raise ValueError(f"unknown model '{name}', choose from {list(registry)}")
    return registry[name](**kwargs)


if __name__ == "__main__":
    x = torch.randn(4, INPUT_DIM)
    configs = [
        ("linear", {}),
        ("conventional_mlp", {"hidden": 64}),
        ("deep_plain_mlp", {"hidden": 64, "n_layers": 4}),
        ("compositional_relu", {"hidden": 64, "n_blocks": 4}),
        ("parallel_bank", {"n_branches": 8, "branch_width": 16}),
    ]
    for name, kwargs in configs:
        model = build_model(name, **kwargs)
        out = model(x)
        assert out.shape == (4,), out.shape
        print(
            f"{name:20s} {kwargs} -> params={model.n_params:7d}  "
            f"linear_layers={model.n_linear_layers:3d}  relu_layers={model.n_relu_layers:3d}  "
            f"flops={estimate_flops(model):9d}"
        )
