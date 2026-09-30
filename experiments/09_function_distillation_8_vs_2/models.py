"""Student architectures that try to reproduce the frozen CNN's logit function
from a flat 784-dim input -- no convolution, no locality, no weight sharing,
no pooling. Every model here takes (batch, 784) and returns (batch,) -- one
predicted logit, directly comparable to the teacher's.

Five families, matching the experiment design:
  1. Linear            -- is the mapping linearly separable at all?
  2. ConventionalMLP    -- the textbook two-hidden-layer MLP baseline.
  3. DeepPlainMLP       -- a plain (non-residual) stack of N*3+1 linear layers,
                           parameter-matched to CompositionalReLUMLP(H, N) --
                           isolates whether residual *structure* matters, not
                           just depth/params.
  4. CompositionalReLUMLP -- N repeated [Linear->ReLU]x3 blocks with a residual
                           skip around each block.
  5. ParallelReLUBank   -- many independent [Linear(784->W)->ReLU] branches,
                           concatenated and linearly combined. Nonlinear
                           *feature generation* without depth.

Every model exposes `.n_params`, `.n_linear_layers`, `.n_relu_layers` and
`estimate_flops(model)` below computes an approximate forward-pass FLOP count,
so architectures can be compared at matched capacity, not just by name.
"""

from __future__ import annotations

import torch
from torch import nn

INPUT_DIM = 784


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
    """Baseline 1: 784 -> 1, no nonlinearity at all."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(INPUT_DIM, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class ConventionalMLP(_StudentBase):
    """Baseline 2: 784 -> H -> ReLU -> H -> ReLU -> 1."""

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
    residual connections. `n_layers` is chosen by the sweep to match the
    parameter count of a CompositionalReLUMLP(H, N) -- same width, same
    total depth, same total linear-layer count, only the skip connections
    (and their implied identity-function prior) are missing.
    """

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
    """Model 4: Linear(784->H), then N residual blocks of 3x[Linear->ReLU],
    then Linear(H->1). The architecture under test: many stacked simple
    nonlinear transformations, composed rather than widened."""

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
    """Model 5: `n_branches` independent Linear(784->W)->ReLU branches,
    concatenated and linearly combined. Nonlinear feature *generation*
    (many diverse basis functions evaluated in parallel) instead of
    nonlinear *composition* (functions of functions)."""

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
    per nn.Linear, summed over the model. ReLU cost is negligible and
    omitted. This undercounts a little (e.g. the branch-concat / residual
    adds are O(H), dwarfed by the matmuls) but is consistent across models,
    which is what matters for a fair comparison."""
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
    # Quick sanity check: every architecture builds, runs, and reports sane
    # capacity numbers.
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
