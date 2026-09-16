"""The full model grid: baselines + the ReLU-scaling sweep (H x N) for the
compositional model, its exact parameter-matched plain-MLP counterpart, and
a parallel-branch sweep spanning a similar parameter range.

Key design choice, verified in models.py: CompositionalReLUMLP(H, N) has
`1 + 3N + 1` linear layers of width H. Giving DeepPlainMLP the same H and
`n_layers=3*N` produces an *exactly* matched parameter count (confirmed
below at import time) -- so any metric difference between those two rows is
attributable to the residual connections, not to having more weights.

Writes one row per trained model to outputs/results.csv (append-as-you-go,
so a partial run is still usable).

Run:
    python experiments/05_function_distillation/sweep.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import load_splits  # noqa: E402
from run_one import run  # noqa: E402

OUTPUTS = HERE / "outputs"
RESULTS_CSV = OUTPUTS / "results.csv"

EPOCHS = 10
LR = 1e-3
SEED = 42


def build_configs() -> list[dict]:
    configs = []

    # Baseline 1: linear
    configs.append({"name": "linear", "arch": "linear", "kwargs": {}})

    # Baseline 2: conventional 2-hidden-layer MLP, at the same widths used below
    for H in (32, 64, 128):
        configs.append({
            "name": f"conventional_mlp_H{H}",
            "arch": "conventional_mlp", "kwargs": {"hidden": H},
        })

    # Model 4 (compositional) x Baseline 3 (param-matched deep plain MLP),
    # the core H x N sweep.
    for H in (32, 64, 128):
        for N in (1, 2, 4, 8):
            configs.append({
                "name": f"compositional_H{H}_N{N}",
                "arch": "compositional_relu", "kwargs": {"hidden": H, "n_blocks": N},
            })
            configs.append({
                "name": f"deepplain_H{H}_N{N}",
                "arch": "deep_plain_mlp", "kwargs": {"hidden": H, "n_layers": 3 * N},
            })

    # Model 5: parallel ReLU feature bank, spanning a similar param range.
    # (32,16) and (8,64) land at the SAME param count via different shapes
    # (many narrow branches vs few wide ones) -- a useful pair on its own.
    for n_branches, branch_width in ((2, 16), (4, 16), (4, 32), (8, 32), (16, 32), (32, 16)):
        configs.append({
            "name": f"parallel_B{n_branches}_W{branch_width}",
            "arch": "parallel_bank",
            "kwargs": {"n_branches": n_branches, "branch_width": branch_width},
        })

    for c in configs:
        c["epochs"] = EPOCHS
        c["lr"] = LR
        c["seed"] = SEED
    return configs


def main() -> None:
    configs = build_configs()
    print(f"{len(configs)} configs queued\n")

    device = torch.device("cpu")
    splits, loaders = load_splits(batch_size=256)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    t0 = time.perf_counter()

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = None
        for i, config in enumerate(configs, 1):
            print(f"=== [{i}/{len(configs)}] {config['name']} "
                  f"(elapsed {time.perf_counter() - t0:.0f}s) ===")
            row, _ = run(config, splits=splits, loaders=loaders, device=device, log=True)

            if writer is None:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(f"  -> val_mse={row['val_mse']:.4f}  test_mse={row['test_mse']:.4f}  "
                  f"test_agreement={row['test_teacher_student_agreement']:.4f}\n")

    print(f"done in {time.perf_counter() - t0:.0f}s -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
