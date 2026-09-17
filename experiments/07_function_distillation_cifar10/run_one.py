"""Train and evaluate a single student config against the frozen CIFAR-10
airplane CNN's logit function. Used both as a standalone CLI and imported by
sweep.py so every run goes through identical code.

Run generate_teacher_dataset.py once before this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from common import count_params, function_error, load_splits, task_agreement, train_student  # noqa: E402
from models import build_model, estimate_flops  # noqa: E402
from utils import set_seed  # noqa: E402

CHECKPOINTS = HERE / "outputs" / "checkpoints"


def run(config: dict, splits=None, loaders=None, device: torch.device | None = None, save_checkpoint: bool = True, log: bool = True) -> dict:
    """config = {name, arch, kwargs, epochs, lr, seed, batch_size}"""
    set_seed(config.get("seed", 42))
    device = device or torch.device("cpu")

    if splits is None or loaders is None:
        splits, loaders = load_splits(batch_size=config.get("batch_size", 256))

    model = build_model(config["arch"], **config.get("kwargs", {}))
    if log:
        print(f"[{config['name']}] arch={config['arch']} kwargs={config.get('kwargs', {})} params={count_params(model)}")

    history = train_student(
        model, loaders["train"], loaders["val"],
        epochs=config.get("epochs", 8), lr=config.get("lr", 1e-3),
        device=device, log=log,
    )

    ALL_CFG_KEYS = ("hidden", "n_blocks", "n_layers", "n_branches", "branch_width")
    kwargs = config.get("kwargs", {})
    row = {
        "name": config["name"],
        "arch": config["arch"],
        **{f"cfg_{k}": kwargs.get(k, "") for k in ALL_CFG_KEYS},
        "n_params": count_params(model),
        "n_linear_layers": model.n_linear_layers,
        "n_relu_layers": model.n_relu_layers,
        "flops_per_sample": estimate_flops(model, batch=1),
        "epochs": config.get("epochs", 8),
        "train_time_sec": history["train_time_sec"],
    }

    for split_name in ("train", "val", "test"):
        err = function_error(model, loaders[split_name], device)
        agree = task_agreement(model, splits[split_name], device)
        for k, v in err.items():
            row[f"{split_name}_{k}"] = v
        for k, v in agree.items():
            row[f"{split_name}_{k}"] = v

    if save_checkpoint:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CHECKPOINTS / f"{config['name']}.pt")

    return row, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=[
        "linear", "conventional_mlp", "deep_plain_mlp", "compositional_relu", "parallel_bank",
    ])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-branches", type=int, default=8)
    parser.add_argument("--branch-width", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    kwargs_by_arch = {
        "linear": {},
        "conventional_mlp": {"hidden": args.hidden},
        "deep_plain_mlp": {"hidden": args.hidden, "n_layers": args.n_layers},
        "compositional_relu": {"hidden": args.hidden, "n_blocks": args.n_blocks},
        "parallel_bank": {"n_branches": args.n_branches, "branch_width": args.branch_width},
    }
    config = {
        "name": args.name or args.arch,
        "arch": args.arch,
        "kwargs": kwargs_by_arch[args.arch],
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
    }

    row, _ = run(config)
    print()
    for k, v in row.items():
        print(f"{k:32s} {v}")


if __name__ == "__main__":
    main()
