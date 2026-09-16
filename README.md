# PyTorch Experiments

A growing collection of small, self-contained PyTorch experiments — built to
**understand the mechanics deeply**, not just to make things train. Each
experiment spells out its own model and training loop (no hidden framework
magic) and ships with results you can see.

## Philosophy

- **Explicit over abstract.** Every experiment shows the full training loop. The
  shared `utils/` package holds only true boilerplate (seeding, device, plotting).
- **One idea per experiment.** Each builds on the last, ideally exposing a
  limitation that motivates the next.
- **Reproducible & visible.** Fixed seeds; every run saves plots to `outputs/`
  so results render right here on GitHub.

## Structure

```
PytorchExperiments/
├── utils/                      # thin shared helpers (seed, device, plotting)
├── experiments/
│   ├── 01_linear_regression/   # each: train.py + README.md + outputs/
│   ├── 02_sine_wave_linear/    #   (committed plots so results show on GitHub)
│   ├── 03_sine_wave_mlp/
│   └── 04_mnist_cnn_digit8/
├── notebooks/                  # interactive companions (inline plots + prose)
│   ├── 01_linear_regression.ipynb
│   ├── 02_sine_wave_linear_vs_mlp.ipynb
│   └── 04_mnist_cnn_digit8.ipynb
├── requirements.txt
└── README.md
```

The **script** is the canonical, clean version; the **notebook** is the
interactive companion for exploring the same idea cell by cell.

## Setup

This repo uses **Python 3.13** with `torch`, `numpy`, `matplotlib`.

```bash
pip install -r requirements.txt
```

To run the notebooks, register your torch-equipped interpreter as a Jupyter
kernel once (the notebooks expect a kernel named `pytorch-experiments`):

```bash
python -m ipykernel install --user --name pytorch-experiments \
    --display-name "Python (PytorchExperiments)"
```

> Heads-up: if you have multiple Python installs, make sure Jupyter's kernel and
> your `python` point at the **same** environment — otherwise the notebook won't
> find `torch`.

## Running

```bash
# Scripts (run from the repo root)
python experiments/01_linear_regression/train.py
python experiments/01_linear_regression/train.py --epochs 400 --lr 0.1 --noise 0.5

# Notebooks
jupyter lab        # then open notebooks/01_linear_regression.ipynb
```

## Experiments

| # | Experiment | Model | Status | Key idea |
|---|------------|-------|--------|----------|
| 01 | [Linear regression](experiments/01_linear_regression/) | `Linear(1,1)` | ✅ done | The 5-step training loop; a linear layer *is* regression; the noise floor |
| 02 | [Sine wave — linear layer](experiments/02_sine_wave_linear/) | `Linear(1,1)` | ✅ done | A linear model **fails** on nonlinear data (loss ~0.41) — motivates hidden layers |
| 03 | [Sine wave — MLP](experiments/03_sine_wave_mlp/) | `Linear → ReLU → Linear` | ✅ done | A hidden layer + ReLU fits the curve (loss ~0.01); ReLU = piecewise-linear |
| 04 | [MNIST CNN — is this an 8?](experiments/04_mnist_cnn_digit8/) | `Conv → ReLU → MaxPool ×2 → Linear` | ✅ done | First real dataset + CNN + classification; 99.1% test accuracy vs. a 90.3% do-nothing baseline — precision/recall catch what accuracy alone would hide |

Experiments 02 & 03 share one [comparison notebook](notebooks/02_sine_wave_linear_vs_mlp.ipynb) that runs both side by side — the clearest way to *see* why nonlinearity matters.

## Roadmap

The natural progression from here:

1. ~~**Nonlinearity matters** — sine wave with vs. without a hidden layer (02 → 03).~~ ✅
2. **Optimization** — SGD vs. Adam, learning-rate effects, mini-batching with
   `DataLoader`.
3. **Generalization** — overfitting, train/val gaps, regularization.
4. ~~**Scaling up** — a real dataset (MNIST) and a CNN (04).~~ ✅ Next: the
   full 10-way digit classifier.
