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
│   ├── 04_mnist_cnn_digit8/
│   ├── 05_function_distillation/
│   ├── 06_cifar10_cnn_airplane/
│   ├── 07_function_distillation_cifar10/
│   ├── 08_mnist_cnn_8_vs_2/
│   ├── 09_function_distillation_8_vs_2/
│   ├── 10_cifar10_cnn_airplane_vs_automobile/
│   ├── 11_function_distillation_airplane_vs_automobile/
│   └── 12_cnn_filter_activations/
├── notebooks/                  # interactive companions (inline plots + prose)
│   ├── 01_linear_regression.ipynb
│   ├── 02_sine_wave_linear_vs_mlp.ipynb
│   ├── 04_mnist_cnn_digit8.ipynb
│   ├── 05_function_distillation_analysis.ipynb
│   ├── 06_cifar10_cnn_airplane.ipynb
│   ├── 07_function_distillation_cifar10_analysis.ipynb
│   ├── 08_mnist_cnn_8_vs_2.ipynb
│   ├── 09_function_distillation_8_vs_2_analysis.ipynb
│   ├── 10_cifar10_cnn_airplane_vs_automobile.ipynb
│   ├── 11_function_distillation_airplane_vs_automobile_analysis.ipynb
│   └── 12_cnn_filter_activations.ipynb
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
| 05 | [Function distillation — CNN → MLP](experiments/05_function_distillation/) | 5 ReLU-MLP architectures vs. a frozen CNN | ✅ done | Not classification — a *function-representation* study: can flat MLPs (no conv, no pooling) learn to reproduce a trained CNN's logit function? Residual composition beats parameter-matched plain depth and parallel width; a plain (non-residual) stack even gets *worse* past 4 blocks — a toy-scale echo of the ResNet degradation problem |
| 06 | [CIFAR-10 CNN — is this an airplane?](experiments/06_cifar10_cnn_airplane/) | `Conv → ReLU → MaxPool ×3 → Linear` | ✅ done | The RGB, harder-dataset sequel to 04; 94.9% test accuracy vs. a 90.0% baseline — real photographs are a genuinely harder gap to close than clean digits |
| 07 | [Function distillation, repeated — CIFAR-10](experiments/07_function_distillation_cifar10/) | Same 5 architectures vs. experiment 06's frozen CNN | ✅ done | Experiment 05's study replicated on a harder, unrelated teacher: the same residual-beats-plain and parallel-underperforms findings hold up (and sharpen), but a real, honest difference emerges — earlier diminishing returns, a much sharper ground-truth recall collapse for small students, and more genuine classification errors in the worst-case analysis |
| 08 | [MNIST CNN — 8 vs. 2](experiments/08_mnist_cnn_8_vs_2/) | Same CNN as 04, filtered data | ✅ done | Removes 04's imbalance shortcut entirely (baseline drops 90.3%→51.4%) by comparing two specific digits instead of one-vs-rest; the CNN reaches 99.2% accuracy — balance didn't make it *harder*, just made accuracy actually mean something |
| 09 | [Function distillation, repeated — MNIST 8-vs-2](experiments/09_function_distillation_8_vs_2/) | Same 5 architectures vs. experiment 08's frozen CNN | ✅ done | A third replication: composition still beats plain depth and parallel width. But the big finding is new — this balanced, single-boundary function is **3x easier to distill** than experiment 05's, and a 26k-param student recovers 98% of the teacher's accuracy (vs. experiment 07's severe recall collapse) |
| 10 | [CIFAR-10 CNN — airplane vs. automobile](experiments/10_cifar10_cnn_airplane_vs_automobile/) | Same CNN as 06, filtered data | ✅ done | CIFAR-10's counterpart to 08: baseline drops 90.0%→50.0%, and removing visually-confusing negatives (birds, ships) lifts recall from 0.57→0.94, a far bigger jump than accuracy alone shows |
| 11 | [Function distillation, repeated — CIFAR-10 airplane-vs-automobile](experiments/11_function_distillation_airplane_vs_automobile/) | Same 5 architectures vs. experiment 10's frozen CNN | ✅ done | **The corrective finding of the series.** Where balancing MNIST (05→09) made distillation dramatically easier, balancing CIFAR-10 to this specific pair (07→11) makes it *harder* — worse MSE, worse agreement, the plain model suffers outright training collapse at depth, and for the first time every single worst-case error is a genuine misclassification. Balance isn't the driver; visual/semantic closeness between the compared classes is |
| 12 | [Do different inputs activate different CNN filters?](experiments/12_cnn_filter_activations/) | 10-way MNIST CNN, inspected not trained-on | ✅ done | An interpretability study, not a classification one: mechanically confirms weights stay invariant across inputs while activations don't, then quantifies *how* — every live filter's activation depends on class (ANOVA, p<0.001, all 23 non-dead filters), and same-class images share 70% of their top-8 active filters vs. 56% for different classes (p≈5×10⁻¹⁹⁵). But classes share the *majority* of filters, not disjoint sets — filter specialization, not dynamic routing |

Experiments 02 & 03 share one [comparison notebook](notebooks/02_sine_wave_linear_vs_mlp.ipynb) that runs both side by side — the clearest way to *see* why nonlinearity matters.

Experiments 05, 07, 09, and 11 are one connected study, not four separate
ones — four frozen-CNN teachers (MNIST imbalanced/balanced, CIFAR-10
imbalanced/balanced) probing the same question ("can a flat ReLU-MLP
represent a CNN's function, and what makes that easier or harder?") from
different angles. Two findings replicate across all four: residual
composition beats parameter-matched plain depth and parallel width, and
compositional's advantage specifically shrinks under Gaussian noise
(4-for-4). One naive generalization gets built and then corrected:
experiment 09 alone would suggest "balanced tasks distill better,"
but experiment 11 shows that's really "visually/texturally close compared
classes distill worse," which balance only sometimes correlates with.

## Roadmap

The natural progression from here:

1. ~~**Nonlinearity matters** — sine wave with vs. without a hidden layer (02 → 03).~~ ✅
2. **Optimization** — SGD vs. Adam, learning-rate effects, mini-batching with
   `DataLoader`.
3. **Generalization** — overfitting, train/val gaps, regularization.
4. ~~**Scaling up** — a real dataset (MNIST) and a CNN (04).~~ ✅
5. ~~**Function representation** — can a non-convolutional architecture learn
   a CNN's function (05)?~~ ✅
6. ~~**A harder teacher, then a balanced one** — repeat 04-05 on CIFAR-10
   (06-07), then on genuinely balanced class pairs for both datasets
   (08-11).~~ ✅
7. ~~**Look inside the network** — do different inputs activate different
   filters, distinct from the weights themselves changing (12)?~~ ✅ Next:
   an ablation to move from correlational (12's filter-class associations)
   to causal, or a visually-close-but-balanced CIFAR-10 pair (e.g.
   automobile vs. truck) to isolate visual closeness from balance directly.
