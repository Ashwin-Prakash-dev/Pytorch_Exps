---
name: ml-experiment-rigor
description: Rules for designing, running, and writing up ML experiments so conclusions are actually supported by the evidence. Use this skill whenever creating a new experiment directory, writing a train/sweep/eval script, comparing results across datasets, teachers, tasks, or architectures, or writing or editing an experiment README, results table, or "interpretation" section. Also use it for distillation, ablation, robustness, or scaling studies, and any time a result is about to be described as a finding, pattern, law, or structural property, even if the user doesn't mention rigor.
---

# ML Experiment Rigor

This repo runs a series of small, numbered experiments (`experiments/NN_name/`), each with scripts, committed outputs, and a README. Earlier experiments (05–11, function distillation) were well executed but over-interpreted. This skill exists so the next ones aren't.

The core principle: **the README may only claim what the design controls for.** Execution is cheap to fix; a narrative built on a confound is not.

Work through the four phases below in order. Don't write interpretation until Phase 3 is done.

---

## Phase 0: Before writing code

### 0.1 State the comparison and its confounds

Write a short `DESIGN.md` in the experiment directory before any script:

- **Question:** one sentence.
- **Independent variable:** the one thing that changes between conditions.
- **Everything else that also changes:** list it honestly. Dataset size, input dimension, number of classes, class balance, logit scale, teacher accuracy, optimizer steps, wall-clock budget.
- **For each item in that list:** how it is held fixed, or which control run removes it, or an explicit note that it is uncontrolled and therefore no claim will be made about it.

Why: in experiment 11, filtering CIFAR-10 to two classes cut the training set from 45k to 9k images. Epochs stayed at 10, so students got 5x fewer optimizer steps than in experiment 07. The README then concluded that difficulty tracks visual similarity "not dataset size," a variable it never controlled.

### 0.2 Write the prediction down first

Put the hypothesis and the result that would falsify it in `DESIGN.md` before running. If a class pair, dataset, or setting is chosen "because it should be easier/harder," record why, and do not later reuse the same pair to explain the opposite result.

Why: experiment 10 chose airplane-vs-automobile as visually distinct; experiment 11 explained a bad result by calling the same pair texturally similar. Both were post-hoc.

### 0.3 Prior-work check

For anything taking more than a day of compute or effort, spend under an hour searching (Google Scholar / Semantic Scholar, 3–4 phrasings, then the "cited by" list of the canonical paper). Add a `## Prior work` section to `DESIGN.md` with the closest 2–4 papers and one line on how this experiment differs. If there's no difference, say so and treat the experiment as a replication.

Known anchors for the distillation line: Ba & Caruana 2014; Urban et al. ICLR 2017; d'Ascoli et al. NeurIPS 2019; Neyshabur NeurIPS 2020; Bachmann et al. 2023 (Scaling MLPs). Verify any citation an LLM suggests before listing it.

---

## Phase 1: Design rules for scripts

### 1.1 Budget in optimizer steps, not epochs

Any comparison across datasets or filtered subsets must fix the **number of optimizer steps** (and batch size), not epochs. Expose `--steps` in `train.py`/`sweep.py`; derive epochs from it. Log `total_steps` and `samples_seen` in `results.csv`.

If the budget genuinely can't be matched, add a control that matches the other way (e.g. subsample the larger dataset to the smaller size) and report both.

### 1.2 Multiple seeds by default

- Default `--seeds 0 1 2` for every config that feeds a comparison. Single-seed runs are allowed for smoke tests only and must be labeled `exploratory` in results.
- `results.csv` stores one row per seed. A separate `results_summary.csv` holds mean, std, and n.
- Seed controls init, data order, and split where applicable. Log it.

### 1.3 Report convergence, not just final numbers

Log train and val loss per epoch or per N steps for every run. Before interpreting a gap, check whether the weaker model was still improving at the end. If train ≈ val and both are still falling, the model is **undertrained**, and the result says nothing about representational limits.

### 1.4 Required controls for cross-condition claims

If a claim is about the *function* learned by a teacher (not just the task):

- **Direct-training baseline:** train each student architecture on ground-truth labels with the same step budget. Distillation only reveals something about the teacher's function if its gap differs from the direct-training gap.
- **Architecture-matched teacher control:** if claiming a CNN's function needs convolutional structure, also distill from an MLP teacher trained to similar accuracy on the same task.

If these aren't run, the README must not claim anything about the teacher's function versus the task.

### 1.5 Keep ablations complete

Parameter/FLOP-matched pairs (as in the compositional vs. plain grid) are good; keep them. Always report the **full grid** in the README table, same rows in every experiment, not a hand-picked subset.

---

## Phase 2: Metrics

### 2.1 Cross-condition metrics must be scale- and balance-invariant

When comparing across teachers, datasets, or tasks:

- **Use** variance-normalized error: `1 - R²` against the teacher-logit distribution on the evaluation split (MSE / Var(teacher logits)). Report it for every model.
- **Do not compare** raw MSE across conditions. Logit scale differs.
- **Do not compare** teacher-agreement across tasks with different class balance. On an imbalanced task most inputs are confident negatives far from the boundary, so agreement is inflated. If agreement is reported, also report the **near-boundary agreement** (e.g. inputs where |teacher logit| < some threshold, with the threshold stated) and the trivial baseline (linear model, always-majority).
- **Do not compare** "recall retained" or accuracy against baselines that differ between balanced and imbalanced tasks. Use balanced accuracy or report the baseline next to every number.

Raw MSE and agreement are fine *within* a single experiment.

### 2.2 Uncertainty on every comparison

Any table comparing models shows `mean ± std (n seeds)`. A difference is called a difference only if it exceeds roughly 2× the pooled std, or a stated test (paired across seeds) says so. Otherwise write "no clear difference at n=3."

Never write "statistically indistinguishable" or "significant" without having computed something.

### 2.3 Robustness: compare degradation ratios

For perturbation tests, report `perturbed_error / clean_error` per model alongside absolute error, across seeds and at least 2 perturbation seeds. A model "not winning outright" on one perturbation is not an exception pattern; a consistently worse degradation ratio is.

### 2.4 Worst-case analysis needs a baseline expectation

Before calling a worst-case pattern notable, state what you'd expect by construction. On a balanced task with high error, the top-|error| examples will naturally include confident classification flips. Compare against a random or median-error sample before calling anything "unprecedented."

---

## Phase 3: Pre-interpretation check

Before writing any `## What to take away` or `## Scientific interpretation` section, write a `## Validity check` section in the README answering:

1. What changed between conditions besides the independent variable? Which of those are controlled, which aren't?
2. Were optimizer steps matched? Were weak models converged?
3. How many seeds? Do the key differences survive the std?
4. Are cross-condition numbers in 1 − R² (or another normalized metric)?
5. Were the controls in 1.4 run? If not, which claims are therefore off-limits?
6. Is any explanation reusing a property of the setup that an earlier README described the opposite way?
7. Would a skeptical reader's most obvious alternative explanation be ruled out by the data shown?

If any answer is "no" or "unknown," the corresponding claim is either dropped or written as an open hypothesis with the specific run that would test it.

---

## Phase 4: Writing the README

### 4.1 Claim strength must match evidence

| Evidence | Allowed wording |
|---|---|
| 1 seed, 1 condition | "In this run…", "suggests" |
| ≥3 seeds, 1 condition, gap > 2 std | "X outperforms Y here" |
| ≥3 seeds, replicated across ≥2 conditions with matched budgets | "consistently", "replicates" |
| Above plus mechanism test (control run isolating the cause) | "because", "driven by" |

Banned without the last row's evidence: "structural property", "law", "the real driver is", "overturns", "unprecedented", "the most reliable finding", "N for N" counts built on single seeds.

### 4.2 Explanations are hypotheses until tested

Any causal story (visual similarity, boundary cleanliness, noise compounding) goes under `## Open hypotheses`, each paired with the concrete run that would falsify it. It does not go in the headline or the takeaway.

### 4.3 Arguments must be valid on their own terms

Don't use comparisons that can't support the claim. Example: "compositional (497k params) beats conventional (117k params)" is not evidence for composition, because the bigger model won. Use the parameter-matched grid instead.

### 4.4 Keep interpretation short

Target: the interpretation section is shorter than the results section. Results tables, validity check, open hypotheses, next runs. No narrative arcs.

### 4.5 Headline format

The top-of-README summary states: the comparison, the budget-matching method, seed count, the main normalized number with std, and the biggest uncontrolled confound. Nothing more.

---

## Quick checklist (copy into each new experiment)

```
[ ] DESIGN.md: question, independent variable, confound list, prediction, prior work
[ ] Step budget fixed across conditions (or subsampling control run)
[ ] ≥3 seeds; results.csv per seed + results_summary.csv
[ ] Train/val curves logged; convergence checked for weakest models
[ ] Direct-training baseline (if claiming anything about a teacher's function)
[ ] Architecture-matched teacher control (if claiming inductive bias is needed)
[ ] Cross-condition metrics in 1 − R²; agreement only within-condition or near-boundary
[ ] mean ± std in every comparison table; full ablation grid shown
[ ] Robustness as degradation ratios, multiple perturbation seeds
[ ] Worst-case compared against a random/median sample
[ ] README Validity check written before interpretation
[ ] Claim wording matches the evidence table in 4.1
[ ] Causal stories live in Open hypotheses with a falsifying run
```

## When the user asks to skip steps

Speed matters and quick exploratory runs are fine. If asked to skip seeds, controls, or budget matching, do it, but label the run `exploratory` in `results.csv` and the README header, and restrict the README to describing what happened, with no cross-condition or causal claims.