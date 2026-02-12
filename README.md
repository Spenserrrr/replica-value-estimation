# Replica Trick Estimators for V* (Offline Estimation)

Implementation of replica trick-based estimators for offline estimation of V* in the A*PO algorithm framework.

## Overview

This repository contains the implementation and experiments for my senior thesis on **replica trick estimators** for offline value function estimation under KL-regularized control. The code implements and compares four estimator families:

1. **LME (Log-Mean-Exp)** — Baseline biased estimator
2. **Single-Replica** — Fixed replica order n > 1
3. **Multi-n Slope** — Linear extrapolation across multiple replica orders
4. **Multi-n Slope + Jackknife** — Slope estimator with delete-one jackknife bias correction

## Repository Structure

```
.
├── src/
│   ├── __init__.py            # Package marker
│   ├── estimators.py          # Core V* estimators (distribution-agnostic)
│   ├── ground_truth.py        # Analytical V* for Gaussian and Bernoulli
│   ├── metrics.py             # Bias, variance, RMSE metrics
│   ├── experiment_runner.py   # Generic Monte Carlo experiment runner
│   └── plotting.py            # Triptych plot generation
├── run_experiment1_gaussian.py    # Experiment 1a: Gaussian rewards
├── run_experiment1_bernoulli.py   # Experiment 1b: Bernoulli rewards
├── results/
│   ├── gaussian/
│   │   └── 2026-02-11_21-34-10/  # Timestamped run
│   │       ├── config.json       # Full config snapshot
│   │       ├── results.csv       # Numerical results
│   │       └── *.png             # Triptych plots
│   └── bernoulli/
│       └── 2026-02-11_21-35-00/
│           ├── config.json
│           ├── results.csv
│           ├── p0.01/            # Plots for p=0.01
│           ├── p0.05/
│           └── ...
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages: `numpy`, `scipy`, `pandas`, `matplotlib`

## Running Experiments

Each run creates a new **timestamped directory** (e.g., `results/gaussian/2026-02-11_21-34-10/`) with a `config.json` snapshot, `results.csv`, and plots. Previous runs are never overwritten.

### Experiment 1a: Gaussian Rewards

Scalar setting with `r ~ N(0, 1)`, where `V* = μ + σ²/(2β)`.

```bash
# Quick test (~0.5 seconds)
python run_experiment1_gaussian.py --quick

# Full experiment (~2-3 minutes)
python run_experiment1_gaussian.py
```

**Configuration:**
- β ∈ {0.5, 1.0, 2.0}
- N_tot ∈ {16, 32, 64, 128, 256, 512, 1024, 2048}
- 1000 Monte Carlo trials
- 5 single-replica variants (n ∈ {2, 3, 4, 5, 8})
- 5 multi-n order sets (each with/without jackknife)

**Output:** `results/gaussian/<timestamp>/` — 9 triptych plots + CSV + config.json

### Experiment 1b: Bernoulli Rewards

Binary reward setting with `r ~ Bernoulli(p)`, where `V* = β·log(1-p + p·exp(1/β))`.
Focuses on the rare-success regime relevant to A*PO with difficult prompts.

```bash
# Quick test (~6 seconds)
python run_experiment1_bernoulli.py --quick

# Full experiment (~3-4 minutes)
python run_experiment1_bernoulli.py
```

**Configuration:**
- p ∈ {0.01, 0.05, 0.1, 0.2, 0.5}
- β ∈ {0.5, 1.0, 2.0}
- N_tot ∈ {4, 8, 16, 32, 64, 128, 256}
- 1000 Monte Carlo trials
- Same estimator configurations as Gaussian

**Output:** `results/bernoulli/<timestamp>/` — 9 triptych plots per p value (45 total) + CSV + config.json

## Understanding the Plots

Each plot is a **triptych** — three side-by-side panels sharing the same y-axis:

| Left Panel | Middle Panel | Right Panel |
|---|---|---|
| Single-Replica estimators | Multi-n Slope estimators | Multi-n Slope + Jackknife |

**LME (baseline)** appears as a solid black line in every panel for reference.

- **Bias plots**: How far estimates are from true V* (closer to 0 is better)
- **Variance plots**: Spread of estimates across trials (log scale; lower is better)
- **RMSE plots**: Overall error combining bias and variance (log scale; lower is better)

## Code Architecture

The codebase is designed to be **modular and distribution-agnostic**:

- **`src/estimators.py`** — All four estimators. They take raw reward arrays and don't know the underlying distribution.
- **`src/experiment_runner.py`** — Generic Monte Carlo loop. Parametrized by `sample_fn(rng, n_tot)` and `v_star_fn(beta)`.
- **`src/plotting.py`** — Triptych plotting code. Works with any DataFrame in the standard format.
- **Experiment scripts** — Thin wrappers that define the distribution-specific configuration and call the runner.

To add a new reward distribution, you only need:
1. Add ground-truth functions to `src/ground_truth.py`
2. Write a thin experiment script (follow `run_experiment1_bernoulli.py` as a template)

## CSV Data Format

The results CSV contains:
- `beta`: KL regularization parameter
- `n_tot`: Sample budget
- `method`: Estimator name (e.g., "lme", "single_n2", "multi_[2,3,4]_jk")
- `bias`, `variance`, `rmse`: Evaluation metrics
- `v_star`: Ground truth V*
- `mean_estimate`: Average estimate across trials
- `n_valid`: Number of valid (non-NaN) estimates
- `p` (Bernoulli only): Success probability

## Troubleshooting

| Problem | Solution |
|---|---|
| Import errors | Activate venv: `source venv/bin/activate && pip install -r requirements.txt` |
| Slow execution | Use `--quick` mode for testing |
| Memory issues | Reduce `T_TRIALS` or `N_TOT_VALUES` in the script |
| Plot warnings | Matplotlib warnings about fonts/cache can usually be ignored |
