# V* Estimation and Policy Learning for A*PO

Implementation of V* estimators and policy learning experiments for the A*PO algorithm framework.

## Overview

This repository contains the implementation and experiments for my senior thesis on **offline value function estimation** under KL-regularized control. The code implements and compares multiple estimator families and studies their downstream impact on policy learning:

1. **LME (Log-Mean-Exp)** — Baseline biased estimator (A*PO default)
2. **Single-Replica** — Fixed replica order n > 1
3. **Multi-n Slope** — Linear extrapolation across multiple replica orders
4. **Multi-n Slope + Jackknife** — Slope estimator with delete-one jackknife bias correction
5. **Beta-Smoothed** — Bayesian shrinkage for Bernoulli pass rates (Laplace, Jeffreys priors)

## Repository Structure

```
.
├── src/
│   ├── __init__.py            # Package marker
│   ├── estimators.py          # Core V* estimators (LME, replica, beta-smooth)
│   ├── ground_truth.py        # Analytical V* for Gaussian and Bernoulli + prompt generation
│   ├── metrics.py             # Bias, variance, RMSE metrics
│   ├── exp1_runner.py         # Experiment 1: scalar Monte Carlo runner
│   ├── exp1_plotting.py       # Experiment 1: triptych plot generation
│   ├── exp2_runner.py         # Experiment 2: contextual bandit runner
│   ├── exp2_plotting.py       # Experiment 2: calibration plots + CSV tables
│   ├── exp3_runner.py         # Experiment 3: Stage 2 policy learning runner
│   └── exp3_plotting.py       # Experiment 3: training curves + policy plots
├── run_experiment1_gaussian.py    # Experiment 1a: Gaussian rewards
├── run_experiment1_bernoulli.py   # Experiment 1b: Bernoulli rewards
├── run_experiment2.py             # Experiment 2: contextual bandit V* estimation
├── run_experiment3.py             # Experiment 3: policy learning in the toy bandit
├── results/
│   ├── gaussian/                  # Experiment 1a outputs
│   ├── bernoulli/                 # Experiment 1b outputs
│   ├── experiment2/               # Experiment 2 outputs
│   └── experiment3/               # Experiment 3 outputs
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

**Output:** `results/bernoulli/<timestamp>/` — triptych plots per p value + CSV + config.json

### Experiment 2: Contextual Bandit V* Estimation

Multi-prompt setting: M prompts with pass rates drawn from Beta distributions.
Evaluates calibration, stratified bias/RMSE, win rates, and advantage distortion.

```bash
# Quick test
python run_experiment2.py --quick

# Full experiment
python run_experiment2.py
```

**Configuration:**
- Two difficulty regimes: Hard — Beta(1, 8), Moderate — Beta(2, 5)
- β ∈ {0.5}, N ∈ {4, 8, 16, 32, 64}
- M = 500 prompts, T = 200 Monte Carlo trials
- Estimators: LME, selected replica methods, Beta-smooth (Laplace, Jeffreys)

**Output:** `results/experiment2/<timestamp>/` — calibration plots (scatter + binned) + CSV tables

### Experiment 3: Policy Learning in the Toy Bandit

Simulates A*PO's full two-stage pipeline in the binary contextual bandit:
- **Stage 1:** Estimate V*(x) using N Bernoulli samples
- **Stage 2:** Train a shared-parameter policy q(x; w, b) = σ(w · logit(p_x) + b) via on-policy SGD

The beta mismatch between stages is eliminated by recomputing V* at β₂ from the estimated pass rate.

```bash
# Quick test (~1 second)
python run_experiment3.py --quick

# Full experiment (~5-10 minutes)
python run_experiment3.py
```

**Configuration:**
- Two difficulty regimes: Hard — Beta(1, 8), Moderate — Beta(2, 5)
- β₂ = 0.5, N = 8, M = 500 prompts
- T = 200 MC trials, 1000 SGD steps per trial
- Estimators: LME, Jeffreys Beta-smooth, Oracle
- Optimal policy: (w*, b*) = (1.0, 2.0)

**Output:** `results/experiment3/<timestamp>/` — training loss curves, parameter trajectories, policy scatter + CSV tables

## Code Architecture

The codebase is designed to be **modular and distribution-agnostic**:

- **`src/estimators.py`** — All estimators (LME, replica, beta-smooth). They take raw reward arrays and don't know the underlying distribution.
- **`src/ground_truth.py`** — Analytical V* for Gaussian/Bernoulli, prompt generation and strata assignment.
- **`src/metrics.py`** — Bias, variance, RMSE computation.
- **`src/exp1_runner.py`** — Generic Monte Carlo loop for scalar V* estimation.
- **`src/exp1_plotting.py`** — Triptych plots for Experiment 1.
- **`src/exp2_runner.py`** — Contextual bandit runner (per-prompt trials).
- **`src/exp2_plotting.py`** — Calibration plots and stratified CSV tables for Experiment 2.
- **`src/exp3_runner.py`** — Stage 1 estimation + Stage 2 SGD policy learning.
- **`src/exp3_plotting.py`** — Training curves, parameter trajectories, and policy scatter for Experiment 3.
- **Experiment scripts** — Thin wrappers that define configuration and call the runners.

## Troubleshooting

| Problem | Solution |
|---|---|
| Import errors | Activate venv: `source venv/bin/activate && pip install -r requirements.txt` |
| Slow execution | Use `--quick` mode for testing |
| Memory issues | Reduce `T_TRIALS` or `N_TOT_VALUES` in the script |
| Plot warnings | Matplotlib warnings about fonts/cache can usually be ignored |
