# Replica-Based and Bayesian Shrinkage Estimators for V\* in A\*-PO

Code for the senior thesis:  
**"Replica-Based and Bayesian Shrinkage Estimators for Offline Value Estimation in KL-Regularized Reinforcement Learning"**  
Haotian Sun, Harvard University, 2026.

## Overview

A\*-PO is a two-stage RL algorithm for fine-tuning large language models. Stage 1 estimates the KL-regularized optimal value function V\*(x) offline; Stage 2 trains the policy online by regressing on optimal advantage targets r − V̂\*(x). The default log-mean-exp (LME) estimator is downward-biased due to Jensen's inequality, and this bias is significant when the per-prompt sample budget is small and the pass rate is low.

This repository implements and evaluates two families of alternative estimators:

- **Replica-based estimators** (inspired by the replica trick from statistical physics): single-replica plug-in and multi-n slope with optional jackknife bias correction.
- **Beta-smooth estimators** (Bayesian shrinkage for binary rewards): replace the MLE pass-rate estimate with a Beta posterior mean, eliminating endpoint snapping at k = 0 or k = N. The Jeffreys prior (α = γ = 0.5) gives p̃ = (k + 0.5)/(N + 1).

### Main finding

The Jeffreys Beta-smooth estimator is a drop-in replacement for LME in A\*-PO Stage 1 for binary rewards. It reduces RMSE and advantage distortion (Experiments 1–2) and produces ~25% lower parameter variance in the learned policy (Experiment 3), at no additional computational cost.

## Repository Structure

```
├── src/
│   ├── estimators.py          # V* estimators: LME, replica, beta-smooth
│   ├── ground_truth.py        # Closed-form V* for Gaussian and Bernoulli
│   ├── metrics.py             # Bias, variance, RMSE
│   ├── exp1_runner.py         # Experiment 1: scalar Monte Carlo runner
│   ├── exp1_plotting.py       # Experiment 1: 2x2 panel plots
│   ├── exp2_runner.py         # Experiment 2: contextual bandit runner
│   ├── exp2_plotting.py       # Experiment 2: calibration plots + CSV tables
│   ├── exp3_runner.py         # Experiment 3: two-stage policy learning
│   └── exp3_plotting.py       # Experiment 3: loss/param/scatter plots
├── run_experiment1_gaussian.py
├── run_experiment1_bernoulli.py
├── run_experiment2.py
├── run_experiment3.py
├── results/                   # Timestamped output directories
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires: `numpy`, `scipy`, `pandas`, `matplotlib`.

## Experiments

Each run creates a timestamped directory under `results/` with a `config.json` snapshot, CSVs, and plots. Use `--quick` for a fast test run.

### Experiment 1: Scalar V\* Estimation

Evaluates bias, variance, and RMSE of all estimators in a controlled single-prompt setting with known ground truth.

- **1a (Gaussian):** r ~ N(0, 1), V\* = μ + σ²/(2β). Sweeps β ∈ {0.5, 1, 2} and N ∈ {16, …, 2048}.
- **1b (Bernoulli):** r ~ Bernoulli(p), V\* = β log(1 − p + p exp(1/β)). Sweeps p ∈ {0.01, …, 0.5} and N ∈ {4, …, 256}. Includes beta-smooth estimators.

```bash
python run_experiment1_gaussian.py [--quick]
python run_experiment1_bernoulli.py [--quick]
```

### Experiment 2: Contextual Bandit Simulation

500 prompts with pass rates drawn from Beta distributions (hard: Beta(1,8), moderate: Beta(2,5)). Evaluates calibration, stratified bias/RMSE, win rates vs LME, and advantage distortion.

```bash
python run_experiment2.py [--quick]
```

### Experiment 3: Two-Stage Policy Learning

Simulates A\*-PO's full pipeline in a toy binary contextual bandit. Stage 1 estimates V\*(x); Stage 2 trains q(x; w, b) = σ(w · logit(pₓ) + b) via on-policy SGD.

```bash
python run_experiment3.py [--quick] [--exclude-camel]
```