#!/usr/bin/env python3
"""
Experiment 3: Policy Learning in the Toy Bandit

Simulates A*PO's full two-stage pipeline in the binary contextual bandit:
  Stage 1: Estimate V*(x) for each prompt using N Bernoulli samples.
  Stage 2: Train a shared-parameter policy via on-policy SGD using
           advantage targets r - V_hat(x).

The policy is: q(x; w, b) = sigmoid(w * logit(p_x) + b)
At (w=1, b=0): reference policy.  At (w=1, b=1/beta2): optimal policy.

Loss modes:
    - MSE:   standard squared error against point target
    - CAMeL: interval-target loss using Bayesian credible intervals

Conditions compared:
    1. LME (MSE)      — A*PO's standard estimator with MSE loss
    2. Jeffreys (MSE)  — Beta-smooth with Jeffreys prior, MSE loss
    3. Oracle (MSE)    — True pass rate, MSE loss (upper bound)
    4. CAMeL 90% CI    — Jeffreys posterior interval, CAMeL loss

Usage:
    python run_experiment3.py           # Full experiment
    python run_experiment3.py --quick   # Fast test run

Each run is saved to: results/experiment3/YYYY-MM-DD_HH-MM-SS/
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ground_truth import generate_prompt_pass_rates, assign_strata
from src.exp3_runner import run_experiment3
from src.exp3_plotting import generate_all_exp3_outputs


# =============================================================================
# Experiment Configuration
# =============================================================================

DIFFICULTY_REGIMES = {
    "hard":     {"a": 1.0, "b": 8.0, "label": "Hard — Beta(1, 8)"},
    "moderate": {"a": 2.0, "b": 5.0, "label": "Moderate — Beta(2, 5)"},
}

BETA2 = 0.5
N_SAMPLES = 8
M_PROMPTS = 500
T_TRIALS = 200
T_SGD = 5000
LR = 0.0005
SEED = 42

# Each condition specifies the Stage 1 estimator and Stage 2 loss mode.
CONDITIONS = [
    {"name": "lme",      "estimator": "lme",      "loss_mode": "mse"},
    {"name": "jeffreys", "estimator": "jeffreys",  "loss_mode": "mse"},
    {"name": "oracle",   "estimator": "oracle",    "loss_mode": "mse"},
    {"name": "camel_90", "estimator": "jeffreys",  "loss_mode": "camel", "confidence": 0.90},
]

BASE_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "experiment3"
)

# =============================================================================
# Quick mode
# =============================================================================

QUICK_CONDITIONS = [
    {"name": "lme",      "estimator": "lme",      "loss_mode": "mse"},
    {"name": "jeffreys", "estimator": "jeffreys",  "loss_mode": "mse"},
    {"name": "oracle",   "estimator": "oracle",    "loss_mode": "mse"},
    {"name": "camel_90", "estimator": "jeffreys",  "loss_mode": "camel", "confidence": 0.90},
]

QUICK_CONFIG = {
    "difficulty_regimes": {"hard": DIFFICULTY_REGIMES["hard"]},
    "m_prompts": 50,
    "t_trials": 20,
    "t_sgd": 500,
    "conditions": QUICK_CONDITIONS,
}


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Policy Learning in the Toy Bandit"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a reduced configuration for quick testing.",
    )
    args = parser.parse_args()

    if args.quick:
        difficulty_regimes = QUICK_CONFIG["difficulty_regimes"]
        m_prompts = QUICK_CONFIG["m_prompts"]
        t_trials = QUICK_CONFIG["t_trials"]
        t_sgd = QUICK_CONFIG["t_sgd"]
        conditions = QUICK_CONFIG["conditions"]
        print("=== QUICK MODE ===")
    else:
        difficulty_regimes = DIFFICULTY_REGIMES
        m_prompts = M_PROMPTS
        t_trials = T_TRIALS
        t_sgd = T_SGD
        conditions = CONDITIONS

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(BASE_RESULTS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "experiment": "3_policy_learning",
        "difficulty_regimes": {
            k: {"a": v["a"], "b": v["b"]} for k, v in difficulty_regimes.items()
        },
        "beta2": BETA2,
        "n_samples": N_SAMPLES,
        "m_prompts": m_prompts,
        "t_trials": t_trials,
        "t_sgd": t_sgd,
        "lr": LR,
        "conditions": [c["name"] for c in conditions],
        "seed": SEED,
        "quick_mode": args.quick,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 65)
    print("Experiment 3: Policy Learning in the Toy Bandit")
    print("=" * 65)
    print(f"  Run directory:   {run_dir}")
    print(f"  Regimes:         {list(difficulty_regimes.keys())}")
    print(f"  beta2:           {BETA2}")
    print(f"  N (Stage 1):     {N_SAMPLES}")
    print(f"  Prompts (M):     {m_prompts}")
    print(f"  MC trials:       {t_trials}")
    print(f"  SGD steps:       {t_sgd}")
    print(f"  Learning rate:   {LR}")
    print(f"  Conditions:      {[c['name'] for c in conditions]}")
    print(f"  Seed:            {SEED}")
    print(f"  Optimal (w*, b*) = (1.0, {1.0/BETA2:.1f})")
    print()

    overall_start = time.time()

    for regime_name, regime_cfg in difficulty_regimes.items():
        a, b_param = regime_cfg["a"], regime_cfg["b"]
        dist_label = regime_cfg["label"]

        print(f"\n{'='*65}")
        print(f"  Regime: {dist_label}")
        print(f"{'='*65}")

        rng_prompts = np.random.default_rng(SEED)
        p_array = generate_prompt_pass_rates(m_prompts, a, b_param, rng_prompts)
        strata = assign_strata(p_array)

        for s in ["very_hard", "hard", "medium", "easy"]:
            count = np.sum(strata == s)
            if count > 0:
                mean_p = p_array[strata == s].mean()
                print(f"    {s:>10s}: {count:3d} prompts  (mean p = {mean_p:.3f})")

        q_star = p_array * np.exp(1.0 / BETA2) / (
            p_array * np.exp(1.0 / BETA2) + (1.0 - p_array)
        )
        print(f"\n  Optimal q* range: [{q_star.min():.3f}, {q_star.max():.3f}], "
              f"mean = {q_star.mean():.3f}")
        print()

        t_start = time.time()
        results = run_experiment3(
            p_array=p_array,
            strata=strata,
            beta2=BETA2,
            n_samples=N_SAMPLES,
            t_trials=t_trials,
            t_sgd=t_sgd,
            lr=LR,
            conditions=conditions,
            seed=SEED,
        )
        elapsed = time.time() - t_start
        print(f"\n  Regime '{regime_name}' completed in {elapsed:.1f}s")

        regime_dir = os.path.join(run_dir, regime_name)
        generate_all_exp3_outputs(results, regime_dir, BETA2, dist_label)

        print(f"\n  Summary ({regime_name}):")
        summary = results["summary_df"]
        for cond in conditions:
            cname = cond["name"]
            df_e = summary[summary["estimator"] == cname]
            print(
                f"    {cname:>10s}:  "
                f"J = {df_e['objective'].mean():.5f} ± {df_e['objective'].std():.5f}  "
                f"w = {df_e['w_final'].mean():.3f}  "
                f"b = {df_e['b_final'].mean():.3f}  "
                f"|q-q*| = {df_e['policy_error'].mean():.5f}"
            )

    total_time = time.time() - overall_start
    print(f"\nTotal experiment time: {total_time:.1f}s")
    print(f"Done! Results in {run_dir}")


if __name__ == "__main__":
    main()
