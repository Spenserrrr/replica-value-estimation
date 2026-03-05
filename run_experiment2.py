#!/usr/bin/env python3
"""
Experiment 2: Contextual Bandit Simulation of A*PO Stage 1

This script simulates A*PO's Stage 1 value estimation across M prompts
with heterogeneous difficulty (pass rates drawn from a Beta distribution).
It evaluates whether replica-based estimators produce better-calibrated
value predictions and reduce downstream advantage distortion.

Estimators compared (curated from Experiment 1 winners):
    1. LME (log-mean-exp)                — A*PO's current estimator
    2. Single-replica n=4                — best single-n at low p
    3. Single-replica n=8                — competitive in sparsest regime
    4. Multi-n slope [2,4,6,8]           — best overall non-LME method
    5. Multi-n slope [2,3,4]             — best at very low budgets
    6. Multi-n slope [2,4,6,8] + JK      — best jackknife variant

Two difficulty distributions are tested:
    - Hard regime:     Beta(1, 8),  mean ≈ 0.11
    - Moderate regime: Beta(2, 5),  mean ≈ 0.29

Usage:
    python run_experiment2.py           # Full experiment
    python run_experiment2.py --quick   # Fast test run

Each run is saved to: results/experiment2/YYYY-MM-DD_HH-MM-SS/
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ground_truth import (
    compute_v_star_bernoulli_vec,
    generate_prompt_pass_rates,
    assign_strata,
)
from src.exp2_runner import run_experiment2
from src.exp2_plotting import generate_all_exp2_outputs


# =============================================================================
# Experiment Configuration
# =============================================================================

# ---- Difficulty distributions to sweep (Beta parameters) ----
DIFFICULTY_REGIMES = {
    "hard":     {"a": 1.0, "b": 8.0, "label": "Hard — Beta(1, 8)"},
    "moderate": {"a": 2.0, "b": 5.0, "label": "Moderate — Beta(2, 5)"},
}

# ---- KL regularization (matches Experiment 1) ----
BETAS = [0.5, 1.0, 2.0]

# ---- Per-prompt sample budgets (A*PO default = 8) ----
N_SAMPLES_VALUES = [4, 8, 16, 32, 64]

# ---- Number of prompts ----
M_PROMPTS = 500

# ---- Monte Carlo trials per (prompt, beta, N) ----
T_TRIALS = 200

# ---- Estimators: curated from Experiment 1 results ----
SINGLE_N_VALUES = [4, 8]
MULTI_N_SETS = [
    [2, 4, 6, 8],  # Best overall non-LME in sparse Bernoulli
    [2, 3, 4],     # Best at very low budgets (N=4)
]

# ---- Stage 2 KL coefficient (for distortion analysis) ----
BETA2 = 1e-3

# ---- Random seed ----
SEED = 42

# ---- Output directory ----
BASE_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "experiment2"
)


# =============================================================================
# Quick mode (reduced config for fast iteration)
# =============================================================================

QUICK_CONFIG = {
    "difficulty_regimes": {"hard": DIFFICULTY_REGIMES["hard"]},
    "betas": [0.5, 1.0],
    "n_samples_values": [8, 32],
    "m_prompts": 50,
    "t_trials": 20,
    "single_n_values": [4, 8],
    "multi_n_sets": [[2, 4, 6, 8]],
}


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Contextual Bandit Simulation of A*PO Stage 1"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a reduced configuration for quick testing.",
    )
    args = parser.parse_args()

    # ---- Select configuration ----
    if args.quick:
        cfg = QUICK_CONFIG
        difficulty_regimes = cfg["difficulty_regimes"]
        betas = cfg["betas"]
        n_samples_values = cfg["n_samples_values"]
        m_prompts = cfg["m_prompts"]
        t_trials = cfg["t_trials"]
        single_n_values = cfg["single_n_values"]
        multi_n_sets = cfg["multi_n_sets"]
        print("=== QUICK MODE ===")
    else:
        difficulty_regimes = DIFFICULTY_REGIMES
        betas = BETAS
        n_samples_values = N_SAMPLES_VALUES
        m_prompts = M_PROMPTS
        t_trials = T_TRIALS
        single_n_values = SINGLE_N_VALUES
        multi_n_sets = MULTI_N_SETS

    # ---- Create timestamped run directory ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(BASE_RESULTS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # ---- Save configuration ----
    config = {
        "experiment": "2_contextual_bandit",
        "difficulty_regimes": {
            k: {"a": v["a"], "b": v["b"]} for k, v in difficulty_regimes.items()
        },
        "betas": betas,
        "n_samples_values": n_samples_values,
        "m_prompts": m_prompts,
        "t_trials": t_trials,
        "single_n_values": single_n_values,
        "multi_n_sets": multi_n_sets,
        "beta2": BETA2,
        "seed": SEED,
        "quick_mode": args.quick,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- Print header ----
    print("=" * 65)
    print("Experiment 2: Contextual Bandit Simulation of A*PO Stage 1")
    print("=" * 65)
    print(f"  Run directory:        {run_dir}")
    print(f"  Difficulty regimes:   {list(difficulty_regimes.keys())}")
    print(f"  Beta values:          {betas}")
    print(f"  N values:             {n_samples_values}")
    print(f"  Prompts (M):          {m_prompts}")
    print(f"  MC trials (T):        {t_trials}")
    print(f"  Single-n values:      {single_n_values}")
    print(f"  Multi-n sets:         {multi_n_sets}")
    print(f"  Stage 2 beta_2:       {BETA2}")
    print(f"  Seed:                 {SEED}")
    print()

    overall_start = time.time()
    all_dfs = []

    # ---- Run for each difficulty regime ----
    for regime_name, regime_cfg in difficulty_regimes.items():
        a, b = regime_cfg["a"], regime_cfg["b"]
        dist_label = regime_cfg["label"]

        print(f"\n{'='*65}")
        print(f"  Regime: {dist_label}")
        print(f"{'='*65}")

        # Generate prompt pass rates
        rng_prompts = np.random.default_rng(SEED)
        p_array = generate_prompt_pass_rates(m_prompts, a, b, rng_prompts)
        strata = assign_strata(p_array)

        # Print stratum counts
        for s in ["very_hard", "hard", "medium", "easy"]:
            count = np.sum(strata == s)
            if count > 0:
                mean_p = p_array[strata == s].mean()
                print(f"    {s:>10s}: {count:3d} prompts  (mean p = {mean_p:.3f})")

        # Print ground truth V* range at each beta
        print("\n  Ground truth V*(x) ranges:")
        for beta in betas:
            v_star = compute_v_star_bernoulli_vec(p_array, beta)
            print(
                f"    beta={beta}: V* in [{v_star.min():.4f}, {v_star.max():.4f}], "
                f"mean={v_star.mean():.4f}"
            )
        print()

        # Run the experiment
        t_start = time.time()
        df = run_experiment2(
            p_array=p_array,
            strata=strata,
            betas=betas,
            n_samples_values=n_samples_values,
            t_trials=t_trials,
            single_n_values=single_n_values,
            multi_n_sets=multi_n_sets,
            seed=SEED,
            beta2=BETA2,
        )
        elapsed = time.time() - t_start
        print(f"\n  Regime '{regime_name}' completed in {elapsed:.1f}s")

        # Tag with regime name
        df["regime"] = regime_name

        # Save per-regime CSV
        csv_path = os.path.join(run_dir, f"results_{regime_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved to {csv_path}")

        # Generate plots and summary tables for this regime
        plot_dir = os.path.join(run_dir, regime_name)
        generate_all_exp2_outputs(df, plot_dir, dist_label=dist_label)

        all_dfs.append(df)

    # ---- Combine and save all results ----
    df_all = pd.concat(all_dfs, ignore_index=True)
    csv_all = os.path.join(run_dir, "results_all.csv")
    df_all.to_csv(csv_all, index=False)
    print(f"\nSaved combined results to {csv_all}")

    total_time = time.time() - overall_start
    print(f"Total experiment time: {total_time:.1f}s")

    # ---- Print summary at A*PO operating point (beta=0.5, N=8) ----
    print("\n" + "=" * 65)
    print("Summary at A*PO operating point (beta=0.5, N=8)")
    print("=" * 65)
    for regime_name in difficulty_regimes:
        df_op = df_all[
            (df_all["regime"] == regime_name)
            & (df_all["beta"] == 0.5)
            & (df_all["n_samples"] == 8)
        ]
        if df_op.empty:
            continue
        print(f"\n  Regime: {regime_name}")
        # Aggregate per method
        summary = df_op.groupby("method").agg(
            mean_bias=("bias", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_distortion=("log_ratio_distortion", "mean"),
        ).sort_values("mean_rmse")
        for method, row in summary.iterrows():
            print(
                f"    {method:>25s}:  "
                f"Bias={row['mean_bias']:+.5f}  "
                f"RMSE={row['mean_rmse']:.5f}  "
                f"Distortion={row['mean_distortion']:.1f} nats"
            )

    print(f"\nDone! Results in {run_dir}")


if __name__ == "__main__":
    main()
