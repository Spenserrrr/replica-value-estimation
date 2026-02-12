#!/usr/bin/env python3
"""
Experiment 1a: Gaussian Reward Model — Pure Scalar Estimation of V*

This script evaluates four families of V* estimators in a controlled Gaussian
reward setting where the ground truth is known analytically:

    r ~ N(mu_r, sigma_r^2)
    V* = mu_r + sigma_r^2 / (2 * beta)

Estimators compared (under matched sample budget N_tot):
    1. LME (log-mean-exp)            — standard biased baseline
    2. Single-replica (various n)    — fixed replica order
    3. Multi-n slope (no jackknife)  — linear fit of phi_hat(n) vs n
    4. Multi-n slope (with jackknife)— jackknife-corrected version of (3)

Usage:
    python run_experiment1_gaussian.py          # Full experiment
    python run_experiment1_gaussian.py --quick  # Fast test run

Each run is saved to a timestamped directory: results/gaussian/YYYY-MM-DD_HH-MM-SS/
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import pandas as pd

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ground_truth import compute_v_star_gaussian
from src.experiment_runner import run_experiment
from src.plotting import plot_results


# =============================================================================
# Experiment Configuration
# =============================================================================

# ---- Reward distribution parameters ----
MU_R = 0.0         # Reward mean (zero-centered for simplicity)
SIGMA_R = 1.0      # Reward standard deviation

# ---- KL regularization parameters to sweep ----
BETAS = [0.5, 1.0, 2.0]

# ---- Sample budgets to sweep (log-spaced) ----
N_TOT_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]

# ---- Monte Carlo trials per configuration ----
T_TRIALS = 1000

# ---- Single-replica: sweep multiple replica orders ----
SINGLE_REPLICA_N_VALUES = [2, 3, 4, 5, 8]

# ---- Multi-n slope: test multiple sets of replica orders ----
MULTI_N_ORDER_SETS = [
    [2, 3],           # Minimal set
    [2, 3, 4],        # Small set
    [2, 3, 4, 5],     # Standard set
    [2, 3, 4, 5, 6],  # Extended set
    [2, 4, 6, 8],     # Even orders only
]

# ---- Random seed for reproducibility ----
SEED = 42

# ---- Base output directory ----
BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "gaussian")


# =============================================================================
# Quick mode (reduced config for fast iteration)
# =============================================================================

QUICK_CONFIG = {
    "betas": [0.5, 1.0, 2.0],
    "n_tot_values": [32, 128, 512],
    "t_trials": 50,
    "single_n_values": [2, 4],
    "multi_n_sets": [[2, 3, 4, 5]],
}


# =============================================================================
# Sample and Ground-Truth Functions
# =============================================================================

def sample_fn(rng, n_tot):
    """Draw n_tot i.i.d. samples from N(MU_R, SIGMA_R^2)."""
    return rng.normal(MU_R, SIGMA_R, size=n_tot)


def v_star_fn(beta):
    """Return the exact V* for the Gaussian reward model."""
    return compute_v_star_gaussian(MU_R, SIGMA_R, beta)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1a: Gaussian rewards — Pure Scalar V* Estimation"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a reduced configuration for quick testing."
    )
    args = parser.parse_args()

    dist_label = f"Gaussian(μ={MU_R}, σ={SIGMA_R})"

    # ---- Select configuration ----
    if args.quick:
        cfg = QUICK_CONFIG
        betas = cfg["betas"]
        n_tot_values = cfg["n_tot_values"]
        t_trials = cfg["t_trials"]
        single_n_values = cfg["single_n_values"]
        multi_n_sets = cfg["multi_n_sets"]
        print("=== QUICK MODE ===")
    else:
        betas = BETAS
        n_tot_values = N_TOT_VALUES
        t_trials = T_TRIALS
        single_n_values = SINGLE_REPLICA_N_VALUES
        multi_n_sets = MULTI_N_ORDER_SETS

    # ---- Create timestamped run directory ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(BASE_RESULTS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # ---- Save configuration snapshot ----
    config = {
        "experiment": "1a_gaussian",
        "distribution": {"type": "gaussian", "mu_r": MU_R, "sigma_r": SIGMA_R},
        "betas": betas,
        "n_tot_values": n_tot_values,
        "t_trials": t_trials,
        "single_n_values": single_n_values,
        "multi_n_sets": multi_n_sets,
        "seed": SEED,
        "quick_mode": args.quick,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- Print experiment header ----
    print("=" * 60)
    print("Experiment 1a: Gaussian Reward Model")
    print("=" * 60)
    print(f"  Run directory:           {run_dir}")
    print(f"  Reward distribution:     N({MU_R}, {SIGMA_R}²)")
    print(f"  Beta values:             {betas}")
    print(f"  N_tot values:            {n_tot_values}")
    print(f"  Monte Carlo trials:      {t_trials}")
    print(f"  Single-replica n values: {single_n_values}")
    print(f"  Multi-n order sets:      {multi_n_sets}")
    print(f"  Random seed:             {SEED}")
    print()

    # Print ground truth for each beta
    print("Ground truth V* values:")
    for beta in betas:
        v_star = v_star_fn(beta)
        print(f"  beta={beta}:  V* = {v_star:.6f}")
    print()

    # ---- Run experiment ----
    print("Running Monte Carlo trials...")
    t_start = time.time()
    df = run_experiment(
        sample_fn=sample_fn,
        v_star_fn=v_star_fn,
        betas=betas,
        n_tot_values=n_tot_values,
        t_trials=t_trials,
        single_n_values=single_n_values,
        multi_n_sets=multi_n_sets,
        seed=SEED,
    )
    total_time = time.time() - t_start
    print(f"\nTotal experiment time: {total_time:.1f}s")

    # ---- Save results ----
    csv_path = os.path.join(run_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("Summary (largest N_tot for first beta)")
    print("=" * 60)
    df_sample = df[(df["beta"] == betas[0]) & (df["n_tot"] == n_tot_values[-1])]
    for _, row in df_sample.head(10).iterrows():
        print(
            f"  {row['method']:>30s}:  "
            f"Bias={row['bias']:+.5f}  "
            f"Var={row['variance']:.5f}  "
            f"RMSE={row['rmse']:.5f}"
        )

    # ---- Generate plots ----
    print("\nGenerating plots...")
    plot_results(df, run_dir, dist_label=dist_label)
    print(f"\nDone! Results in {run_dir}")


if __name__ == "__main__":
    main()
