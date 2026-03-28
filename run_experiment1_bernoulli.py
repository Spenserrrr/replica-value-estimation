#!/usr/bin/env python3
"""
Experiment 1b: Bernoulli V* (same estimators as 1a). Small p mimics rare successes / low pass@N.

Usage:
    python run_experiment1_bernoulli.py
    python run_experiment1_bernoulli.py --quick

Outputs: results/bernoulli/YYYY-MM-DD_HH-MM-SS/
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ground_truth import compute_v_star_bernoulli
from src.exp1_runner import run_experiment
from src.exp1_plotting import plot_results


P_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]

BETAS = [0.5, 1.0, 2.0]

N_TOT_VALUES = [4, 8, 16, 32, 64, 128, 256]

T_TRIALS = 1000

SINGLE_REPLICA_N_VALUES = [2, 3, 4, 5, 8]

MULTI_N_ORDER_SETS = [
    [2, 3],
    [2, 3, 4],
    [2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [2, 4, 6, 8],
]

# (name, alpha, gamma) for Beta(alpha, gamma) smoothing of the success rate.
BETA_SMOOTH_PRIORS = [
    ("laplace", 1.0, 1.0),    # p_tilde = (k+1)/(N+2)
    ("jeffreys", 0.5, 0.5),   # p_tilde = (k+0.5)/(N+1)
]

SEED = 42

BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "bernoulli")


QUICK_CONFIG = {
    "p_values": [0.1, 0.5],
    "betas": [0.5, 1.0, 2.0],
    "n_tot_values": [8, 32, 128],
    "t_trials": 50,
    "single_n_values": [2, 4],
    "multi_n_sets": [[2, 3, 4, 5]],
    "beta_smooth_priors": [("laplace", 1.0, 1.0)],
}


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1b: Bernoulli rewards — Pure Scalar V* Estimation"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a reduced configuration for quick testing."
    )
    args = parser.parse_args()

    if args.quick:
        cfg = QUICK_CONFIG
        p_values = cfg["p_values"]
        betas = cfg["betas"]
        n_tot_values = cfg["n_tot_values"]
        t_trials = cfg["t_trials"]
        single_n_values = cfg["single_n_values"]
        multi_n_sets = cfg["multi_n_sets"]
        beta_smooth_priors = cfg["beta_smooth_priors"]
        print("=== QUICK MODE ===")
    else:
        p_values = P_VALUES
        betas = BETAS
        n_tot_values = N_TOT_VALUES
        t_trials = T_TRIALS
        single_n_values = SINGLE_REPLICA_N_VALUES
        multi_n_sets = MULTI_N_ORDER_SETS
        beta_smooth_priors = BETA_SMOOTH_PRIORS

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(BASE_RESULTS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "experiment": "1b_bernoulli",
        "distribution": {"type": "bernoulli", "p_values": p_values},
        "betas": betas,
        "n_tot_values": n_tot_values,
        "t_trials": t_trials,
        "single_n_values": single_n_values,
        "multi_n_sets": multi_n_sets,
        "beta_smooth_priors": [(n, a, g) for n, a, g in beta_smooth_priors],
        "seed": SEED,
        "quick_mode": args.quick,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("Experiment 1b: Bernoulli Reward Model")
    print("=" * 60)
    print(f"  Run directory:           {run_dir}")
    print(f"  p values:                {p_values}")
    print(f"  Beta values:             {betas}")
    print(f"  N_tot values:            {n_tot_values}")
    print(f"  Monte Carlo trials:      {t_trials}")
    print(f"  Single-replica n values: {single_n_values}")
    print(f"  Multi-n order sets:      {multi_n_sets}")
    print(f"  Beta-smooth priors:      {[(n, a, g) for n, a, g in beta_smooth_priors]}")
    print(f"  Random seed:             {SEED}")
    print()

    print("Ground truth V* values:")
    for p in p_values:
        for beta in betas:
            v_star = compute_v_star_bernoulli(p, beta)
            print(f"  p={p}, beta={beta}:  V* = {v_star:.6f}")
    print()

    all_dfs = []
    overall_start = time.time()

    for p_idx, p in enumerate(p_values):
        print(f"\n{'='*60}")
        print(f"  Running p={p}  ({p_idx+1}/{len(p_values)})")
        print(f"{'='*60}")

        def make_sample_fn(p_val):
            def sample_fn(rng, n_tot):
                return rng.binomial(1, p_val, size=n_tot).astype(float)
            return sample_fn

        def make_v_star_fn(p_val):
            def v_star_fn(beta):
                return compute_v_star_bernoulli(p_val, beta)
            return v_star_fn

        t_start = time.time()
        df_p = run_experiment(
            sample_fn=make_sample_fn(p),
            v_star_fn=make_v_star_fn(p),
            betas=betas,
            n_tot_values=n_tot_values,
            t_trials=t_trials,
            single_n_values=single_n_values,
            multi_n_sets=multi_n_sets,
            seed=SEED,
            extra_columns={"p": p},
            beta_smooth_priors=beta_smooth_priors,
        )
        elapsed = time.time() - t_start
        print(f"\n  p={p} completed in {elapsed:.1f}s")

        all_dfs.append(df_p)

        output_dir_p = os.path.join(run_dir, f"p{p}")
        plot_results(df_p, output_dir_p, dist_label=f"Bernoulli(p={p})")

    df = pd.concat(all_dfs, ignore_index=True)
    csv_path = os.path.join(run_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved combined results to {csv_path}")

    total_time = time.time() - overall_start
    print(f"Total experiment time: {total_time:.1f}s")

    print("\n" + "=" * 60)
    print("Summary (largest N_tot, first beta, each p)")
    print("=" * 60)
    for p in p_values:
        df_sample = df[
            (df["p"] == p) &
            (df["beta"] == betas[0]) &
            (df["n_tot"] == n_tot_values[-1])
        ]
        print(f"\n  p={p}, beta={betas[0]}, N_tot={n_tot_values[-1]}:")
        for _, row in df_sample.head(5).iterrows():
            print(
                f"    {row['method']:>30s}:  "
                f"Bias={row['bias']:+.5f}  "
                f"Var={row['variance']:.5f}  "
                f"RMSE={row['rmse']:.5f}"
            )

    print(f"\nDone! Results in {run_dir}")


if __name__ == "__main__":
    main()
