#!/usr/bin/env python3
"""
Experiment 1: Pure Scalar Estimation of V*

This script evaluates four estimators of V* = beta * log Z in a fully
controlled scalar setting where the ground truth is known analytically.

Reward model:
    r ~ N(mu_r, sigma_r^2)
    V* = mu_r + sigma_r^2 / (2 * beta)

Estimators compared (under matched sample budget N_tot):
    1. LME (log-mean-exp)           — standard biased baseline
    2. Single-replica (n=n_fixed)    — fixed replica order
    3. Multi-n slope (no jackknife)  — linear fit of phi_hat(n) vs n
    4. Multi-n slope (with jackknife)— jackknife-corrected version of (3)

For each (beta, N_tot), we run T independent Monte Carlo trials and report:
    - Bias   = mean(V_hat) - V*
    - Variance = sample variance of V_hat across trials
    - RMSE  = sqrt(mean((V_hat - V*)^2))

Usage:
    python run_experiment1.py [--quick]

The --quick flag runs a reduced configuration for faster iteration.
Results (CSV + plots) are saved to the results/ directory.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 100,  # Reduced from 150 to avoid >2000px images
})

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.estimators import (
    estimate_lme,
    estimate_single_replica,
    estimate_multi_n_slope,
)
from src.ground_truth import compute_v_star_gaussian
from src.metrics import compute_all_metrics


# =============================================================================
# Experiment Configuration
# =============================================================================

# ---- Reward distribution ----
MU_R = 0.0         # Reward mean (zero-centered for simplicity)
SIGMA_R = 1.0      # Reward standard deviation

# ---- KL regularization parameters to sweep ----
BETAS = [0.5, 1.0, 2.0]

# ---- Sample budgets to sweep (log-spaced, extended range) ----
N_TOT_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]

# ---- Monte Carlo trials ----
T_TRIALS = 1000

# ---- Single-replica: sweep multiple replica orders ----
# We'll test each of these separately
SINGLE_REPLICA_N_VALUES = [2, 3, 4, 5, 8]

# ---- Multi-n slope: test multiple sets of replica orders ----
# Each set represents a different choice of orders to fit the linear model
MULTI_N_ORDER_SETS = [
    [2, 3],           # Minimal set
    [2, 3, 4],        # Small set
    [2, 3, 4, 5],     # Original set
    [2, 3, 4, 5, 6],  # Extended set
    [2, 4, 6, 8],     # Even orders only
]

# ---- Random seed for reproducibility ----
SEED = 42

# ---- Output directory ----
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# =============================================================================
# Quick mode (for fast iteration / debugging)
# =============================================================================

def get_quick_config():
    """Return a reduced configuration for quick testing."""
    return {
        "betas": [0.5, 1.0, 2.0],  # Use all betas even in quick mode
        "n_tot_values": [32, 128, 512],
        "t_trials": 50,
        "single_n_values": [2, 4],  # Test fewer n values
        "multi_n_sets": [[2, 3, 4, 5]],  # Test just one set
    }


# =============================================================================
# Main Experiment Loop
# =============================================================================

def run_single_trial(
    rng: np.random.Generator,
    beta: float,
    n_tot: int,
    single_n_values: list,
    multi_n_sets: list,
) -> dict:
    """
    Run a single Monte Carlo trial: draw N_tot samples and compute estimates for all methods.

    Each estimator receives the same N_tot i.i.d. reward samples for fair comparison.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator (for reproducibility).
    beta : float
        KL regularization parameter.
    n_tot : int
        Total sample budget.
    single_n_values : list of int
        Replica orders to test for single-replica estimator (e.g., [2, 3, 4, 5, 8]).
    multi_n_sets : list of lists
        Sets of replica orders for multi-n slope estimators (e.g., [[2,3,4], [2,4,6,8]]).

    Returns
    -------
    dict
        Keys: 'lme', 'single_n2', 'single_n3', ..., 'multi_[2,3,4]', 'multi_[2,3,4]_jk', ...
        Values: estimated V* from each method configuration.
    """
    # Draw N_tot i.i.d. reward samples
    rewards = rng.normal(MU_R, SIGMA_R, size=n_tot)

    results = {
        "lme": estimate_lme(rewards, beta),
    }
    
    # Test each single-replica order
    for n in single_n_values:
        results[f"single_n{n}"] = estimate_single_replica(rewards, beta, n)
    
    # Test each multi-n set (with and without jackknife)
    for orders in multi_n_sets:
        set_key = str(orders)  # e.g., "[2, 3, 4, 5]"
        results[f"multi_{set_key}"] = estimate_multi_n_slope(
            rewards, beta, orders, use_jackknife=False
        )
        results[f"multi_{set_key}_jk"] = estimate_multi_n_slope(
            rewards, beta, orders, use_jackknife=True
        )
    
    return results


def run_experiment(
    betas: list,
    n_tot_values: list,
    t_trials: int,
    single_n_values: list,
    multi_n_sets: list,
    seed: int,
) -> pd.DataFrame:
    """
    Run the full Experiment 1 sweep over (beta, N_tot) configurations.

    Parameters
    ----------
    betas : list of float
        KL regularization parameters to sweep.
    n_tot_values : list of int
        Sample budgets to sweep.
    t_trials : int
        Number of Monte Carlo trials per configuration.
    single_n_values : list of int
        Replica orders to test for single-replica estimator.
    multi_n_sets : list of lists
        Sets of replica orders for multi-n slope estimators.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Results table with columns:
        [beta, n_tot, method, bias, variance, rmse, v_star, mean_estimate, n_valid]
    """
    rng = np.random.default_rng(seed)
    results = []

    total_configs = len(betas) * len(n_tot_values)
    config_idx = 0

    for beta in betas:
        v_star = compute_v_star_gaussian(MU_R, SIGMA_R, beta)

        for n_tot in n_tot_values:
            config_idx += 1
            t_start = time.time()

            # Run T trials and collect all method estimates
            all_trial_results = []
            for t in range(t_trials):
                trial_results = run_single_trial(
                    rng, beta, n_tot, single_n_values, multi_n_sets
                )
                all_trial_results.append(trial_results)
            
            # Get all method names from first trial
            method_names = list(all_trial_results[0].keys())
            
            # Compute metrics for each method
            for method in method_names:
                estimates = np.array([trial[method] for trial in all_trial_results])
                metrics = compute_all_metrics(estimates, v_star)
                n_valid = np.sum(np.isfinite(estimates))

                results.append({
                    "beta": beta,
                    "n_tot": n_tot,
                    "method": method,
                    "bias": metrics["bias"],
                    "variance": metrics["variance"],
                    "rmse": metrics["rmse"],
                    "v_star": v_star,
                    "mean_estimate": np.nanmean(estimates),
                    "n_valid": int(n_valid),
                })

            elapsed = time.time() - t_start
            print(
                f"  [{config_idx}/{total_configs}] "
                f"beta={beta}, N_tot={n_tot:>5d} — "
                f"{elapsed:.2f}s  (V*={v_star:.4f})"
            )

    return pd.DataFrame(results)


# =============================================================================
# Plotting
# =============================================================================

# ---- Distinct color palettes for each method family ----
# Each family gets its own palette so lines within a family are easy to distinguish,
# and the LME baseline (black) stands out clearly across all plots.

LME_STYLE = {"color": "black", "marker": "o", "linestyle": "-", "linewidth": 2.5, "markersize": 7}

# Single-replica: warm oranges/reds with dashed lines
SINGLE_COLORS = ["#e6550d", "#fd8d3c", "#d62728", "#9467bd", "#8c564b"]
SINGLE_MARKERS = ["s", "^", "D", "v", "p"]

# Multi-n (no jackknife): cool blues/greens with dash-dot lines
MULTI_COLORS = ["#1f77b4", "#2ca02c", "#17becf", "#31a354", "#756bb1"]
MULTI_MARKERS = ["s", "^", "D", "v", "p"]

# Multi-n (jackknife): same hues but lighter, with dotted lines
MULTI_JK_COLORS = ["#6baed6", "#74c476", "#9ecae1", "#a1d99b", "#bcbddc"]
MULTI_JK_MARKERS = ["s", "^", "D", "v", "p"]

METRIC_LABELS = {
    "bias": "Bias",
    "variance": "Variance",
    "rmse": "RMSE",
}


def _classify_methods(df: pd.DataFrame):
    """
    Classify all methods in the DataFrame into three families, each with
    its own style mapping. LME is included in every family as a baseline.

    Returns
    -------
    dict
        Keys: family name ("single_replica", "multi_n", "multi_n_jk")
        Values: list of (method_key, label, plot_kwargs) tuples.
    """
    all_methods = sorted(df["method"].unique())

    # Collect method keys by family
    single_keys = [m for m in all_methods if m.startswith("single_n")]
    multi_keys = [m for m in all_methods if m.startswith("multi_") and not m.endswith("_jk")]
    multi_jk_keys = [m for m in all_methods if m.endswith("_jk")]

    # Sort single-replica by n value
    single_keys.sort(key=lambda m: int(m.replace("single_n", "")))

    families = {}

    # ---- Family 1: Single-replica ----
    items = [("lme", "LME (baseline)", LME_STYLE)]
    for i, key in enumerate(single_keys):
        n = key.replace("single_n", "")
        items.append((key, f"Single-replica (n={n})", {
            "color": SINGLE_COLORS[i % len(SINGLE_COLORS)],
            "marker": SINGLE_MARKERS[i % len(SINGLE_MARKERS)],
            "linestyle": "--",
            "linewidth": 2,
            "markersize": 7,
        }))
    families["single_replica"] = items

    # ---- Family 2: Multi-n slope (no jackknife) ----
    items = [("lme", "LME (baseline)", LME_STYLE)]
    for i, key in enumerate(multi_keys):
        orders_str = key.replace("multi_", "")
        items.append((key, f"Multi-n {orders_str}", {
            "color": MULTI_COLORS[i % len(MULTI_COLORS)],
            "marker": MULTI_MARKERS[i % len(MULTI_MARKERS)],
            "linestyle": "-.",
            "linewidth": 2,
            "markersize": 7,
        }))
    families["multi_n"] = items

    # ---- Family 3: Multi-n slope (with jackknife) ----
    items = [("lme", "LME (baseline)", LME_STYLE)]
    for i, key in enumerate(multi_jk_keys):
        orders_str = key.replace("multi_", "").replace("_jk", "")
        items.append((key, f"Multi-n {orders_str} + JK", {
            "color": MULTI_JK_COLORS[i % len(MULTI_JK_COLORS)],
            "marker": MULTI_JK_MARKERS[i % len(MULTI_JK_MARKERS)],
            "linestyle": ":",
            "linewidth": 2.2,
            "markersize": 7,
        }))
    families["multi_n_jk"] = items

    return families


# Human-readable titles for each method family
FAMILY_TITLES = {
    "single_replica": "Single-Replica Estimators",
    "multi_n":        "Multi-n Slope Estimators (no Jackknife)",
    "multi_n_jk":     "Multi-n Slope Estimators (with Jackknife)",
}


def _plot_one(
    ax: plt.Axes,
    df_beta: pd.DataFrame,
    metric: str,
    family_items: list,
    show_ylabel: bool = True,
):
    """Plot one (metric, beta, family) panel onto the given axes."""
    for method_key, label, style in family_items:
        df_m = df_beta[df_beta["method"] == method_key]
        if df_m.empty:
            continue

        x = df_m["n_tot"].values
        y = df_m[metric].values
        mask = df_m["n_valid"].values > 0
        if not np.any(mask):
            continue

        ax.plot(x[mask], y[mask], label=label, **style)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("$N_{\\mathrm{tot}}$", fontsize=10)

    if show_ylabel:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)

    if metric == "bias":
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    elif metric in ("variance", "rmse"):
        ax.set_yscale("log")

    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


# Ordered list of families for the triptych (left to right)
FAMILY_ORDER = ["single_replica", "multi_n", "multi_n_jk"]

# Short titles for panel headers (space-constrained)
FAMILY_SHORT_TITLES = {
    "single_replica": "Single-Replica",
    "multi_n":        "Multi-n Slope",
    "multi_n_jk":     "Multi-n Slope + Jackknife",
}


def plot_results(df: pd.DataFrame, output_dir: str):
    """
    Generate side-by-side triptych plots: one figure per (metric, beta).
    Each figure has 3 panels sharing the same y-axis:
        [Single-Replica | Multi-n Slope | Multi-n Slope + JK]

    LME (baseline) appears in every panel as a black solid line for reference.

    Total output: 3 metrics × len(betas) figures.

    Parameters
    ----------
    df : pd.DataFrame
        Results from run_experiment().
    output_dir : str
        Directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    betas = sorted(df["beta"].unique())
    metrics = ["bias", "variance", "rmse"]
    families = _classify_methods(df)

    n_panels = len(FAMILY_ORDER)
    total_plots = len(metrics) * len(betas)
    print(f"\nGenerating {total_plots} triptych plots (3 panels each) ...")

    for metric in metrics:
        for beta in betas:
            df_beta = df[df["beta"] == beta]
            v_star = df_beta["v_star"].iloc[0]

            # Create figure with shared y-axis.
            # Width: ~6 inches per panel × 3 = 18 inches.
            # At 100 DPI → 1800px base, but sharey saves space on middle/right
            # panels, so effective width after tight_layout is ~1600-1750px.
            fig, axes = plt.subplots(
                1, n_panels,
                figsize=(18, 5),
                sharey=True,
            )

            for col_idx, family_name in enumerate(FAMILY_ORDER):
                ax = axes[col_idx]
                family_items = families[family_name]
                short_title = FAMILY_SHORT_TITLES[family_name]

                # Only show y-axis label on leftmost panel
                _plot_one(
                    ax, df_beta, metric, family_items,
                    show_ylabel=(col_idx == 0),
                )
                ax.set_title(short_title, fontsize=11, fontweight="bold")

                # Adjust tick label size for compactness
                ax.tick_params(axis="both", labelsize=9)

            # Overall title
            fig.suptitle(
                f"{METRIC_LABELS[metric]}  —  $\\beta = {beta}$  ($V^* = {v_star:.3f}$)",
                fontsize=13, y=1.02,
            )

            fig.tight_layout()

            filename = f"{metric}_beta{beta}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, bbox_inches="tight", dpi=100)
            plt.close(fig)
            print(f"  {filepath}")

    plt.close("all")
    print(f"\nAll {total_plots} plots saved to {output_dir}/")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Pure Scalar V* Estimation")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a reduced configuration for quick testing."
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate plots from existing CSV without re-running trials."
    )
    args = parser.parse_args()

    # ---- Plot-only mode: skip experiment, just regenerate plots ----
    if args.plot_only:
        csv_path = os.path.join(RESULTS_DIR, "experiment1_results.csv")
        if not os.path.exists(csv_path):
            print(f"ERROR: No results CSV found at {csv_path}")
            print("Run the experiment first (without --plot-only).")
            return
        print(f"Loading existing results from {csv_path}")
        df = pd.read_csv(csv_path)
        # Clean old plots
        for f in os.listdir(RESULTS_DIR):
            if f.endswith(".png"):
                os.remove(os.path.join(RESULTS_DIR, f))
        print("Generating plots...")
        plot_results(df, RESULTS_DIR)
        print("\nDone!")
        return

    # Select configuration
    if args.quick:
        cfg = get_quick_config()
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

    # Print experiment configuration
    print("=" * 60)
    print("Experiment 1: Pure Scalar Estimation of V*")
    print("=" * 60)
    print(f"  Reward distribution:  N({MU_R}, {SIGMA_R}^2)")
    print(f"  Beta values:          {betas}")
    print(f"  N_tot values:         {n_tot_values}")
    print(f"  Monte Carlo trials:   {t_trials}")
    print(f"  Single-replica n values: {single_n_values}")
    print(f"  Multi-n order sets:   {multi_n_sets}")
    print(f"  Random seed:          {SEED}")
    print()

    # Print ground truth for each beta
    print("Ground truth V* values:")
    for beta in betas:
        v_star = compute_v_star_gaussian(MU_R, SIGMA_R, beta)
        print(f"  beta={beta}:  V* = {v_star:.6f}")
    print()

    # Run experiment
    print("Running Monte Carlo trials...")
    t_start = time.time()
    df = run_experiment(
        betas=betas,
        n_tot_values=n_tot_values,
        t_trials=t_trials,
        single_n_values=single_n_values,
        multi_n_sets=multi_n_sets,
        seed=SEED,
    )
    total_time = time.time() - t_start
    print(f"\nTotal experiment time: {total_time:.1f}s")

    # Save raw results to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "experiment1_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Print summary table (abbreviated - full results in CSV)
    print("\n" + "=" * 60)
    print("Summary of Results (sample)")
    print("=" * 60)
    print("Showing first few methods for beta={}...".format(betas[0]))
    df_sample = df[(df["beta"] == betas[0]) & (df["n_tot"] == n_tot_values[-1])]
    for _, row in df_sample.head(10).iterrows():
        print(
            f"  {row['method']:>30s}:  "
            f"Bias={row['bias']:+.5f}  "
            f"Var={row['variance']:.5f}  "
            f"RMSE={row['rmse']:.5f}"
        )
    print(f"\n(Full results saved to {csv_path})")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(df, RESULTS_DIR)
    print("\nDone!")


if __name__ == "__main__":
    main()

