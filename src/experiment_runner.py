"""
Generic Monte Carlo experiment runner for V* estimators.

This module provides a distribution-agnostic runner that:
1. Draws samples using a provided sample function
2. Applies all configured estimators (LME, single-replica, multi-n slope)
3. Collects metrics (bias, variance, RMSE) across T independent trials
4. Returns results as a pandas DataFrame

The runner is parametrized by:
    - sample_fn(rng, n_tot) -> np.ndarray : draws n_tot i.i.d. reward samples
    - v_star_fn(beta) -> float            : returns ground-truth V* for given beta

This design allows the same runner to handle Gaussian, Bernoulli, or any
other reward distribution without modification.
"""

import time
import numpy as np
import pandas as pd

from src.estimators import (
    estimate_lme,
    estimate_single_replica,
    estimate_multi_n_slope,
)
from src.metrics import compute_all_metrics


def run_single_trial(
    rng: np.random.Generator,
    sample_fn,
    beta: float,
    n_tot: int,
    single_n_values: list,
    multi_n_sets: list,
) -> dict:
    """
    Run a single Monte Carlo trial: draw N_tot samples and compute all estimators.

    Each estimator receives the SAME N_tot i.i.d. reward samples for fair comparison.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator (for reproducibility).
    sample_fn : callable(rng, n_tot) -> np.ndarray
        Function that draws n_tot i.i.d. reward samples.
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
    # Draw N_tot i.i.d. reward samples using the provided sample function
    rewards = sample_fn(rng, n_tot)

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
    sample_fn,
    v_star_fn,
    betas: list,
    n_tot_values: list,
    t_trials: int,
    single_n_values: list,
    multi_n_sets: list,
    seed: int,
    extra_columns: dict = None,
) -> pd.DataFrame:
    """
    Run the full experiment sweep over (beta, N_tot) configurations.

    For each (beta, N_tot), runs T independent Monte Carlo trials, computes
    all estimator outputs, and then evaluates bias, variance, and RMSE
    against the ground-truth V*.

    Parameters
    ----------
    sample_fn : callable(rng, n_tot) -> np.ndarray
        Draws n_tot i.i.d. reward samples. The rng argument ensures
        reproducibility.
    v_star_fn : callable(beta) -> float
        Returns the ground-truth V* for a given beta.
    betas : list of float
        KL regularization parameters to sweep.
    n_tot_values : list of int
        Sample budgets to sweep.
    t_trials : int
        Number of Monte Carlo trials per (beta, N_tot) configuration.
    single_n_values : list of int
        Replica orders for single-replica estimator.
    multi_n_sets : list of lists
        Sets of replica orders for multi-n slope estimators.
    seed : int
        Random seed for reproducibility.
    extra_columns : dict, optional
        Additional columns to add to every row of the output DataFrame.
        Useful for tagging results with distribution parameters,
        e.g., {"p": 0.1} for Bernoulli.

    Returns
    -------
    pd.DataFrame
        Results table with columns:
        [beta, n_tot, method, bias, variance, rmse, v_star, mean_estimate, n_valid]
        Plus any extra_columns.
    """
    rng = np.random.default_rng(seed)
    results = []

    total_configs = len(betas) * len(n_tot_values)
    config_idx = 0

    for beta in betas:
        v_star = v_star_fn(beta)

        for n_tot in n_tot_values:
            config_idx += 1
            t_start = time.time()

            # Run T trials and collect all method estimates
            all_trial_results = []
            for t in range(t_trials):
                trial_results = run_single_trial(
                    rng, sample_fn, beta, n_tot, single_n_values, multi_n_sets
                )
                all_trial_results.append(trial_results)

            # Get all method names from first trial
            method_names = list(all_trial_results[0].keys())

            # Compute metrics for each method
            for method in method_names:
                estimates = np.array([trial[method] for trial in all_trial_results])
                metrics = compute_all_metrics(estimates, v_star)
                n_valid = np.sum(np.isfinite(estimates))

                row = {
                    "beta": beta,
                    "n_tot": n_tot,
                    "method": method,
                    "bias": metrics["bias"],
                    "variance": metrics["variance"],
                    "rmse": metrics["rmse"],
                    "v_star": v_star,
                    "mean_estimate": np.nanmean(estimates),
                    "n_valid": int(n_valid),
                }
                # Add any extra columns (e.g., distribution parameters)
                if extra_columns:
                    row.update(extra_columns)
                results.append(row)

            elapsed = time.time() - t_start
            print(
                f"  [{config_idx}/{total_configs}] "
                f"beta={beta}, N_tot={n_tot:>5d} — "
                f"{elapsed:.2f}s  (V*={v_star:.4f})"
            )

    return pd.DataFrame(results)

