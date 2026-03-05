"""
Experiment 2 runner: contextual bandit simulation of A*PO Stage 1.

For each prompt x with pass rate p_x, we run T independent Monte Carlo trials.
Each trial draws N Bernoulli(p_x) rewards and applies all configured estimators.
Per-prompt metrics (bias, variance, RMSE) are then computed from the T estimates.

The runner returns a DataFrame with one row per (beta, N, method, prompt),
ready for calibration plotting, stratified analysis, and distortion computation.
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


# =============================================================================
# Per-prompt trial runner
# =============================================================================

def _run_trials_for_prompt(
    rng: np.random.Generator,
    p_x: float,
    beta: float,
    n_samples: int,
    t_trials: int,
    estimator_configs: list,
) -> dict:
    """
    Run T independent trials for a single prompt.

    For each trial: sample N rewards from Bernoulli(p_x),
    then apply every estimator in estimator_configs.

    Parameters
    ----------
    rng : NumPy random generator.
    p_x : Pass rate for this prompt.
    beta : KL regularization temperature.
    n_samples : Number of reward samples per trial (N).
    t_trials : Number of Monte Carlo trials (T).
    estimator_configs : List of (name, callable) pairs.
        Each callable takes (rewards, beta) and returns a float estimate.

    Returns
    -------
    dict mapping method_name -> np.ndarray of shape (T,) with the T estimates.
    """
    # Pre-generate all rewards at once: shape (T, N)
    all_rewards = rng.binomial(1, p_x, size=(t_trials, n_samples)).astype(float)

    results = {name: np.empty(t_trials) for name, _ in estimator_configs}

    for t in range(t_trials):
        rewards = all_rewards[t]
        for name, estimator_fn in estimator_configs:
            results[name][t] = estimator_fn(rewards, beta)

    return results


# =============================================================================
# Build estimator configs from user-specified parameters
# =============================================================================

def build_estimator_configs(single_n_values: list, multi_n_sets: list) -> list:
    """
    Build a list of (name, callable) estimator configs.

    Each callable has signature (rewards, beta) -> float.

    Parameters
    ----------
    single_n_values : e.g. [4, 8]
    multi_n_sets : e.g. [[2,4,6,8], [2,3,4]]
        For each set, both plain and jackknife versions are created.

    Returns
    -------
    List of (name, fn) tuples.
    """
    configs = []

    # LME baseline
    configs.append(("lme", lambda r, b: estimate_lme(r, b)))

    # Single-replica estimators
    for n in single_n_values:
        configs.append((
            f"single_n{n}",
            lambda r, b, _n=n: estimate_single_replica(r, b, _n),
        ))

    # Multi-n slope estimators (with and without jackknife)
    for orders in multi_n_sets:
        key = str(orders)
        configs.append((
            f"multi_{key}",
            lambda r, b, _o=orders: estimate_multi_n_slope(r, b, _o, use_jackknife=False),
        ))
        configs.append((
            f"multi_{key}_jk",
            lambda r, b, _o=orders: estimate_multi_n_slope(r, b, _o, use_jackknife=True),
        ))

    return configs


# =============================================================================
# Main experiment runner
# =============================================================================

def run_experiment2(
    p_array: np.ndarray,
    strata: np.ndarray,
    betas: list,
    n_samples_values: list,
    t_trials: int,
    single_n_values: list,
    multi_n_sets: list,
    seed: int,
    beta2: float = 1e-3,
) -> pd.DataFrame:
    """
    Run the full Experiment 2 sweep.

    For each (beta, N), runs T trials per prompt, computes all estimators,
    and records per-prompt metrics. Ground-truth V*(x) is computed
    internally for each beta using the Bernoulli closed form.

    Parameters
    ----------
    p_array : np.ndarray of shape (M,), pass rates for each prompt.
    strata : np.ndarray of shape (M,), stratum labels per prompt.
    betas : List of beta values to sweep.
    n_samples_values : List of per-prompt sample budgets N.
    t_trials : Number of Monte Carlo trials per (prompt, beta, N).
    single_n_values : Replica orders for single-n estimators.
    multi_n_sets : Order sets for multi-n slope estimators.
    seed : Random seed.
    beta2 : Stage 2 KL coefficient for advantage distortion (default 1e-3).

    Returns
    -------
    pd.DataFrame with columns:
        beta, n_samples, method, prompt_idx, p_x, stratum, v_star,
        mean_estimate, bias, variance, rmse, n_valid,
        advantage_shift, log_ratio_distortion
    """
    from src.ground_truth import compute_v_star_bernoulli_vec

    rng = np.random.default_rng(seed)
    M = len(p_array)

    estimator_configs = build_estimator_configs(single_n_values, multi_n_sets)
    method_names = [name for name, _ in estimator_configs]

    total_configs = len(betas) * len(n_samples_values)
    config_idx = 0
    rows = []

    for beta in betas:
        # Recompute V*(x) for this beta
        v_star = compute_v_star_bernoulli_vec(p_array, beta)

        for n_samples in n_samples_values:
            config_idx += 1
            t_start = time.time()

            # Run all prompts for this (beta, N) configuration
            # all_estimates[method_name] = np.ndarray of shape (M, T)
            all_estimates = {name: np.empty((M, t_trials)) for name in method_names}

            for xi in range(M):
                trial_results = _run_trials_for_prompt(
                    rng, p_array[xi], beta, n_samples, t_trials, estimator_configs,
                )
                for name in method_names:
                    all_estimates[name][xi, :] = trial_results[name]

            # Compute per-prompt metrics for each method
            for name in method_names:
                for xi in range(M):
                    estimates_t = all_estimates[name][xi, :]
                    metrics = compute_all_metrics(estimates_t, v_star[xi])
                    n_valid = int(np.sum(np.isfinite(estimates_t)))
                    mean_est = np.nanmean(estimates_t)

                    # Advantage shift: V*(x) - mean_estimate = -bias
                    adv_shift = v_star[xi] - mean_est
                    log_ratio_dist = abs(adv_shift) / beta2

                    rows.append({
                        "beta": beta,
                        "n_samples": n_samples,
                        "method": name,
                        "prompt_idx": xi,
                        "p_x": p_array[xi],
                        "stratum": strata[xi],
                        "v_star": v_star[xi],
                        "mean_estimate": mean_est,
                        "bias": metrics["bias"],
                        "variance": metrics["variance"],
                        "rmse": metrics["rmse"],
                        "n_valid": n_valid,
                        "advantage_shift": adv_shift,
                        "log_ratio_distortion": log_ratio_dist,
                    })

            elapsed = time.time() - t_start
            print(
                f"  [{config_idx}/{total_configs}] "
                f"beta={beta}, N={n_samples:>3d} — "
                f"{elapsed:.1f}s  ({M} prompts × {t_trials} trials)"
            )

    return pd.DataFrame(rows)
