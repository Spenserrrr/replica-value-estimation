"""
Experiment 2: contextual bandit simulation (A*PO Stage 1).

Per prompt x with pass rate p_x, runs T Monte Carlo trials of N Bernoulli(p_x)
rewards and all configured estimators. Returns one row per
(beta, N, method, prompt) with bias, variance, RMSE, advantage shift, and
log-ratio distortion for calibration and stratified analysis.
"""

import time
import numpy as np
import pandas as pd

from src.estimators import (
    estimate_lme,
    estimate_single_replica,
    estimate_multi_n_slope,
    estimate_beta_smoothed,
)
from src.metrics import compute_all_metrics


def _run_trials_for_prompt(
    rng: np.random.Generator,
    p_x: float,
    beta: float,
    n_samples: int,
    t_trials: int,
    estimator_configs: list,
) -> dict:
    """
    T trials for one prompt: each callable in ``estimator_configs`` has
    signature ``(rewards, beta) -> float``.
    """
    all_rewards = rng.binomial(1, p_x, size=(t_trials, n_samples)).astype(float)

    results = {name: np.empty(t_trials) for name, _ in estimator_configs}

    for t in range(t_trials):
        rewards = all_rewards[t]
        for name, estimator_fn in estimator_configs:
            results[name][t] = estimator_fn(rewards, beta)

    return results


def build_estimator_configs(
    single_n_values: list,
    multi_n_sets: list,
    beta_smooth_priors: list = None,
) -> list:
    """
    List of ``(name, fn)`` with ``fn(rewards, beta) -> float``.

    Each multi-n set adds plain and jackknife slope estimators.
    """
    configs = []

    configs.append(("lme", lambda r, b: estimate_lme(r, b)))

    for n in single_n_values:
        configs.append((
            f"single_n{n}",
            lambda r, b, _n=n: estimate_single_replica(r, b, _n),
        ))

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

    if beta_smooth_priors:
        for name, alpha, gamma in beta_smooth_priors:
            configs.append((
                f"beta_smooth_{name}",
                lambda r, b, _a=alpha, _g=gamma: estimate_beta_smoothed(r, b, alpha=_a, gamma=_g),
            ))

    return configs


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
    beta_smooth_priors: list = None,
) -> pd.DataFrame:
    """
    Full sweep over beta and N; V*(x) from Bernoulli closed form per beta.

    ``beta2`` scales log-ratio distortion ``|advantage_shift| / beta2``.
    """
    from src.ground_truth import compute_v_star_bernoulli_vec

    rng = np.random.default_rng(seed)
    M = len(p_array)

    estimator_configs = build_estimator_configs(
        single_n_values, multi_n_sets, beta_smooth_priors=beta_smooth_priors
    )
    method_names = [name for name, _ in estimator_configs]

    total_configs = len(betas) * len(n_samples_values)
    config_idx = 0
    rows = []

    for beta in betas:
        v_star = compute_v_star_bernoulli_vec(p_array, beta)

        for n_samples in n_samples_values:
            config_idx += 1
            t_start = time.time()

            all_estimates = {name: np.empty((M, t_trials)) for name in method_names}

            for xi in range(M):
                trial_results = _run_trials_for_prompt(
                    rng, p_array[xi], beta, n_samples, t_trials, estimator_configs,
                )
                for name in method_names:
                    all_estimates[name][xi, :] = trial_results[name]

            for name in method_names:
                for xi in range(M):
                    estimates_t = all_estimates[name][xi, :]
                    metrics = compute_all_metrics(estimates_t, v_star[xi])
                    n_valid = int(np.sum(np.isfinite(estimates_t)))
                    mean_est = np.nanmean(estimates_t)

                    # advantage_shift = V*(x) - mean_estimate = -bias
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
