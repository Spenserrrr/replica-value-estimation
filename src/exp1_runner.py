"""
Monte Carlo runner for V* estimators.

Draws i.i.d. rewards via ``sample_fn``, runs LME, single-replica, multi-n slope,
and optional beta-smoothed estimators on the same samples, then aggregates bias,
variance, and RMSE over T trials. Parametrized by ``sample_fn`` and ``v_star_fn``
so Gaussian, Bernoulli, or other rewards reuse the same code path.
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


def run_single_trial(
    rng: np.random.Generator,
    sample_fn,
    beta: float,
    n_tot: int,
    single_n_values: list,
    multi_n_sets: list,
    beta_smooth_priors: list = None,
) -> dict:
    """
    One trial: draw ``n_tot`` rewards, run all estimators on that draw.

    All methods share the same reward vector. ``beta_smooth_priors`` is
    ``(name, alpha, gamma)`` tuples for Bernoulli-only beta smoothing.
    """
    rewards = sample_fn(rng, n_tot)

    results = {
        "lme": estimate_lme(rewards, beta),
    }

    for n in single_n_values:
        results[f"single_n{n}"] = estimate_single_replica(rewards, beta, n)

    for orders in multi_n_sets:
        set_key = str(orders)
        results[f"multi_{set_key}"] = estimate_multi_n_slope(
            rewards, beta, orders, use_jackknife=False
        )
        results[f"multi_{set_key}_jk"] = estimate_multi_n_slope(
            rewards, beta, orders, use_jackknife=True
        )

    if beta_smooth_priors:
        for name, alpha, gamma in beta_smooth_priors:
            results[f"beta_smooth_{name}"] = estimate_beta_smoothed(
                rewards, beta, alpha=alpha, gamma=gamma
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
    beta_smooth_priors: list = None,
) -> pd.DataFrame:
    """
    Sweep (beta, n_tot): T trials each, metrics vs. ``v_star_fn(beta)``.

    ``extra_columns`` is merged into every row (e.g. tagging Bernoulli p).
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

            all_trial_results = []
            for t in range(t_trials):
                trial_results = run_single_trial(
                    rng, sample_fn, beta, n_tot, single_n_values, multi_n_sets,
                    beta_smooth_priors=beta_smooth_priors,
                )
                all_trial_results.append(trial_results)

            method_names = list(all_trial_results[0].keys())

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
