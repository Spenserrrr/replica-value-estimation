"""
Bias, sample variance (ddof=1), and RMSE for Monte Carlo V* estimates.
Bias = mean(V_hat) - V*; RMSE = sqrt(mean((V_hat - V*)^2)). NaNs ignored where noted.
"""

import numpy as np


def compute_bias(estimates: np.ndarray, v_star: float) -> float:
    """Mean estimate minus v_star (uses nanmean on estimates)."""
    return np.nanmean(estimates) - v_star


def compute_variance(estimates: np.ndarray) -> float:
    """Sample variance with ddof=1; NaN if fewer than two finite values."""
    valid = estimates[~np.isnan(estimates)]
    if len(valid) < 2:
        return np.nan
    return np.var(valid, ddof=1)


def compute_rmse(estimates: np.ndarray, v_star: float) -> float:
    """sqrt(mean((estimates - v_star)^2)) on finite values; NaN if none."""
    valid = estimates[~np.isnan(estimates)]
    if len(valid) == 0:
        return np.nan
    return np.sqrt(np.nanmean((valid - v_star) ** 2))


def compute_all_metrics(estimates: np.ndarray, v_star: float) -> dict:
    """Returns dict with keys bias, variance, rmse."""
    return {
        "bias": compute_bias(estimates, v_star),
        "variance": compute_variance(estimates),
        "rmse": compute_rmse(estimates, v_star),
    }
