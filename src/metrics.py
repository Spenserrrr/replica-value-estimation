"""
Evaluation metrics for comparing V* estimators.

Given T independent Monte Carlo trials, each producing an estimate V_hat^(t),
and the known ground-truth V*, we compute:

    Bias     = mean(V_hat) - V*
    Variance = (1 / (T-1)) * sum((V_hat^(t) - mean(V_hat))^2)
    RMSE     = sqrt( (1/T) * sum((V_hat^(t) - V*)^2) )

Note: RMSE^2 = Bias^2 + Variance (approximately, for large T).
"""

import numpy as np


def compute_bias(estimates: np.ndarray, v_star: float) -> float:
    """
    Compute the bias of an estimator.

    Parameters:
        estimates : np.ndarray with shape (T,), containing V* estimates from T independent trials.
        v_star : The true V*.

    Returns:
        Bias = mean(estimates) - v_star.
    """
    return np.nanmean(estimates) - v_star


def compute_variance(estimates: np.ndarray) -> float:
    """
    Compute the sample variance of an estimator (using ddof=1).

    Parameters:
        estimates : np.ndarray with shape (T,), containing V* estimates from T independent trials.

    Returns:
        Sample variance.
    """
    valid = estimates[~np.isnan(estimates)]
    if len(valid) < 2:
        return np.nan
    return np.var(valid, ddof=1)


def compute_rmse(estimates: np.ndarray, v_star: float) -> float:
    """
    Compute the root mean squared error (RMSE) of an estimator.

    Parameters:
        estimates : np.ndarray with shape (T,), containing V* estimates from T independent trials.
        v_star : The true V*.

    Returns:
        RMSE = sqrt(mean((estimates - v_star)^2)).
    """
    valid = estimates[~np.isnan(estimates)]
    if len(valid) == 0:
        return np.nan
    return np.sqrt(np.nanmean((valid - v_star) ** 2))


def compute_all_metrics(estimates: np.ndarray, v_star: float) -> dict:
    """
    Compute all three metrics (bias, variance, RMSE) at once.

    Parameters: 
    estimates : np.ndarray with shape (T,), containing V* estimates from T independent trials.
    v_star : The true V*.

    Returns:
        Dictionary with keys 'bias', 'variance', 'rmse'.
    """
    return {
        "bias": compute_bias(estimates, v_star),
        "variance": compute_variance(estimates),
        "rmse": compute_rmse(estimates, v_star),
    }

