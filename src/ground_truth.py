"""
Ground-truth computation for V* under Gaussian rewards.

For the KL-regularized RL objective, the optimal value function is:
    V*(x) = beta * log Z(x),
where
    Z(x) = E_{y ~ pi_ref}[exp(r(x,y) / beta)].

In the scalar Gaussian setting (no context x), we have:
    r ~ N(mu_r, sigma_r^2)
    Z   = E[exp(r / beta)] = exp(mu_r / beta + sigma_r^2 / (2 * beta^2))
    V*  = beta * log Z     = mu_r + sigma_r^2 / (2 * beta)

This closed-form expression allows us to evaluate estimator quality exactly.
"""

import numpy as np


def compute_v_star_gaussian(mu_r: float, sigma_r: float, beta: float) -> float:
    """
    Compute the exact V* for Gaussian rewards.

    Parameters:
        mu_r : Mean of the Gaussian reward distribution.
        sigma_r : Standard deviation of the Gaussian reward distribution.
        beta : KL regularization parameter (temperature). Must be > 0.

    Returns
        The exact optimal value V* = mu_r + sigma_r^2 / (2 * beta).
    """
    return mu_r + (sigma_r ** 2) / (2.0 * beta)


def compute_log_Z_gaussian(mu_r: float, sigma_r: float, beta: float) -> float:
    """
    Compute log Z for Gaussian rewards.

    Parameters:
        mu_r : Mean of the Gaussian reward distribution.
        sigma_r : Standard deviation of the Gaussian reward distribution.
        beta : KL regularization parameter. Must be > 0.

    Returns:
        log Z = mu_r / beta + sigma_r^2 / (2 * beta^2).
    """
    return mu_r / beta + (sigma_r ** 2) / (2.0 * beta ** 2)

