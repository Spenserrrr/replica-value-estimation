"""
Ground-truth computation for V* under known reward distributions.

For the KL-regularized RL objective, the optimal value function is:
    V*(x) = beta * log Z(x),
where
    Z(x) = E_{y ~ pi_ref}[exp(r(x,y) / beta)].

This module provides closed-form V* and log(Z) for two reward models:

1. Gaussian:  r ~ N(mu_r, sigma_r^2)
       Z  = exp(mu_r / beta + sigma_r^2 / (2 * beta^2))
       V* = mu_r + sigma_r^2 / (2 * beta)

2. Bernoulli: r ~ Bernoulli(p), r in {0, 1}
       Z  = 1 - p + p * exp(1 / beta)
       V* = beta * log(1 - p + p * exp(1 / beta))
"""

import numpy as np


# =============================================================================
# Gaussian reward model
# =============================================================================

def compute_v_star_gaussian(mu_r: float, sigma_r: float, beta: float) -> float:
    """
    Compute the exact V* for Gaussian rewards.

    Parameters:
        mu_r : Mean of the Gaussian reward distribution.
        sigma_r : Standard deviation of the Gaussian reward distribution.
        beta : KL regularization parameter (temperature). Must be > 0.

    Returns:
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


# =============================================================================
# Bernoulli reward model
# =============================================================================

def compute_v_star_bernoulli(p: float, beta: float) -> float:
    """
    Compute the exact V* for Bernoulli rewards.

    r ~ Bernoulli(p), so r = 1 with probability p, r = 0 otherwise.

    Z  = E[exp(r / beta)] = (1 - p) + p * exp(1 / beta)
    V* = beta * log(Z)    = beta * log(1 - p + p * exp(1 / beta))

    Parameters:
        p : Success probability of the Bernoulli distribution. Must be in [0, 1].
        beta : KL regularization parameter. Must be > 0.

    Returns:
        The exact optimal value V*.
    """
    return beta * np.log(1.0 - p + p * np.exp(1.0 / beta))


def compute_log_Z_bernoulli(p: float, beta: float) -> float:
    """
    Compute log Z for Bernoulli rewards.

    Parameters:
        p : Success probability. Must be in [0, 1].
        beta : KL regularization parameter. Must be > 0.

    Returns:
        log Z = log(1 - p + p * exp(1 / beta)).
    """
    return np.log(1.0 - p + p * np.exp(1.0 / beta))
