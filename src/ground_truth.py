"""
Closed-form V* = beta * log Z and log Z for KL-regularized objectives with
Gaussian rewards, Bernoulli rewards, and vectorized Bernoulli pass rates (contextual bandit).
"""

import numpy as np


def compute_v_star_gaussian(mu_r: float, sigma_r: float, beta: float) -> float:
    """V* = mu_r + sigma_r^2 / (2*beta). beta > 0."""
    return mu_r + (sigma_r ** 2) / (2.0 * beta)


def compute_log_Z_gaussian(mu_r: float, sigma_r: float, beta: float) -> float:
    """log Z = mu_r/beta + sigma_r^2/(2*beta^2). beta > 0."""
    return mu_r / beta + (sigma_r ** 2) / (2.0 * beta ** 2)


def compute_v_star_bernoulli(p: float, beta: float) -> float:
    """Bernoulli(p): Z = (1-p) + p*exp(1/beta), V* = beta*log(Z). p in [0,1], beta > 0."""
    return beta * np.log(1.0 - p + p * np.exp(1.0 / beta))


def compute_log_Z_bernoulli(p: float, beta: float) -> float:
    """log Z = log((1-p) + p*exp(1/beta))."""
    return np.log(1.0 - p + p * np.exp(1.0 / beta))


def compute_v_star_bernoulli_vec(p_array: np.ndarray, beta: float) -> np.ndarray:
    """Elementwise V* for pass rates p_array (shape (M,)). Same formula as compute_v_star_bernoulli."""
    return beta * np.log(1.0 - p_array + p_array * np.exp(1.0 / beta))


def generate_prompt_pass_rates(
    M: int, a: float, b: float, rng: np.random.Generator
) -> np.ndarray:
    """M draws from Beta(a, b), clipped to (0.001, 0.999) for numerical stability."""
    p = rng.beta(a, b, size=M)
    return np.clip(p, 0.001, 0.999)


def assign_strata(p_array: np.ndarray) -> np.ndarray:
    """Stratum labels from pass rate: <0.05 very_hard; [0.05,0.15) hard; [0.15,0.35) medium; else easy."""
    strata = np.empty(len(p_array), dtype=object)
    strata[p_array < 0.05] = "very_hard"
    strata[(p_array >= 0.05) & (p_array < 0.15)] = "hard"
    strata[(p_array >= 0.15) & (p_array < 0.35)] = "medium"
    strata[p_array >= 0.35] = "easy"
    return strata
