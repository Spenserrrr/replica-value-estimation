"""
Estimators for V* = beta * log Z, where Z = E[exp(r / beta)].

1. LME — beta * log(mean(exp(r/beta))).
2. Single-replica — (beta/n) * log(mean block replica product) with block size n.
3. Multi-n slope — linear regression of log(mean block product) on n; V_hat = beta * slope.
4. Multi-n slope + jackknife — same, with jackknife-corrected log(mean block product) per n.
5. Beta-smoothed (Bernoulli) — Beta posterior mean for p, plugged into closed-form V*(p).

Uses scipy.special.logsumexp where needed for numerical stability.
"""

import numpy as np
from scipy.special import logsumexp


def estimate_lme(rewards: np.ndarray, beta: float) -> float:
    """
    Log-mean-exp estimate of V*.

    rewards: shape (N,). beta: KL / inverse temperature.
    """
    N = len(rewards)
    if N == 0:
        return np.nan

    # log(mean(exp(r/beta))) = logsumexp(r/beta) - log(N)
    scaled = rewards / beta
    return beta * (logsumexp(scaled) - np.log(N))


def _partition_and_compute_log_block_products(rewards: np.ndarray, beta: float, n: int) -> np.ndarray:
    """
    Non-overlapping blocks of size n; return log W_j = sum(r/beta) per block (empty if no full blocks).

    rewards: allocated samples. beta, n: scale and block size.
    """
    N_alloc = len(rewards)
    M_n = N_alloc // n

    if M_n == 0:
        return np.array([])

    used = rewards[: M_n * n]
    blocks = used.reshape(M_n, n)

    log_W = blocks.sum(axis=1) / beta

    return log_W


def _compute_log_psi_hat(log_W: np.ndarray) -> float:
    """
    log(mean_j W_j) = logsumexp(log_W) - log(M_n). NaN if empty.

    log_W: shape (M_n,).
    """
    M_n = len(log_W)
    if M_n == 0:
        return np.nan

    return logsumexp(log_W) - np.log(M_n)


def _compute_jackknife_phi(log_W: np.ndarray) -> float:
    """
    Jackknife bias-corrected phi_hat = log(mean W_j). NaN if M_n < 2.

    log_W: shape (M_n,).
    """
    M_n = len(log_W)
    if M_n < 2:
        return np.nan

    phi_hat = logsumexp(log_W) - np.log(M_n)

    # LOO sum via log(sum - W_j) = log_sum_all + log1p(-exp(log_W_j - log_sum_all))
    log_sum_all = logsumexp(log_W)

    phi_loo = np.empty(M_n)
    for j in range(M_n):
        ratio = np.exp(log_W[j] - log_sum_all)

        if ratio > 1.0 - 1e-15:
            # Near-total mass on one block; log1p path is ill-conditioned
            mask = np.ones(M_n, dtype=bool)
            mask[j] = False
            phi_loo[j] = logsumexp(log_W[mask]) - np.log(M_n - 1)
        else:
            log_sum_without_j = log_sum_all + np.log1p(-ratio)
            phi_loo[j] = log_sum_without_j - np.log(M_n - 1)

    y_jk = M_n * phi_hat - (M_n - 1) * np.mean(phi_loo)

    return y_jk


def estimate_single_replica(
    rewards: np.ndarray, beta: float, n: int
) -> float:
    """
    Single-replica V* with block size n. NaN if there are no full blocks.

    rewards, beta, n.
    """
    log_W = _partition_and_compute_log_block_products(rewards, beta, n)
    if len(log_W) == 0:
        return np.nan

    log_psi_hat = _compute_log_psi_hat(log_W)
    return (beta / n) * log_psi_hat


def estimate_multi_n_slope(
    rewards: np.ndarray,
    beta: float,
    replica_orders: list,
    use_jackknife: bool = False,
) -> float:
    """
    For each n in replica_orders, phi_hat(n) from the full sample; fit phi ~ a + b*n; return beta * b.

    rewards, beta, replica_orders, use_jackknife. NaN if fewer than two valid n.
    """
    N_tot = len(rewards)
    K = len(replica_orders)

    if K == 0 or N_tot == 0:
        return np.nan

    n_values = []
    phi_values = []

    for n in replica_orders:
        log_W = _partition_and_compute_log_block_products(rewards, beta, n)

        min_blocks = 2 if use_jackknife else 1
        if len(log_W) < min_blocks:
            continue

        if use_jackknife:
            phi_n = _compute_jackknife_phi(log_W)
        else:
            phi_n = _compute_log_psi_hat(log_W)

        if np.isfinite(phi_n):
            n_values.append(n)
            phi_values.append(phi_n)

    if len(n_values) < 2:
        return np.nan

    n_arr = np.array(n_values, dtype=float)
    phi_arr = np.array(phi_values)

    coeffs = np.polyfit(n_arr, phi_arr, 1)
    slope = coeffs[0]

    return beta * slope


def estimate_beta_smoothed(
    rewards: np.ndarray,
    beta: float,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """
    Bernoulli V* with smoothed p: p_tilde = (k+alpha)/(N+alpha+gamma), V_hat = beta*log(1 - p_tilde + p_tilde*exp(1/beta)).

    rewards: binary 0/1. beta, alpha, gamma: Beta(alpha, gamma) prior on p (defaults Laplace).
    """
    N = len(rewards)
    if N == 0:
        return np.nan

    k = rewards.sum()
    p_tilde = (k + alpha) / (N + alpha + gamma)
    return beta * np.log(1.0 - p_tilde + p_tilde * np.exp(1.0 / beta))
