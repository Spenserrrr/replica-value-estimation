"""
Estimators for V* = beta * log Z, where Z = E[exp(r / beta)].

This module implements four estimators:

1. **LME (Log-Mean-Exp)**: The standard biased estimator.
       V_hat = beta * log( (1/N) * sum_i exp(r_i / beta) )

2. **Single-Replica**: Uses a fixed integer replica order n > 1.
   Partitions N samples into M = floor(N/n) blocks of size n.
   For each block j, computes the "replica product" W_j = prod_k exp(r_{j,k}/beta).
   Then estimates V* = (beta / n) * log(mean(W_j)).

3. **Multi-n Slope (no jackknife)**: For a set of replica orders N = {n_1,...,n_K},
   computes phi_hat(n) = log(mean(W_j)) for each n, then fits a linear model
   phi_hat(n) ~ a + b*n. Returns V_hat = beta * b.

4. **Multi-n Slope (with jackknife)**: Same as (3), but applies delete-one jackknife
   bias correction to each phi_hat(n) before fitting the linear model.
   This reduces the O(1/M_n) bias to O(1/M_n^2).

All estimators use the same total sample budget N_tot for fair comparison.
Numerical stability is ensured via scipy.special.logsumexp throughout.
"""

import numpy as np
from scipy.special import logsumexp


# =============================================================================
# Estimator 1: Log-Mean-Exp (LME) Baseline
# =============================================================================

def estimate_lme(rewards: np.ndarray, beta: float) -> float:
    """
    Log-Mean-Exp estimator for V*.

    This is the standard estimator used in A*PO Stage 1:
        V_hat_LME = beta * log( (1/N) * sum_i exp(r_i / beta) )

    Parameters:
        rewards : np.ndarray with shape (N,), containing i.i.d. reward samples from the reward distribution.
        beta : KL regularization parameter. Must be > 0.

    Returns:
        Estimated V*.
    """
    N = len(rewards)
    if N == 0:
        return np.nan

    # Compute in log-space for numerical stability:
    # log(mean(exp(r/beta))) = logsumexp(r/beta) - log(N)
    scaled = rewards / beta
    return beta * (logsumexp(scaled) - np.log(N))


# =============================================================================
# Helper: Block Partitioning and Log-Moment Computation
# =============================================================================

def _partition_and_compute_log_block_products(rewards: np.ndarray, beta: float, n: int) -> np.ndarray:
    """
    Partition rewards into blocks of size n, compute log of replica products.

    For each block j of n rewards {r_{j,1}, ..., r_{j,n}}, the replica product is:
        W_j = prod_{k=1}^{n} exp(r_{j,k} / beta)
            = exp( sum_{k=1}^{n} r_{j,k} / beta )

    So log(W_j) = sum_{k=1}^{n} r_{j,k} / beta, which is computed as the sum of scaled rewards within each block.

    Parameters
        rewards : np.ndarray with shape (N_alloc,), containing reward samples allocated to this replica order.
        beta : KL regularization parameter.
        n : Replica order (block size). Must be >= 1.

    Returns:
        log_W : np.ndarray with shape (M_n,), containing log of replica products for each block, where M_n = floor(N_alloc / n).
        Returns empty array if M_n == 0.
    """
    N_alloc = len(rewards)
    M_n = N_alloc // n

    if M_n == 0:
        return np.array([])

    # Use exactly M_n * n samples (discard any remainder)
    used = rewards[: M_n * n]
    blocks = used.reshape(M_n, n)  # shape (M_n, n)

    # log(W_j) = sum of (r_{j,k} / beta) within each block
    log_W = blocks.sum(axis=1) / beta  # shape (M_n,)

    return log_W


def _compute_log_psi_hat(log_W: np.ndarray) -> float:
    """
    Compute log of the sample mean of block products: log(psi_hat(n)).

    psi_hat(n) = (1/M_n) * sum_j W_j = (1/M_n) * sum_j exp(log_W_j)

    So log(psi_hat(n)) = logsumexp(log_W) - log(M_n).

    This estimates Z^n, since E[W_j] = Z^n for independent replicas.

    Parameters
        log_W : np.ndarray with shape (M_n,), containing log of replica block products.

    Returns:
        log(psi_hat(n)), or NaN if input is empty.
    """
    M_n = len(log_W)
    if M_n == 0:
        return np.nan

    return logsumexp(log_W) - np.log(M_n)


def _compute_jackknife_phi(log_W: np.ndarray) -> float:
    """
    Compute jackknife bias-corrected version of phi_hat(n) = log(psi_hat(n)).

    Procedure:
        1. Compute the full estimate: phi_hat = log(psi_hat) = logsumexp(log_W) - log(M_n)
        2. For each j, compute leave-one-out: phi_hat^{(-j)} = logsumexp(log_W_{i!=j}) - log(M_n-1)
        3. Jackknife correction:
             y_JK = M_n * phi_hat - (M_n - 1) * mean_j( phi_hat^{(-j)} )

    Parameters
        log_W : np.ndarray with shape (M_n,), containing log of replica block products.

    Returns:
        float: Jackknife-corrected phi_hat(n), or NaN if M_n < 2.
    """
    M_n = len(log_W)
    if M_n < 2:
        return np.nan

    # Full estimate
    phi_hat = logsumexp(log_W) - np.log(M_n)

    # Leave-one-out estimates using numerically stable computation:
    #   log(sum_all - W_j) = log(sum_all) + log(1 - exp(log_W_j - log(sum_all)))
    #                      = log_sum_all + log1p(-exp(log_W_j - log_sum_all))
    log_sum_all = logsumexp(log_W)  # log of sum of all W_j

    phi_loo = np.empty(M_n)
    for j in range(M_n):
        ratio = np.exp(log_W[j] - log_sum_all)  # W_j / sum_all, in [0, 1)

        if ratio > 1.0 - 1e-15:
            # This block dominates the sum; fall back to direct computation
            # (exclude j-th element and recompute)
            mask = np.ones(M_n, dtype=bool)
            mask[j] = False
            phi_loo[j] = logsumexp(log_W[mask]) - np.log(M_n - 1)
        else:
            log_sum_without_j = log_sum_all + np.log1p(-ratio)
            phi_loo[j] = log_sum_without_j - np.log(M_n - 1)

    # Jackknife bias correction formula
    y_jk = M_n * phi_hat - (M_n - 1) * np.mean(phi_loo)

    return y_jk


# =============================================================================
# Estimator 2: Single-Replica Estimator
# =============================================================================

def estimate_single_replica(
    rewards: np.ndarray, beta: float, n: int
) -> float:
    """
    Single-replica estimator with a fixed replica order n.

    Partitions N_tot rewards into M_n = floor(N_tot / n) blocks of size n.
    For each block j, computes the replica product W_j = prod exp(r/beta).
    Then:
        psi_hat(n) = mean(W_j)  ~  Z^n
        V_hat = (beta / n) * log(psi_hat(n))  ~  (beta / n) * n * log(Z) = V*

    Parameters
        rewards : np.ndarray with shape (N_tot,), containing i.i.d. reward samples.
        beta : KL regularization parameter.
        n : Replica order (block size). Must be >= 2.

    Returns:
        float: Estimated V*.
    """
    log_W = _partition_and_compute_log_block_products(rewards, beta, n)
    if len(log_W) == 0:
        return np.nan

    log_psi_hat = _compute_log_psi_hat(log_W)
    return (beta / n) * log_psi_hat


# =============================================================================
# Estimator 3 & 4: Multi-n Slope Estimator (with/without Jackknife)
# =============================================================================

def estimate_multi_n_slope(
    rewards: np.ndarray,
    beta: float,
    replica_orders: list,
    use_jackknife: bool = False,
) -> float:
    """
    Multi-n slope estimator for V*.

    Each replica order n re-partitions the FULL set of N_tot samples into
    blocks of size n, yielding M_n = floor(N_tot / n) blocks. This ensures
    each phi_hat(n) is estimated from the maximum available data. The phi_hat
    values for different n are correlated (since they share the same raw samples),
    but this does not affect the consistency of the slope estimate.

    Then fits a linear model:
        phi_hat(n) ~ a + b * n

    and returns V_hat = beta * b.

    Parameters
        rewards : np.ndarray with shape (N_tot,), containing i.i.d. reward samples.
        beta : KL regularization parameter.
        replica_orders : list of int, containing set of replica orders N, e.g. [2, 3, 4, 5]. Each must be >= 2.
        use_jackknife : bool, if True, apply jackknife bias correction to each phi_hat(n). Requires M_n >= 2 for each replica order.

    Returns:
        float: Estimated V*. Returns NaN if fewer than 2 valid replica orders.
    """
    N_tot = len(rewards)
    K = len(replica_orders)

    if K == 0 or N_tot == 0:
        return np.nan

    # ---- Compute phi_hat(n) for each replica order ----
    # Each replica order uses ALL N_tot samples, partitioned into blocks of size n.
    n_values = []
    phi_values = []

    for n in replica_orders:
        # Partition the full N_tot samples into blocks of size n
        log_W = _partition_and_compute_log_block_products(rewards, beta, n)

        # Check minimum block count
        min_blocks = 2 if use_jackknife else 1
        if len(log_W) < min_blocks:
            continue  # skip this replica order (insufficient samples)

        # Compute phi_hat(n), with or without jackknife
        if use_jackknife:
            phi_n = _compute_jackknife_phi(log_W)
        else:
            phi_n = _compute_log_psi_hat(log_W)

        if np.isfinite(phi_n):
            n_values.append(n)
            phi_values.append(phi_n)

    # ---- Fit linear model: phi(n) = a + b * n ----
    if len(n_values) < 2:
        return np.nan  # not enough points for regression

    n_arr = np.array(n_values, dtype=float)
    phi_arr = np.array(phi_values)

    # np.polyfit(x, y, 1) returns [slope, intercept]
    coeffs = np.polyfit(n_arr, phi_arr, 1)
    slope = coeffs[0]

    return beta * slope

