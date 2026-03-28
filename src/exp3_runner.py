"""
Experiment 3: toy contextual bandit, two-stage policy learning (A*PO Stage 2).

Stage 1 estimates V*(x) from N Bernoulli samples per prompt. Stage 2 trains
shared parameters q(x)=sigmoid(w·logit(p_x)+b) with on-policy SGD on targets
r - V_hat(x). Loss modes: MSE on a point estimate, or CAMeL (interval target
from Jeffreys Beta credible intervals on p_x).
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from src.ground_truth import compute_v_star_bernoulli_vec


def _sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _logit(p):
    """Log-odds; clip p to avoid inf."""
    p_clipped = np.clip(p, 1e-10, 1.0 - 1e-10)
    return np.log(p_clipped / (1.0 - p_clipped))


def _p_to_vstar(p, beta2):
    """Bernoulli closed form V* from pass rate."""
    p_safe = np.clip(p, 1e-10, 1.0 - 1e-10)
    return beta2 * np.log(1.0 - p_safe + p_safe * np.exp(1.0 / beta2))


def _draw_stage1_samples(rng, p_array, n_samples):
    """Bernoulli samples per prompt; returns success counts k (length M)."""
    M = len(p_array)
    rewards = rng.binomial(1, p_array[:, None] * np.ones((M, n_samples)))
    return rewards.sum(axis=1).astype(float)


def estimate_v_star_stage1(rng, p_array, n_samples, beta2, estimator):
    """
    Stage 1: sample, estimate p, map to V_hat at beta2 (no beta mismatch vs. Stage 2).

    ``estimator``: ``oracle`` | ``lme`` | ``jeffreys``.
    """
    if estimator == "oracle":
        return compute_v_star_bernoulli_vec(p_array, beta2)

    k = _draw_stage1_samples(rng, p_array, n_samples)

    if estimator == "lme":
        p_hat = k / n_samples
    elif estimator == "jeffreys":
        p_hat = (k + 0.5) / (n_samples + 1.0)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    return _p_to_vstar(p_hat, beta2)


def estimate_v_star_with_interval(rng, p_array, n_samples, beta2, confidence=0.90):
    """
    Jeffreys posterior Beta(k+0.5, N-k+0.5); map credible interval for p through V*.

    V* is monotone in p, so the p-interval maps to [v_lower, v_upper].
    """
    k = _draw_stage1_samples(rng, p_array, n_samples)

    p_hat = (k + 0.5) / (n_samples + 1.0)

    alpha_post = k + 0.5
    beta_post = n_samples - k + 0.5

    alpha_level = (1.0 - confidence) / 2.0
    p_lower = beta_dist.ppf(alpha_level, alpha_post, beta_post)
    p_upper = beta_dist.ppf(1.0 - alpha_level, alpha_post, beta_post)

    return _p_to_vstar(p_hat, beta2), _p_to_vstar(p_lower, beta2), _p_to_vstar(p_upper, beta2)


def _compute_mse_loss_and_grad(w, b, logit_p, p_array, v_hat, beta2, rng):
    """One SGD step: MSE (log_ratio - (r - v_hat))^2."""
    z = w * logit_p + b
    q = _sigmoid(z)

    r = rng.binomial(1, q).astype(float)

    log_ratio = np.where(
        r == 1,
        beta2 * np.log(np.clip(q / p_array, 1e-30, None)),
        beta2 * np.log(np.clip((1.0 - q) / (1.0 - p_array), 1e-30, None)),
    )

    target = r - v_hat
    residual = log_ratio - target
    loss = float(np.sum(residual ** 2))

    d_lograt_dz = np.where(r == 1, beta2 * (1.0 - q), -beta2 * q)
    common = 2.0 * residual * d_lograt_dz
    grad_w = float(np.sum(common * logit_p))
    grad_b = float(np.sum(common))

    return loss, grad_w, grad_b


def _compute_camel_loss_and_grad(w, b, logit_p, p_array, v_lower, v_upper, beta2, rng):
    """
    Interval target [r - v_upper, r - v_lower]; squared distance outside the band,
    zero gradient inside (dead zone).
    """
    z = w * logit_p + b
    q = _sigmoid(z)

    r = rng.binomial(1, q).astype(float)

    log_ratio = np.where(
        r == 1,
        beta2 * np.log(np.clip(q / p_array, 1e-30, None)),
        beta2 * np.log(np.clip((1.0 - q) / (1.0 - p_array), 1e-30, None)),
    )

    target_lo = r - v_upper
    target_hi = r - v_lower

    residual = np.where(
        log_ratio < target_lo, log_ratio - target_lo,
        np.where(log_ratio > target_hi, log_ratio - target_hi, 0.0),
    )

    loss = float(np.sum(residual ** 2))

    d_lograt_dz = np.where(r == 1, beta2 * (1.0 - q), -beta2 * q)
    common = 2.0 * residual * d_lograt_dz
    grad_w = float(np.sum(common * logit_p))
    grad_b = float(np.sum(common))

    return loss, grad_w, grad_b


def run_stage2_shared(
    logit_p, p_array, beta2, t_sgd, lr, seed,
    loss_mode="mse",
    v_hat=None,
    v_lower=None, v_upper=None,
    record_every=10,
):
    """
    Shared (w, b) policy SGD. ``loss_mode``: ``mse`` (needs ``v_hat``) or
    ``camel`` (needs ``v_lower``, ``v_upper``).
    """
    rng = np.random.default_rng(seed)

    w, b = 1.0, 0.0
    loss_curve, w_curve, b_curve = [], [], []

    for t in range(t_sgd):
        if loss_mode == "mse":
            loss, grad_w, grad_b = _compute_mse_loss_and_grad(
                w, b, logit_p, p_array, v_hat, beta2, rng
            )
        elif loss_mode == "camel":
            loss, grad_w, grad_b = _compute_camel_loss_and_grad(
                w, b, logit_p, p_array, v_lower, v_upper, beta2, rng
            )
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        w -= lr * grad_w
        b -= lr * grad_b

        if t % record_every == 0 or t == t_sgd - 1:
            loss_curve.append((t, loss))
            w_curve.append((t, w))
            b_curve.append((t, b))

    return {
        "w_final": w, "b_final": b,
        "loss_curve": loss_curve, "w_curve": w_curve, "b_curve": b_curve,
    }


def evaluate_policy(w, b, logit_p, p_array, beta2):
    """Metrics for q = sigmoid(w*logit(p)+b) vs. closed-form optimal q* per prompt."""
    z = w * logit_p + b
    q = _sigmoid(z)
    q_star = p_array * np.exp(1.0 / beta2) / (p_array * np.exp(1.0 / beta2) + (1.0 - p_array))

    q_clip = np.clip(q, 1e-10, 1.0 - 1e-10)
    p_clip = np.clip(p_array, 1e-10, 1.0 - 1e-10)
    kl_per_prompt = (
        q_clip * np.log(q_clip / p_clip)
        + (1.0 - q_clip) * np.log((1.0 - q_clip) / (1.0 - p_clip))
    )

    expected_reward = float(np.mean(q))
    mean_kl = float(np.mean(kl_per_prompt))
    objective = expected_reward - beta2 * mean_kl
    policy_error = float(np.mean(np.abs(q - q_star)))
    w_star, b_star = 1.0, 1.0 / beta2
    param_error = float(np.sqrt((w - w_star) ** 2 + (b - b_star) ** 2))

    return {
        "q": q, "q_star": q_star,
        "expected_reward": expected_reward, "mean_kl": mean_kl,
        "objective": objective, "policy_error": policy_error,
        "param_error": param_error, "w": w, "b": b,
    }


def run_experiment3(
    p_array, strata, beta2, n_samples, t_trials, t_sgd, lr,
    conditions, seed, record_every=10,
):
    """
    Each condition dict: ``name``, ``estimator`` (``lme`` | ``jeffreys`` | ``oracle``),
    ``loss_mode`` (``mse`` | ``camel``), ``confidence`` for CAMeL (ignored for MSE).

    Returns ``summary_df``, ``curves``, ``per_prompt_df`` (last trial per condition).
    """
    M = len(p_array)
    logit_p = _logit(p_array)

    summary_rows = []
    curves = {c["name"]: [] for c in conditions}
    last_per_prompt = {}

    total_runs = len(conditions) * t_trials
    run_idx = 0
    t_start = time.time()

    for cond in conditions:
        cname = cond["name"]
        est = cond["estimator"]
        loss_mode = cond["loss_mode"]
        confidence = cond.get("confidence", 0.90)

        for trial in range(t_trials):
            run_idx += 1

            trial_seed = seed + trial * 1000 + hash(cname) % 10000
            stage1_rng = np.random.default_rng(trial_seed)
            sgd_seed = trial_seed + 500

            if loss_mode == "camel" and est != "oracle":
                v_hat, v_lo, v_hi = estimate_v_star_with_interval(
                    stage1_rng, p_array, n_samples, beta2, confidence
                )
                result = run_stage2_shared(
                    logit_p, p_array, beta2, t_sgd, lr, sgd_seed,
                    loss_mode="camel", v_lower=v_lo, v_upper=v_hi,
                    record_every=record_every,
                )
            else:
                v_hat = estimate_v_star_stage1(
                    stage1_rng, p_array, n_samples, beta2, est
                )
                result = run_stage2_shared(
                    logit_p, p_array, beta2, t_sgd, lr, sgd_seed,
                    loss_mode="mse", v_hat=v_hat,
                    record_every=record_every,
                )

            eval_result = evaluate_policy(
                result["w_final"], result["b_final"],
                logit_p, p_array, beta2,
            )

            summary_rows.append({
                "estimator": cname,
                "trial": trial,
                "w_final": result["w_final"],
                "b_final": result["b_final"],
                "expected_reward": eval_result["expected_reward"],
                "mean_kl": eval_result["mean_kl"],
                "objective": eval_result["objective"],
                "policy_error": eval_result["policy_error"],
                "param_error": eval_result["param_error"],
            })

            curves[cname].append({
                "loss_curve": result["loss_curve"],
                "w_curve": result["w_curve"],
                "b_curve": result["b_curve"],
            })

            if trial == t_trials - 1:
                last_per_prompt[cname] = {
                    "q": eval_result["q"],
                    "q_star": eval_result["q_star"],
                    "v_hat": v_hat,
                }

            if run_idx % 50 == 0 or run_idx == total_runs:
                elapsed = time.time() - t_start
                print(
                    f"  [{run_idx}/{total_runs}] "
                    f"{cname}, trial {trial+1}/{t_trials} — "
                    f"{elapsed:.1f}s elapsed"
                )

    summary_df = pd.DataFrame(summary_rows)

    per_prompt_rows = []
    for cname, data in last_per_prompt.items():
        for xi in range(M):
            per_prompt_rows.append({
                "estimator": cname,
                "prompt_idx": xi,
                "p_x": p_array[xi],
                "stratum": strata[xi],
                "q_learned": data["q"][xi],
                "q_star": data["q_star"][xi],
                "q_error": abs(data["q"][xi] - data["q_star"][xi]),
                "v_hat": data["v_hat"][xi],
                "v_star": compute_v_star_bernoulli_vec(
                    np.array([p_array[xi]]), beta2
                )[0],
            })
    per_prompt_df = pd.DataFrame(per_prompt_rows)

    return {
        "summary_df": summary_df,
        "curves": curves,
        "per_prompt_df": per_prompt_df,
    }
