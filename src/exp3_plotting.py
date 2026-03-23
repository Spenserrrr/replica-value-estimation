"""
Output utilities for Experiment 3: policy learning in the toy bandit.

Produces:
1. Training loss curves (PNG): loss vs SGD step for each estimator.
2. Parameter trajectory plots (PNG): (w, b) convergence over SGD steps.
3. Policy comparison scatter (PNG): learned q_x vs optimal q*_x per prompt.
4. Summary tables (CSV): aggregate metrics, stratified breakdowns.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 100,
})

ESTIMATOR_COLORS = {
    "oracle": "#2ca02c",
    "jeffreys": "#1f77b4",
    "lme": "#d62728",
    "camel_90": "#9467bd",
    "camel_80": "#8c564b",
    "camel_95": "#e377c2",
}
ESTIMATOR_LABELS = {
    "oracle": "Oracle",
    "jeffreys": "Jeffreys (MSE)",
    "lme": "LME (MSE)",
    "camel_90": "CAMeL 90% CI",
    "camel_80": "CAMeL 80% CI",
    "camel_95": "CAMeL 95% CI",
}
STRATUM_ORDER = ["very_hard", "hard", "medium", "easy"]


def _est_label(est):
    return ESTIMATOR_LABELS.get(est, est)


def _est_color(est):
    return ESTIMATOR_COLORS.get(est, "#333333")


# =============================================================================
# 1. Training loss curves
# =============================================================================

def plot_loss_curves(curves: dict, output_dir: str, dist_label: str = ""):
    """
    Plot mean training loss vs SGD step for each estimator.

    Parameters
    ----------
    curves : dict mapping estimator name -> list of dicts with "loss_curve".
             Each loss_curve is a list of (step, loss) tuples.
    output_dir : directory to save the plot.
    dist_label : label for the difficulty regime.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for est in curves:
        all_curves = curves[est]
        if not all_curves:
            continue

        steps = np.array([s for s, _ in all_curves[0]["loss_curve"]])
        loss_matrix = np.array([
            [loss for _, loss in trial["loss_curve"]]
            for trial in all_curves
        ])
        mean_loss = np.mean(loss_matrix, axis=0)
        std_loss = np.std(loss_matrix, axis=0)

        color = _est_color(est)
        ax.plot(steps, mean_loss, color=color, linewidth=2, label=_est_label(est))
        ax.fill_between(
            steps, mean_loss - std_loss, mean_loss + std_loss,
            color=color, alpha=0.15,
        )

    ax.set_xlabel("SGD step")
    ax.set_ylabel("Regression loss")
    ax.set_yscale("log")
    title = "Stage 2 Training Loss"
    if dist_label:
        title += f" — {dist_label}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    path = os.path.join(output_dir, "training_loss.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"    [plot] {path}")


# =============================================================================
# 2. Parameter trajectory plots
# =============================================================================

def plot_param_trajectories(
    curves: dict, beta2: float, output_dir: str, dist_label: str = ""
):
    """
    Plot w and b convergence over SGD steps, with optimal values marked.
    """
    fig, (ax_w, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    w_star, b_star = 1.0, 1.0 / beta2

    for est in curves:
        all_curves = curves[est]
        if not all_curves:
            continue

        steps = np.array([s for s, _ in all_curves[0]["w_curve"]])
        w_matrix = np.array([[w for _, w in trial["w_curve"]] for trial in all_curves])
        b_matrix = np.array([[b for _, b in trial["b_curve"]] for trial in all_curves])

        color = _est_color(est)
        mean_w = np.mean(w_matrix, axis=0)
        mean_b = np.mean(b_matrix, axis=0)

        ax_w.plot(steps, mean_w, color=color, linewidth=2, label=_est_label(est))
        ax_b.plot(steps, mean_b, color=color, linewidth=2, label=_est_label(est))

    ax_w.axhline(w_star, color="gray", linestyle="--", linewidth=1, label=f"$w^\\star = {w_star}$")
    ax_b.axhline(b_star, color="gray", linestyle="--", linewidth=1, label=f"$b^\\star = {b_star}$")

    ax_w.set_xlabel("SGD step")
    ax_w.set_ylabel("$w$")
    ax_w.set_title("Weight $w$ convergence")
    ax_w.legend(fontsize=8)
    ax_w.grid(True, alpha=0.2)

    ax_b.set_xlabel("SGD step")
    ax_b.set_ylabel("$b$")
    ax_b.set_title("Bias $b$ convergence")
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.2)

    suptitle = "Parameter Trajectories"
    if dist_label:
        suptitle += f" — {dist_label}"
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "param_trajectories.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"    [plot] {path}")


# =============================================================================
# 3. Policy comparison scatter
# =============================================================================

def plot_policy_scatter(per_prompt_df: pd.DataFrame, output_dir: str, dist_label: str = ""):
    """
    Scatter plot: learned q_x vs optimal q*_x, one panel per estimator.
    Points colored by difficulty stratum.
    """
    estimators = per_prompt_df["estimator"].unique()
    n_est = len(estimators)

    fig, axes = plt.subplots(1, n_est, figsize=(6 * n_est, 5.5), squeeze=False)

    stratum_colors = {
        "very_hard": "#d62728", "hard": "#ff7f0e",
        "medium": "#2ca02c", "easy": "#1f77b4",
    }

    for idx, est in enumerate(estimators):
        ax = axes[0][idx]
        df_e = per_prompt_df[per_prompt_df["estimator"] == est]

        for stratum in STRATUM_ORDER:
            df_s = df_e[df_e["stratum"] == stratum]
            if df_s.empty:
                continue
            ax.scatter(
                df_s["q_star"], df_s["q_learned"],
                c=stratum_colors[stratum], s=15, alpha=0.5,
                label=stratum.replace("_", " ").title(), edgecolors="none",
            )

        lo = min(df_e["q_star"].min(), df_e["q_learned"].min())
        hi = max(df_e["q_star"].max(), df_e["q_learned"].max())
        margin = (hi - lo) * 0.05
        lims = [lo - margin, hi + margin]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Optimal $q^\\star_x$")
        ax.set_ylabel("Learned $q(x; \\hat{w}, \\hat{b})$")
        ax.set_title(_est_label(est), fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", markerscale=2)

    suptitle = "Learned vs Optimal Policy"
    if dist_label:
        suptitle += f" — {dist_label}"
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "policy_scatter.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"    [plot] {path}")


# =============================================================================
# 4. Summary tables
# =============================================================================

def compute_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-trial metrics into mean +/- std for each estimator.
    """
    rows = []
    for est, grp in summary_df.groupby("estimator"):
        rows.append({
            "estimator": est,
            "label": _est_label(est),
            "mean_reward": grp["expected_reward"].mean(),
            "std_reward": grp["expected_reward"].std(),
            "mean_kl": grp["mean_kl"].mean(),
            "std_kl": grp["mean_kl"].std(),
            "mean_objective": grp["objective"].mean(),
            "std_objective": grp["objective"].std(),
            "mean_policy_error": grp["policy_error"].mean(),
            "std_policy_error": grp["policy_error"].std(),
            "mean_param_error": grp["param_error"].mean(),
            "std_param_error": grp["param_error"].std(),
            "mean_w": grp["w_final"].mean(),
            "std_w": grp["w_final"].std(),
            "mean_b": grp["b_final"].mean(),
            "std_b": grp["b_final"].std(),
        })
    return pd.DataFrame(rows)


def compute_stratified_policy_table(per_prompt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified per-prompt policy error for each estimator and stratum.
    """
    rows = []
    for (est, stratum), grp in per_prompt_df.groupby(["estimator", "stratum"]):
        rows.append({
            "estimator": est,
            "label": _est_label(est),
            "stratum": stratum,
            "n_prompts": len(grp),
            "mean_q_error": grp["q_error"].mean(),
            "mean_q_learned": grp["q_learned"].mean(),
            "mean_q_star": grp["q_star"].mean(),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Main entry point
# =============================================================================

def generate_all_exp3_outputs(
    results: dict,
    output_dir: str,
    beta2: float,
    dist_label: str = "",
):
    """
    Generate all Experiment 3 outputs: plots and CSVs.

    Parameters
    ----------
    results : dict returned by run_experiment3().
    output_dir : directory to save outputs.
    beta2 : Stage 2 temperature (for parameter trajectory reference lines).
    dist_label : label for the difficulty regime.
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_df = results["summary_df"]
    curves = results["curves"]
    per_prompt_df = results["per_prompt_df"]

    # Plots
    print(f"\n  Generating Experiment 3 plots ...")
    plot_loss_curves(curves, output_dir, dist_label)
    plot_param_trajectories(curves, beta2, output_dir, dist_label)
    plot_policy_scatter(per_prompt_df, output_dir, dist_label)

    # Tables
    print(f"\n  Computing summary tables ...")
    agg_df = compute_summary_table(summary_df)
    path = os.path.join(output_dir, "summary_metrics.csv")
    agg_df.to_csv(path, index=False, float_format="%.6f")
    print(f"    [csv] {path}")

    strat_df = compute_stratified_policy_table(per_prompt_df)
    path = os.path.join(output_dir, "stratified_policy_error.csv")
    strat_df.to_csv(path, index=False, float_format="%.6f")
    print(f"    [csv] {path}")

    # Save raw summary
    path = os.path.join(output_dir, "trial_results.csv")
    summary_df.to_csv(path, index=False, float_format="%.6f")
    print(f"    [csv] {path}")

    plt.close("all")
    print(f"\n  All Experiment 3 outputs saved to {output_dir}/")
