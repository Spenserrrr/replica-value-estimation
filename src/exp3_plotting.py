"""
Experiment 3 plots and tables: training loss, parameter error trajectories,
policy scatter (q vs q*), and CSV summaries.
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

EXCLUDE_ESTIMATORS = {"camel_90", "camel_80", "camel_95"}


def _est_label(est):
    return ESTIMATOR_LABELS.get(est, est)


def _est_color(est):
    return ESTIMATOR_COLORS.get(est, "#333333")


def _should_include(est):
    return est not in EXCLUDE_ESTIMATORS


EXCLUDE_FROM_LOSS = {"oracle"}


def plot_loss_curves(curves: dict, output_dir: str, dist_label: str = ""):
    """Mean training loss vs SGD step (Oracle excluded for legibility)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for est in curves:
        if not _should_include(est) or est in EXCLUDE_FROM_LOSS:
            continue
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
    title = "Stage 2 Training Loss (LME vs Jeffreys)"
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


def plot_param_trajectories(
    curves: dict, beta2: float, output_dir: str, dist_label: str = ""
):
    """Parameter error ||(w,b) - (w*,b*)||_2 over SGD steps, log scale."""
    fig, ax = plt.subplots(figsize=(8, 5))

    w_star, b_star = 1.0, 1.0 / beta2

    for est in curves:
        if not _should_include(est):
            continue
        all_curves = curves[est]
        if not all_curves:
            continue

        steps = np.array([s for s, _ in all_curves[0]["w_curve"]])
        w_matrix = np.array([[w for _, w in trial["w_curve"]] for trial in all_curves])
        b_matrix = np.array([[b for _, b in trial["b_curve"]] for trial in all_curves])

        err_matrix = np.sqrt((w_matrix - w_star)**2 + (b_matrix - b_star)**2)

        color = _est_color(est)
        mean_err = np.mean(err_matrix, axis=0)
        std_err = np.std(err_matrix, axis=0)

        ax.plot(steps, mean_err, color=color, linewidth=2, label=_est_label(est))
        ax.fill_between(steps, mean_err - std_err, mean_err + std_err,
                        color=color, alpha=0.15)

    ax.set_xlabel("SGD step")
    ax.set_ylabel("Parameter error $\\| (w,b) - (w^\\star, b^\\star) \\|_2$")
    ax.set_yscale("log")
    title = "Parameter Error vs SGD Step"
    if dist_label:
        title += f" — {dist_label}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    path = os.path.join(output_dir, "param_trajectories.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"    [plot] {path}")


def plot_policy_scatter(per_prompt_df: pd.DataFrame, output_dir: str, dist_label: str = ""):
    """2x2 scatter: learned q vs optimal q* per prompt, colored by stratum."""
    estimators = [e for e in per_prompt_df["estimator"].unique() if _should_include(e)]
    n_est = len(estimators)

    nrows, ncols = 2, 2
    fig, axes_2d = plt.subplots(nrows, ncols, figsize=(10, 10))
    axes_flat = axes_2d.flatten()

    for i in range(n_est, nrows * ncols):
        axes_flat[i].set_visible(False)

    stratum_colors = {
        "very_hard": "#d62728", "hard": "#ff7f0e",
        "medium": "#2ca02c", "easy": "#1f77b4",
    }

    for idx, est in enumerate(estimators):
        ax = axes_flat[idx]
        df_e = per_prompt_df[per_prompt_df["estimator"] == est]

        for stratum in STRATUM_ORDER:
            df_s = df_e[df_e["stratum"] == stratum]
            if df_s.empty:
                continue
            ax.scatter(
                df_s["q_star"], df_s["q_learned"],
                c=stratum_colors[stratum], s=20, alpha=0.5,
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
        ax.set_title(_est_label(est), fontweight="bold", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left", markerscale=2)

    suptitle = "Learned vs Optimal Policy"
    if dist_label:
        suptitle += f" — {dist_label}"
    fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(output_dir, "policy_scatter.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    [plot] {path}")


def compute_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Per-estimator mean +/- std of reward, KL, objective, policy/param error."""
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
    """Per (estimator, stratum): prompt count, mean q-error, mean q-learned, mean q*."""
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


def generate_all_exp3_outputs(
    results: dict, output_dir: str, beta2: float, dist_label: str = "",
):
    """Write all Experiment 3 plots (loss, param error, scatter) and CSV tables."""
    os.makedirs(output_dir, exist_ok=True)

    summary_df = results["summary_df"]
    curves = results["curves"]
    per_prompt_df = results["per_prompt_df"]

    print(f"\n  Generating Experiment 3 plots ...")
    plot_loss_curves(curves, output_dir, dist_label)
    plot_param_trajectories(curves, beta2, output_dir, dist_label)
    plot_policy_scatter(per_prompt_df, output_dir, dist_label)

    print(f"\n  Computing summary tables ...")
    agg_df = compute_summary_table(summary_df)
    path = os.path.join(output_dir, "summary_metrics.csv")
    agg_df.to_csv(path, index=False, float_format="%.6f")
    print(f"    [csv] {path}")

    strat_df = compute_stratified_policy_table(per_prompt_df)
    path = os.path.join(output_dir, "stratified_policy_error.csv")
    strat_df.to_csv(path, index=False, float_format="%.6f")
    print(f"    [csv] {path}")

    path = os.path.join(output_dir, "trial_results.csv")
    summary_df.to_csv(path, index=False, float_format="%.6f")
    print(f"    [csv] {path}")

    plt.close("all")
    print(f"\n  All Experiment 3 outputs saved to {output_dir}/")
