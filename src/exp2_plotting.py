"""
Output utilities for Experiment 2: contextual bandit simulation.

Produces two types of output:

1. **Calibration plots** (PNG): scatter and binned calibration curves.
   These are inherently visual and best communicated as figures.

2. **Summary tables** (CSV): stratified performance (bias, RMSE, win rate)
   and advantage distortion. These are saved as CSVs for easy inclusion
   as LaTeX tables in the thesis write-up.
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
    "legend.fontsize": 8,
    "figure.dpi": 100,
})

# Stratum display order and colors
STRATUM_ORDER = ["very_hard", "hard", "medium", "easy"]
STRATUM_LABELS = {
    "very_hard": "Very Hard\n($p < 0.05$)",
    "hard": "Hard\n($0.05 \\leq p < 0.15$)",
    "medium": "Medium\n($0.15 \\leq p < 0.35$)",
    "easy": "Easy\n($p \\geq 0.35$)",
}
STRATUM_COLORS = {
    "very_hard": "#d62728",
    "hard": "#ff7f0e",
    "medium": "#2ca02c",
    "easy": "#1f77b4",
}

# Method colors for calibration overlay plots
METHOD_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _method_label(method_name: str) -> str:
    """Convert internal method keys to readable labels for plots/tables."""
    if method_name == "lme":
        return "LME"
    if method_name.startswith("single_n"):
        n = method_name.replace("single_n", "")
        return f"Single n={n}"
    if method_name.endswith("_jk"):
        orders = method_name.replace("multi_", "").replace("_jk", "")
        return f"Multi {orders} + JK"
    if method_name.startswith("multi_"):
        orders = method_name.replace("multi_", "")
        return f"Multi {orders}"
    return method_name


# =============================================================================
# 1. Calibration plots (visual — kept as PNG)
# =============================================================================

def plot_calibration_scatter(df, beta, n_samples, output_dir, dist_label=""):
    """
    Multi-panel scatter: one panel per estimator, V_hat_avg(x) vs V*(x).
    Points are colored by difficulty stratum.
    """
    df_cfg = df[(df["beta"] == beta) & (df["n_samples"] == n_samples)]
    if df_cfg.empty:
        return

    methods = sorted(df_cfg["method"].unique(), key=lambda m: (m != "lme", m))
    n_methods = len(methods)
    ncols = min(n_methods, 3)
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)

    for idx, method in enumerate(methods):
        ax = axes[idx // ncols][idx % ncols]
        df_m = df_cfg[df_cfg["method"] == method]

        for stratum in STRATUM_ORDER:
            df_s = df_m[df_m["stratum"] == stratum]
            if df_s.empty:
                continue
            ax.scatter(
                df_s["v_star"], df_s["mean_estimate"],
                c=STRATUM_COLORS[stratum], s=12, alpha=0.5,
                label=STRATUM_LABELS[stratum], edgecolors="none",
            )

        # Diagonal reference line
        vmin = min(df_m["v_star"].min(), df_m["mean_estimate"].min())
        vmax = max(df_m["v_star"].max(), df_m["mean_estimate"].max())
        margin = (vmax - vmin) * 0.05
        lims = [vmin - margin, vmax + margin]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("$V^\\star(x)$")
        ax.set_ylabel("$\\bar{\\hat{V}}(x)$  (trial-averaged)")
        ax.set_title(_method_label(method), fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", markerscale=2)

    for idx in range(n_methods, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    title = f"Calibration Scatter — $\\beta={beta}$, $N={n_samples}$"
    if dist_label:
        title += f" — {dist_label}"
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(output_dir, f"calibration_scatter_beta{beta}_N{n_samples}.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"    [plot] {path}")


def plot_calibration_binned(df, beta, n_samples, output_dir, dist_label="", n_bins=10):
    """
    Binned calibration curves for all estimators overlaid on one plot.

    Prompts are sorted by V*(x), binned into n_bins equal-count groups, and
    (mean_true, mean_predicted) is plotted per bin.  Axes match the scatter
    plot convention: x = V*(x), y = V_hat(x).
    """
    df_cfg = df[(df["beta"] == beta) & (df["n_samples"] == n_samples)]
    if df_cfg.empty:
        return

    methods = sorted(df_cfg["method"].unique(), key=lambda m: (m != "lme", m))

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, method in enumerate(methods):
        df_m = df_cfg[df_cfg["method"] == method].sort_values("v_star")
        if len(df_m) < n_bins:
            continue

        bins = np.array_split(df_m, n_bins)
        true_means = [b["v_star"].mean() for b in bins]
        pred_means = [b["mean_estimate"].mean() for b in bins]

        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        lw = 2.5 if method == "lme" else 1.8
        ls = "-" if method == "lme" else "--"
        marker = "o" if method == "lme" else "s"

        ax.plot(
            true_means, pred_means,
            color=color, linewidth=lw, linestyle=ls,
            marker=marker, markersize=6,
            label=_method_label(method),
        )

    all_vals = df_cfg[["v_star", "mean_estimate"]].values.ravel()
    lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k:", linewidth=1, alpha=0.4, label="Perfect calibration")

    ax.set_xlabel("$V^\\star(x)$ (bin average)")
    ax.set_ylabel("$\\bar{\\hat{V}}(x)$ (bin average)")
    title = f"Binned Calibration — $\\beta={beta}$, $N={n_samples}$"
    if dist_label:
        title += f" — {dist_label}"
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")
    fig.tight_layout()

    path = os.path.join(output_dir, f"calibration_binned_beta{beta}_N{n_samples}.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"    [plot] {path}")


# =============================================================================
# 2. Stratified performance tables (CSV)
# =============================================================================

def compute_stratified_tables(df) -> pd.DataFrame:
    """
    Compute per-(beta, N, method, stratum) summary: mean bias, RMSE, prompt count.

    Returns a single DataFrame with all configurations, ready to save as CSV.
    """
    rows = []
    for (beta, n_samples, method, stratum), grp in df.groupby(
        ["beta", "n_samples", "method", "stratum"]
    ):
        rows.append({
            "beta": beta,
            "n_samples": n_samples,
            "method": method,
            "method_label": _method_label(method),
            "stratum": stratum,
            "n_prompts": len(grp),
            "mean_bias": grp["bias"].mean(),
            "mean_rmse": grp["rmse"].mean(),
            "mean_variance": grp["variance"].mean(),
        })
    return pd.DataFrame(rows)


def compute_win_rate_table(df) -> pd.DataFrame:
    """
    Compute per-(beta, N, method, stratum) win rate over LME.

    Win rate = fraction of prompts where the method has lower per-prompt RMSE
    than LME. LME itself is excluded from the output.
    """
    rows = []
    for (beta, n_samples), grp_cfg in df.groupby(["beta", "n_samples"]):
        lme_rmse = (
            grp_cfg[grp_cfg["method"] == "lme"]
            .set_index("prompt_idx")["rmse"]
        )
        methods = [m for m in grp_cfg["method"].unique() if m != "lme"]
        strata_present = [s for s in STRATUM_ORDER if s in grp_cfg["stratum"].values]

        for method in methods:
            df_m = grp_cfg[grp_cfg["method"] == method].set_index("prompt_idx")
            for stratum in strata_present:
                prompt_idxs = grp_cfg[
                    (grp_cfg["method"] == method) & (grp_cfg["stratum"] == stratum)
                ]["prompt_idx"].values
                if len(prompt_idxs) == 0:
                    continue
                method_rmse = df_m.loc[prompt_idxs, "rmse"].values
                lme_rmse_s = lme_rmse.loc[prompt_idxs].values
                win_frac = float(np.mean(method_rmse < lme_rmse_s))
                rows.append({
                    "beta": beta,
                    "n_samples": n_samples,
                    "method": method,
                    "method_label": _method_label(method),
                    "stratum": stratum,
                    "n_prompts": len(prompt_idxs),
                    "win_rate_vs_lme": win_frac,
                })
    return pd.DataFrame(rows)


# =============================================================================
# 3. Advantage distortion table (CSV)
# =============================================================================

def compute_distortion_table(df) -> pd.DataFrame:
    """
    Compute per-(beta, N, method, stratum) mean advantage shift and log-ratio
    distortion |delta(x)| / beta_2.
    """
    rows = []
    for (beta, n_samples, method, stratum), grp in df.groupby(
        ["beta", "n_samples", "method", "stratum"]
    ):
        rows.append({
            "beta": beta,
            "n_samples": n_samples,
            "method": method,
            "method_label": _method_label(method),
            "stratum": stratum,
            "n_prompts": len(grp),
            "mean_advantage_shift": grp["advantage_shift"].mean(),
            "mean_abs_advantage_shift": grp["advantage_shift"].abs().mean(),
            "mean_log_ratio_distortion": grp["log_ratio_distortion"].mean(),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Main entry point: generate all outputs for a run
# =============================================================================

def generate_all_exp2_outputs(df, output_dir, dist_label=""):
    """
    Generate all Experiment 2 outputs for a results DataFrame.

    - Calibration plots (PNG) for each (beta, N) configuration.
    - Summary CSV tables for stratified metrics, win rates, and distortion.
    """
    os.makedirs(output_dir, exist_ok=True)

    betas = sorted(df["beta"].unique())
    n_values = sorted(df["n_samples"].unique())

    # ---- Calibration plots ----
    total = len(betas) * len(n_values)
    print(f"\n  Generating calibration plots ({total} configurations) ...")
    for beta in betas:
        for n_samples in n_values:
            print(f"\n    beta={beta}, N={n_samples}")
            plot_calibration_scatter(df, beta, n_samples, output_dir, dist_label)
            plot_calibration_binned(df, beta, n_samples, output_dir, dist_label)

    # ---- Summary tables ----
    print("\n  Computing summary tables ...")

    stratified_df = compute_stratified_tables(df)
    path = os.path.join(output_dir, "stratified_metrics.csv")
    stratified_df.to_csv(path, index=False, float_format="%.6f")
    print(f"    [csv] {path}")

    win_rate_df = compute_win_rate_table(df)
    path = os.path.join(output_dir, "win_rate_vs_lme.csv")
    win_rate_df.to_csv(path, index=False, float_format="%.6f")
    print(f"    [csv] {path}")

    distortion_df = compute_distortion_table(df)
    path = os.path.join(output_dir, "distortion.csv")
    distortion_df.to_csv(path, index=False, float_format="%.6f")
    print(f"    [csv] {path}")

    plt.close("all")
    print(f"\n  All outputs saved to {output_dir}/")


# Keep backward-compatible alias
plot_all_exp2 = generate_all_exp2_outputs
