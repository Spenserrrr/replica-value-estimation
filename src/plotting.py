"""
Plotting utilities for V* estimator experiments.

This module generates "triptych" plots: three side-by-side panels sharing
the same y-axis, each dedicated to one estimator family:

    [Single-Replica | Multi-n Slope | Multi-n Slope + Jackknife]

LME (the standard biased baseline) appears in every panel as a black
solid line for reference. This layout keeps each panel readable (≤6 lines)
while allowing direct visual comparison across families.

For each (metric, beta) combination, one triptych figure is produced.
Metrics are: Bias, Variance, RMSE.

All figures respect the 2000-pixel width limit at 100 DPI.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---- Global matplotlib configuration ----
matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 100,  # 18 in × 100 DPI = 1800 px width (within 2000 limit)
})


# =============================================================================
# Style Configuration
# =============================================================================

# LME baseline: solid black, always present in every panel
LME_STYLE = {
    "color": "black",
    "marker": "o",
    "linestyle": "-",
    "linewidth": 2.5,
    "markersize": 7,
}

# Single-replica family: warm oranges/reds, dashed lines
SINGLE_COLORS = ["#e6550d", "#fd8d3c", "#d62728", "#9467bd", "#8c564b"]
SINGLE_MARKERS = ["s", "^", "D", "v", "p"]

# Multi-n (no jackknife) family: cool blues/greens, dash-dot lines
MULTI_COLORS = ["#1f77b4", "#2ca02c", "#17becf", "#31a354", "#756bb1"]
MULTI_MARKERS = ["s", "^", "D", "v", "p"]

# Multi-n (with jackknife) family: lighter cool hues, dotted lines
MULTI_JK_COLORS = ["#6baed6", "#74c476", "#9ecae1", "#a1d99b", "#bcbddc"]
MULTI_JK_MARKERS = ["s", "^", "D", "v", "p"]

# Human-readable metric labels (used in y-axis and titles)
METRIC_LABELS = {
    "bias": "Bias",
    "variance": "Variance",
    "rmse": "RMSE",
}

# Panel titles for each estimator family
FAMILY_SHORT_TITLES = {
    "single_replica": "Single-Replica",
    "multi_n": "Multi-n Slope",
    "multi_n_jk": "Multi-n Slope + Jackknife",
}

# Ordered list of families for the triptych (left to right)
FAMILY_ORDER = ["single_replica", "multi_n", "multi_n_jk"]


# =============================================================================
# Method Classification
# =============================================================================

def _classify_methods(df):
    """
    Classify all methods in the DataFrame into three families, each with
    its own style mapping. LME is included in every family as a baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'method' column.

    Returns
    -------
    dict
        Keys: family name ("single_replica", "multi_n", "multi_n_jk")
        Values: list of (method_key, label, plot_kwargs) tuples.
    """
    all_methods = sorted(df["method"].unique())

    # Collect method keys by family
    single_keys = [m for m in all_methods if m.startswith("single_n")]
    multi_keys = [m for m in all_methods if m.startswith("multi_") and not m.endswith("_jk")]
    multi_jk_keys = [m for m in all_methods if m.endswith("_jk")]

    # Sort single-replica by n value for consistent ordering
    single_keys.sort(key=lambda m: int(m.replace("single_n", "")))

    families = {}

    # ---- Family 1: Single-replica estimators ----
    items = [("lme", "LME (baseline)", LME_STYLE)]
    for i, key in enumerate(single_keys):
        n = key.replace("single_n", "")
        items.append((key, f"Single-replica (n={n})", {
            "color": SINGLE_COLORS[i % len(SINGLE_COLORS)],
            "marker": SINGLE_MARKERS[i % len(SINGLE_MARKERS)],
            "linestyle": "--",
            "linewidth": 2,
            "markersize": 7,
        }))
    families["single_replica"] = items

    # ---- Family 2: Multi-n slope (no jackknife) ----
    items = [("lme", "LME (baseline)", LME_STYLE)]
    for i, key in enumerate(multi_keys):
        orders_str = key.replace("multi_", "")
        items.append((key, f"Multi-n {orders_str}", {
            "color": MULTI_COLORS[i % len(MULTI_COLORS)],
            "marker": MULTI_MARKERS[i % len(MULTI_MARKERS)],
            "linestyle": "-.",
            "linewidth": 2,
            "markersize": 7,
        }))
    families["multi_n"] = items

    # ---- Family 3: Multi-n slope (with jackknife) ----
    items = [("lme", "LME (baseline)", LME_STYLE)]
    for i, key in enumerate(multi_jk_keys):
        orders_str = key.replace("multi_", "").replace("_jk", "")
        items.append((key, f"Multi-n {orders_str} + JK", {
            "color": MULTI_JK_COLORS[i % len(MULTI_JK_COLORS)],
            "marker": MULTI_JK_MARKERS[i % len(MULTI_JK_MARKERS)],
            "linestyle": ":",
            "linewidth": 2.2,
            "markersize": 7,
        }))
    families["multi_n_jk"] = items

    return families


# =============================================================================
# Single-Panel Plotting
# =============================================================================

def _plot_one(ax, df_beta, metric, family_items, show_ylabel=True):
    """
    Plot one (metric, beta, family) panel onto the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot onto.
    df_beta : pd.DataFrame
        DataFrame filtered to a single beta value.
    metric : str
        One of "bias", "variance", "rmse".
    family_items : list of (method_key, label, style_dict)
        Methods to plot, including LME baseline.
    show_ylabel : bool
        Whether to show the y-axis label (only for leftmost panel).
    """
    for method_key, label, style in family_items:
        df_m = df_beta[df_beta["method"] == method_key]
        if df_m.empty:
            continue

        x = df_m["n_tot"].values
        y = df_m[metric].values
        mask = df_m["n_valid"].values > 0
        if not np.any(mask):
            continue

        ax.plot(x[mask], y[mask], label=label, **style)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("$N_{\\mathrm{tot}}$", fontsize=10)

    if show_ylabel:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)

    # Add reference line at y=0 for bias plots
    if metric == "bias":
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    # Use log scale for variance and RMSE (always positive)
    elif metric in ("variance", "rmse"):
        ax.set_yscale("log")

    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


# =============================================================================
# Main Plotting Function
# =============================================================================

def plot_results(df, output_dir, dist_label=""):
    """
    Generate side-by-side triptych plots: one figure per (metric, beta).

    Each figure has 3 panels sharing the same y-axis:
        [Single-Replica | Multi-n Slope | Multi-n Slope + JK]

    LME (baseline) appears in every panel as a black solid line for reference.

    Total output: 3 metrics × len(betas) figures.

    Parameters
    ----------
    df : pd.DataFrame
        Results from run_experiment(). Must contain columns:
        [beta, n_tot, method, bias, variance, rmse, v_star, n_valid]
    output_dir : str
        Directory to save figures.
    dist_label : str, optional
        Label for the distribution, added to the plot title.
        E.g., "Gaussian(μ=0, σ=1)" or "Bernoulli(p=0.1)".
    """
    os.makedirs(output_dir, exist_ok=True)

    betas = sorted(df["beta"].unique())
    metrics = ["bias", "variance", "rmse"]
    families = _classify_methods(df)

    n_panels = len(FAMILY_ORDER)
    total_plots = len(metrics) * len(betas)
    print(f"\nGenerating {total_plots} triptych plots (3 panels each) ...")

    for metric in metrics:
        for beta in betas:
            df_beta = df[df["beta"] == beta]
            v_star = df_beta["v_star"].iloc[0]

            # Create figure with shared y-axis.
            # Width: 18 inches × 100 DPI = 1800 px (within 2000 limit)
            fig, axes = plt.subplots(
                1, n_panels,
                figsize=(18, 5),
                sharey=True,
            )

            for col_idx, family_name in enumerate(FAMILY_ORDER):
                ax = axes[col_idx]
                family_items = families[family_name]
                short_title = FAMILY_SHORT_TITLES[family_name]

                # Only show y-axis label on leftmost panel
                _plot_one(
                    ax, df_beta, metric, family_items,
                    show_ylabel=(col_idx == 0),
                )
                ax.set_title(short_title, fontsize=11, fontweight="bold")
                ax.tick_params(axis="both", labelsize=9)

            # Overall title: include metric, beta, V*, and distribution label
            title_parts = [
                f"{METRIC_LABELS[metric]}",
                f"$\\beta = {beta}$",
                f"($V^* = {v_star:.3f}$)",
            ]
            if dist_label:
                title_parts.append(f"— {dist_label}")
            fig.suptitle(
                "  —  ".join(title_parts[:3]) + ("  " + title_parts[3] if len(title_parts) > 3 else ""),
                fontsize=13, y=1.02,
            )

            fig.tight_layout()

            filename = f"{metric}_beta{beta}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, bbox_inches="tight", dpi=100)
            plt.close(fig)
            print(f"  {filepath}")

    plt.close("all")
    print(f"\nAll {total_plots} plots saved to {output_dir}/")

