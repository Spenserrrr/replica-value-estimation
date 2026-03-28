"""
Experiment 1 plots: 2x2 panels per (metric, beta), one panel per estimator family.
LME baseline in every panel. Metrics: bias, variance, RMSE.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

LME_STYLE = {
    "color": "black",
    "marker": "o",
    "linestyle": "-",
    "linewidth": 2.5,
    "markersize": 7,
}

SINGLE_COLORS = ["#e6550d", "#fd8d3c", "#d62728", "#9467bd", "#8c564b"]
SINGLE_MARKERS = ["s", "^", "D", "v", "p"]

MULTI_COLORS = ["#1f77b4", "#2ca02c", "#17becf", "#31a354", "#756bb1"]
MULTI_MARKERS = ["s", "^", "D", "v", "p"]

MULTI_JK_COLORS = ["#6baed6", "#74c476", "#9ecae1", "#a1d99b", "#bcbddc"]
MULTI_JK_MARKERS = ["s", "^", "D", "v", "p"]

BETA_SMOOTH_COLORS = ["#7b3294", "#c2a5cf", "#a6611a"]
BETA_SMOOTH_MARKERS = ["*", "P", "X"]

METRIC_LABELS = {
    "bias": "Bias",
    "variance": "Variance",
    "rmse": "RMSE",
}

FAMILY_SHORT_TITLES = {
    "single_replica": "Single-Replica",
    "multi_n": "Multi-n Slope",
    "multi_n_jk": "Multi-n Slope + Jackknife",
    "beta_smooth": "Beta-Smoothed",
}

FAMILY_ORDER_BASE = ["single_replica", "multi_n", "multi_n_jk"]


def _classify_methods(df):
    """Group methods into styled families for panel assignment."""
    all_methods = sorted(df["method"].unique())

    single_keys = sorted(
        [m for m in all_methods if m.startswith("single_n")],
        key=lambda m: int(m.replace("single_n", "")),
    )
    multi_keys = [m for m in all_methods if m.startswith("multi_") and not m.endswith("_jk")]
    multi_jk_keys = [m for m in all_methods if m.endswith("_jk")]
    beta_smooth_keys = [m for m in all_methods if m.startswith("beta_smooth_")]

    families = {}

    items = [("lme", "LME (baseline)", LME_STYLE)]
    for i, key in enumerate(single_keys):
        n = key.replace("single_n", "")
        items.append((key, f"Single-replica (n={n})", {
            "color": SINGLE_COLORS[i % len(SINGLE_COLORS)],
            "marker": SINGLE_MARKERS[i % len(SINGLE_MARKERS)],
            "linestyle": "--", "linewidth": 2, "markersize": 7,
        }))
    families["single_replica"] = items

    items = [("lme", "LME (baseline)", LME_STYLE)]
    for i, key in enumerate(multi_keys):
        orders_str = key.replace("multi_", "")
        items.append((key, f"Multi-n {orders_str}", {
            "color": MULTI_COLORS[i % len(MULTI_COLORS)],
            "marker": MULTI_MARKERS[i % len(MULTI_MARKERS)],
            "linestyle": "-.", "linewidth": 2, "markersize": 7,
        }))
    families["multi_n"] = items

    items = [("lme", "LME (baseline)", LME_STYLE)]
    for i, key in enumerate(multi_jk_keys):
        orders_str = key.replace("multi_", "").replace("_jk", "")
        items.append((key, f"Multi-n {orders_str} + JK", {
            "color": MULTI_JK_COLORS[i % len(MULTI_JK_COLORS)],
            "marker": MULTI_JK_MARKERS[i % len(MULTI_JK_MARKERS)],
            "linestyle": ":", "linewidth": 2.2, "markersize": 7,
        }))
    families["multi_n_jk"] = items

    if beta_smooth_keys:
        items = [("lme", "LME (baseline)", LME_STYLE)]
        for i, key in enumerate(beta_smooth_keys):
            prior_name = key.replace("beta_smooth_", "").capitalize()
            items.append((key, f"Beta-smooth ({prior_name})", {
                "color": BETA_SMOOTH_COLORS[i % len(BETA_SMOOTH_COLORS)],
                "marker": BETA_SMOOTH_MARKERS[i % len(BETA_SMOOTH_MARKERS)],
                "linestyle": "-", "linewidth": 2, "markersize": 8,
            }))
        families["beta_smooth"] = items

    return families


def _plot_one(ax, df_beta, metric, family_items, show_ylabel=True):
    """Render one (metric, beta, family) panel."""
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
    ax.set_xlabel("$N_{\\mathrm{tot}}$", fontsize=11)

    if show_ylabel:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)

    if metric == "bias":
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    elif metric in ("variance", "rmse"):
        ax.set_yscale("log")

    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)


def plot_results(df, output_dir, dist_label=""):
    """One 2x2 figure per (metric, beta); 3 or 4 family panels with shared y-axis."""
    os.makedirs(output_dir, exist_ok=True)

    betas = sorted(df["beta"].unique())
    metrics = ["bias", "variance", "rmse"]
    families = _classify_methods(df)

    family_order = list(FAMILY_ORDER_BASE)
    if "beta_smooth" in families:
        family_order.append("beta_smooth")

    n_panels = len(family_order)
    total_plots = len(metrics) * len(betas)
    has_beta_smooth = "beta_smooth" in families
    n_base = len(FAMILY_ORDER_BASE)

    print(f"\nGenerating {total_plots} plots ({n_panels} panels each) ...")

    for metric in metrics:
        for beta in betas:
            df_beta = df[df["beta"] == beta]
            v_star = df_beta["v_star"].iloc[0]

            import matplotlib.gridspec as gridspec
            nrows, ncols = 2, 2
            fig = plt.figure(figsize=(12, 10))
            gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.3)

            axes = []
            if has_beta_smooth and n_panels == 4:
                ax0 = fig.add_subplot(gs[0, 0])
                axes.append(ax0)
                axes.append(fig.add_subplot(gs[0, 1], sharey=ax0))
                axes.append(fig.add_subplot(gs[1, 0], sharey=ax0))
                # Beta-smoothed gets its own y-axis (different scale)
                axes.append(fig.add_subplot(gs[1, 1]))
            else:
                ax0 = fig.add_subplot(gs[0, 0])
                axes.append(ax0)
                for i in range(1, n_panels):
                    r, c = divmod(i, ncols)
                    axes.append(fig.add_subplot(gs[r, c], sharey=ax0))
                if n_panels < nrows * ncols:
                    for i in range(n_panels, nrows * ncols):
                        r, c = divmod(i, ncols)
                        ax_empty = fig.add_subplot(gs[r, c])
                        ax_empty.set_visible(False)

            for panel_idx, family_name in enumerate(family_order):
                ax = axes[panel_idx]
                family_items = families[family_name]
                short_title = FAMILY_SHORT_TITLES[family_name]

                show_ylabel = (panel_idx % ncols == 0) or (has_beta_smooth and panel_idx == n_base)
                _plot_one(ax, df_beta, metric, family_items, show_ylabel=show_ylabel)
                ax.set_title(short_title, fontsize=12, fontweight="bold")
                ax.tick_params(axis="both", labelsize=10)

            title_parts = [
                f"{METRIC_LABELS[metric]}",
                f"$\\beta = {beta}$",
                f"($V^* = {v_star:.3f}$)",
            ]
            if dist_label:
                title_parts.append(f"— {dist_label}")
            fig.suptitle(
                "  —  ".join(title_parts[:3]) + ("  " + title_parts[3] if len(title_parts) > 3 else ""),
                fontsize=14, y=0.98,
            )

            filename = f"{metric}_beta{beta}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  {filepath}")

    plt.close("all")
    print(f"\nAll {total_plots} plots saved to {output_dir}/")
