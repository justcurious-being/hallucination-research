"""
correlation.py
==============
Reproduces Table 2:

  Correlation analysis between each data quality dimension and the
  Hallucination Index (HI), including Pearson r, p-value, and
  OLS regression to identify strongest predictors.

Input:
    results/quality_scores.csv     (from quality_metrics.py)
    results/hallucination_index.csv (from hallucination_index.py)

Output:
    results/correlation_table.csv
    results/regression_summary.csv
    results/figures/               (all plots)
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

# Quality dimensions to correlate against HI
QUALITY_DIMS = [
    "verified_fact_ratio",
    "contradiction_density",
    "domain_balance_score",
    "noise_ratio",
    "shannon_entropy",
]

# Friendly display names matching Table 2 in the paper
DIM_LABELS = {
    "verified_fact_ratio":   "Data Accuracy",
    "contradiction_density": "Contradiction Density",
    "domain_balance_score":  "Representational Balance",
    "noise_ratio":           "Noise Ratio",
    "shannon_entropy":       "Data Entropy",
}

# Expected direction for sanity-check annotations
DIM_DIRECTION = {
    "verified_fact_ratio":   "negative",
    "contradiction_density": "positive",
    "domain_balance_score":  "negative",
    "noise_ratio":           "positive",
    "shannon_entropy":       "positive",
}


# ── Correlation table ─────────────────────────────────────────────────────────

def compute_correlation_table(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson r and p-value for each quality dimension vs. HI.
    Also computes Spearman rho for robustness check.
    """
    rows = []
    for dim in QUALITY_DIMS:
        if dim not in merged.columns:
            continue

        x = merged[dim].values
        y = merged["hallucination_index"].values

        # Drop any NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) < 3:
            logger.warning(f"  Skipping {dim}: insufficient data ({len(x)} points)")
            continue

        pearson_r,  pearson_p  = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        rows.append({
            "data_quality_dimension":  DIM_LABELS.get(dim, dim),
            "pearson_r":               round(pearson_r,  3),
            "pearson_p":               round(pearson_p,  4),
            "spearman_r":              round(spearman_r, 3),
            "spearman_p":              round(spearman_p, 4),
            "significant_p01":         pearson_p < 0.01,
            "expected_direction":      DIM_DIRECTION.get(dim, "?"),
            "direction_confirmed":     (
                (DIM_DIRECTION[dim] == "positive" and pearson_r > 0) or
                (DIM_DIRECTION[dim] == "negative" and pearson_r < 0)
            ) if dim in DIM_DIRECTION else None,
        })

    return pd.DataFrame(rows)


# ── OLS regression ────────────────────────────────────────────────────────────

def run_regression(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Simple OLS regression: each quality dimension as sole predictor of HI.
    Returns slope, intercept, R², and standardized beta for each predictor.
    """
    rows = []
    for dim in QUALITY_DIMS:
        if dim not in merged.columns:
            continue
        x = merged[[dim]].dropna()
        y = merged.loc[x.index, "hallucination_index"]

        slope, intercept, r, p, se = stats.linregress(x[dim], y)
        rows.append({
            "predictor":    DIM_LABELS.get(dim, dim),
            "slope":        round(slope,     4),
            "intercept":    round(intercept, 4),
            "r_squared":    round(r ** 2,    4),
            "p_value":      round(p,         4),
            "std_error":    round(se,        4),
        })

    df = pd.DataFrame(rows).sort_values("r_squared", ascending=False)
    return df


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_hi_by_corpus(hi_df: pd.DataFrame):
    """Bar chart of HI per corpus — reproduces Figure 1 from the paper."""
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ["#00A896", "#4BA3BD", "#E07B5D", "#E8A87C"]
    order  = ["baseline", "noise_augmented", "contradiction_enriched", "imbalanced"]
    labels = ["High-Quality\nCurated", "Noise-\nAugmented", "Contradiction-\nEnriched", "Distributionally\nImbalanced"]

    # Reorder
    plot_df = hi_df.set_index("corpus").reindex(order).reset_index()

    bars = ax.bar(labels, plot_df["hallucination_index"], color=colors, width=0.55, edgecolor="white", linewidth=1.2)

    # Value labels
    for bar, val in zip(bars, plot_df["hallucination_index"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # RAG comparison if available
    if "hi_with_rag" in plot_df.columns:
        ax.bar(labels, plot_df["hi_with_rag"], color="none",
               edgecolor="navy", linestyle="--", linewidth=1.5, width=0.55,
               label="HI with RAG")
        ax.legend(fontsize=10)

    ax.set_ylabel("Hallucination Index (%)", fontsize=12)
    ax.set_title("Hallucination Index by Training Data Condition\n(Reproducing Paper Figure 1)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 30)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0, symbol="%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = FIGURES_DIR / "fig1_hi_by_corpus.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"  Saved: {out}")


def plot_correlation_heatmap(merged: pd.DataFrame):
    """Correlation heatmap of quality dimensions vs. HI."""
    cols   = QUALITY_DIMS + ["hallucination_index"]
    subset = merged[[c for c in cols if c in merged.columns]].rename(
        columns={**DIM_LABELS, "hallucination_index": "Hallucination\nIndex"}
    )

    corr = subset.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # show lower triangle

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, square=True, ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Correlation Matrix — Data Quality Dimensions vs. Hallucination Index",
                 fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    out = FIGURES_DIR / "fig2_correlation_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"  Saved: {out}")


def plot_scatter_grid(merged: pd.DataFrame):
    """Scatter plots: each quality dimension vs. HI with regression line."""
    dims_present = [d for d in QUALITY_DIMS if d in merged.columns]
    n = len(dims_present)
    cols_plot = min(n, 3)
    rows_plot = (n + cols_plot - 1) // cols_plot

    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(5 * cols_plot, 4 * rows_plot))
    axes = np.array(axes).flatten()

    for i, dim in enumerate(dims_present):
        ax = axes[i]
        x  = merged[dim].values
        y  = merged["hallucination_index"].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        ax.scatter(x, y, color="#1C7293", s=80, alpha=0.8, zorder=3)

        # Regression line
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 100)
            ax.plot(xline, m * xline + b, color="#E07B5D", linewidth=2, zorder=4)

            r, p = stats.pearsonr(x, y)
            ax.set_title(f"{DIM_LABELS.get(dim, dim)}\nr={r:+.2f}, p={p:.3f}",
                         fontsize=10, fontweight="bold")

        ax.set_xlabel(DIM_LABELS.get(dim, dim), fontsize=9)
        ax.set_ylabel("HI (%)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle="--", alpha=0.3)

    # Hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Data Quality Dimensions vs. Hallucination Index — Scatter + Regression",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "fig3_scatter_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {out}")


def plot_regression_r2(reg_df: pd.DataFrame):
    """Horizontal bar chart of R² by predictor — shows predictive strength."""
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = ["#E07B5D" if r > 0.5 else "#4BA3BD" for r in reg_df["r_squared"]]
    ax.barh(reg_df["predictor"], reg_df["r_squared"], color=colors, edgecolor="white")

    for i, (val, pred) in enumerate(zip(reg_df["r_squared"], reg_df["predictor"])):
        ax.text(val + 0.01, i, f"R²={val:.3f}", va="center", fontsize=10)

    ax.set_xlabel("R² (Proportion of HI variance explained)", fontsize=11)
    ax.set_title("Predictive Strength of Data Quality Dimensions on HI", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "fig4_r2_by_predictor.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    qs_path = RESULTS_DIR / "quality_scores.csv"
    hi_path = RESULTS_DIR / "hallucination_index.csv"

    if not qs_path.exists() or not hi_path.exists():
        logger.error("Missing results files. Run quality_metrics.py and hallucination_index.py first.")
        return

    quality = pd.read_csv(qs_path)
    hi      = pd.read_csv(hi_path)
    merged  = quality.merge(hi[["corpus", "hallucination_index"]], on="corpus", how="inner")

    logger.info(f"Merged {len(merged)} corpus rows for analysis.")

    # Correlation table
    corr_table = compute_correlation_table(merged)
    corr_out   = RESULTS_DIR / "correlation_table.csv"
    corr_table.to_csv(corr_out, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("TABLE 2 — CORRELATION: DATA QUALITY vs. HALLUCINATION INDEX")
    logger.info("=" * 70)
    logger.info(corr_table.to_string(index=False))
    logger.info(f"\nSaved to: {corr_out}")

    # Regression
    reg_df  = run_regression(merged)
    reg_out = RESULTS_DIR / "regression_summary.csv"
    reg_df.to_csv(reg_out, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("OLS REGRESSION SUMMARY (sorted by R²)")
    logger.info("=" * 70)
    logger.info(reg_df.to_string(index=False))
    logger.info(f"\nSaved to: {reg_out}")

    # Plots
    logger.info("\nGenerating figures...")
    plot_hi_by_corpus(hi)
    plot_correlation_heatmap(merged)
    plot_scatter_grid(merged)
    plot_regression_r2(reg_df)

    logger.info(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run()
