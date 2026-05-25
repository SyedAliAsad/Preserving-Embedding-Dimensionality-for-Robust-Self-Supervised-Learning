"""Generate publication-quality figures for the thesis.

Produces all main figures from the experimental results:
  1. Trade-off curve: effective rank vs lambda (with accuracy)
  2. Robustness curve: relative robustness vs lambda
  3. Dual-axis plot: accuracy + rank vs lambda
  4. Per-corruption comparison bar chart
  5. Per-category robustness comparison
  6. Eigenvalue spectrum comparison (if checkpoints available)
  7. Effective rank evolution over training (from W&B data if provided)
  8. Method comparison summary table as figure

Usage:
    python scripts/make_plots.py --results_dir ./results --output_dir ./figures
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# ============================================================
# STYLE
# ============================================================

def set_style():
    """Apply clean publication style."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#cccccc",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#666666",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })

# Color palette - colorblind friendly
COLORS = {
    "baseline":   "#333333",
    "lambda_0.1": "#4C72B0",
    "lambda_0.25": "#55A868",
    "lambda_0.5": "#C44E52",
    "lambda_1.0": "#8172B3",
    "vicreg_base": "#937860",
    "vicreg_dim": "#DA8BC3",
    "accent": "#DD8452",
}


# ============================================================
# DATA
# ============================================================

# Hard-coded results from the experiments (from your training + robustness evals)
RESULTS = {
    "simclr_baseline": {
        "lambda": 0.0,
        "method": "SimCLR",
        "clean_acc": 0.4849,
        "corruption_acc": 0.3901,
        "relative_robustness": 0.8046,
        "effective_rank": 121.6,
        "vn_entropy": 4.80,
        "category_means": {
            "noise": 0.3280, "blur": 0.3972,
            "weather": 0.3918, "digital": 0.4281,
        },
    },
    "simclr_dimreg_0.1": {
        "lambda": 0.1,
        "method": "SimCLR + L_dim",
        "clean_acc": 0.4821,
        "corruption_acc": 0.3868,
        "relative_robustness": 0.8023,
        "effective_rank": 210.0,
        "vn_entropy": 5.35,
        "category_means": {
            "noise": 0.3243, "blur": 0.3950,
            "weather": 0.3874, "digital": 0.4249,
        },
    },
    "simclr_dimreg_0.25": {
        "lambda": 0.25,
        "method": "SimCLR + L_dim",
        "clean_acc": 0.4549,
        "corruption_acc": 0.3722,
        "relative_robustness": 0.8183,
        "effective_rank": 302.1,
        "vn_entropy": 5.71,
        "category_means": {
            "noise": 0.3129, "blur": 0.3798,
            "weather": 0.3725, "digital": 0.4089,
        },
    },
    "simclr_dimreg_0.5": {
        "lambda": 0.5,
        "method": "SimCLR + L_dim",
        "clean_acc": 0.4205,
        "corruption_acc": 0.3467,
        "relative_robustness": 0.8245,
        "effective_rank": 363.5,
        "vn_entropy": 5.90,
        "category_means": {
            "noise": 0.2943, "blur": 0.3549,
            "weather": 0.3475, "digital": 0.3769,
        },
    },
    "simclr_dimreg_1.0": {
        "lambda": 1.0,
        "method": "SimCLR + L_dim",
        "clean_acc": 0.3409,
        "corruption_acc": 0.2764,
        "relative_robustness": 0.8108,
        "effective_rank": 411.8,
        "vn_entropy": 6.02,
        "category_means": {
            "noise": 0.2269, "blur": 0.2836,
            "weather": 0.2773, "digital": 0.3054,
        },
    },
    "vicreg_baseline": {
        "lambda": 0.0,
        "method": "VICReg",
        "clean_acc": 0.4720,
        "corruption_acc": 0.3870,
        "relative_robustness": 0.8198,
        "effective_rank": 159.1,
        "vn_entropy": 5.07,
        "category_means": {
            "noise": 0.3362, "blur": 0.3915,
            "weather": 0.3862, "digital": 0.4213,
        },
    },
    "vicreg_dimreg": {
        "lambda": 0.05,
        "method": "VICReg + L_dim",
        "clean_acc": 0.4768,
        "corruption_acc": 0.3882,
        "relative_robustness": 0.8143,
        "effective_rank": 166.6,
        "vn_entropy": 5.12,
        "category_means": {
            "noise": 0.3331, "blur": 0.3938,
            "weather": 0.3879, "digital": 0.4244,
        },
    },
}

# Training trajectory data (from training logs)
TRAJECTORIES = {
    "baseline": {
        "epochs": [49, 99, 199, 399],
        "test_acc": [0.378, 0.431, 0.466, 0.488],
        "effective_rank": [56.8, 72.8, 101.7, 121.6],
        "vn_entropy": [4.04, 4.29, 4.62, 4.80],
    },
    "lambda_0.25": {
        "epochs": [49, 99, 199, 399],
        "test_acc": [0.376, 0.421, 0.450, 0.456],
        "effective_rank": [121.5, 188.6, 260.3, 302.1],
        "vn_entropy": [4.80, 5.24, 5.56, 5.71],
    },
    "lambda_0.5": {
        "epochs": [49, 99, 199, 399],
        "test_acc": [0.374, 0.411, 0.411, 0.418],
        "effective_rank": [182.4, 264.0, 322.6, 363.5],
        "vn_entropy": [5.21, 5.54, 5.78, 5.90],
    },
}


# ============================================================
# PLOTS
# ============================================================

def plot_tradeoff_curve(outdir: Path):
    """Figure 1: Trade-off curve — accuracy + effective rank vs lambda."""
    simclr_lambdas = [0.0, 0.1, 0.25, 0.5, 1.0]
    simclr_acc = [RESULTS[f"simclr_baseline" if l == 0 else f"simclr_dimreg_{l}"]["clean_acc"] * 100
                  for l in simclr_lambdas]
    simclr_rank = [RESULTS[f"simclr_baseline" if l == 0 else f"simclr_dimreg_{l}"]["effective_rank"]
                   for l in simclr_lambdas]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # Accuracy on left axis
    color1 = COLORS["lambda_0.5"]
    ax1.set_xlabel(r"Regularization weight $\lambda$")
    ax1.set_ylabel("Clean test accuracy (%)", color=color1)
    line1 = ax1.plot(simclr_lambdas, simclr_acc, marker="o", linewidth=2.2,
                     markersize=9, color=color1, label="Clean accuracy",
                     markerfacecolor=color1, markeredgecolor="white",
                     markeredgewidth=1.5, zorder=3)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(30, 52)
    ax1.grid(True, alpha=0.3)

    # Effective rank on right axis
    ax2 = ax1.twinx()
    color2 = COLORS["lambda_0.1"]
    ax2.set_ylabel("Effective rank", color=color2)
    line2 = ax2.plot(simclr_lambdas, simclr_rank, marker="s", linewidth=2.2,
                     markersize=9, color=color2, label="Effective rank",
                     markerfacecolor=color2, markeredgecolor="white",
                     markeredgewidth=1.5, zorder=3)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(80, 450)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.95)

    plt.title(r"Trade-off: Accuracy vs Effective Rank under $L_{dim}$ (SimCLR)",
              pad=12)
    plt.tight_layout()
    plt.savefig(outdir / "fig1_tradeoff_curve.pdf")
    plt.savefig(outdir / "fig1_tradeoff_curve.png")
    plt.close()
    print(f"  Saved: fig1_tradeoff_curve.{{pdf,png}}")


def plot_robustness_curve(outdir: Path):
    """Figure 2: Relative robustness vs lambda."""
    simclr_lambdas = [0.0, 0.1, 0.25, 0.5, 1.0]
    rel_rob = [RESULTS["simclr_baseline"]["relative_robustness"] * 100,
               RESULTS["simclr_dimreg_0.1"]["relative_robustness"] * 100,
               RESULTS["simclr_dimreg_0.25"]["relative_robustness"] * 100,
               RESULTS["simclr_dimreg_0.5"]["relative_robustness"] * 100,
               RESULTS["simclr_dimreg_1.0"]["relative_robustness"] * 100]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Main line
    ax.plot(simclr_lambdas, rel_rob, marker="o", linewidth=2.5,
            markersize=11, color=COLORS["lambda_0.5"],
            markerfacecolor=COLORS["lambda_0.5"], markeredgecolor="white",
            markeredgewidth=1.8, zorder=3, label="SimCLR + L$_{dim}$")

    # Baseline reference line
    baseline_rr = RESULTS["simclr_baseline"]["relative_robustness"] * 100
    ax.axhline(baseline_rr, color="#555555", linestyle="--",
               linewidth=1.2, alpha=0.7, zorder=1, label="SimCLR baseline")

    # Highlight the peak
    peak_idx = int(np.argmax(rel_rob))
    ax.annotate(f"Peak: {rel_rob[peak_idx]:.2f}%",
                xy=(simclr_lambdas[peak_idx], rel_rob[peak_idx]),
                xytext=(simclr_lambdas[peak_idx] + 0.15, rel_rob[peak_idx] + 0.3),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))

    ax.set_xlabel(r"Regularization weight $\lambda$")
    ax.set_ylabel("Relative robustness (%)")
    ax.set_title("Corruption robustness vs dimensionality regularization strength",
                 pad=12)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="lower right")
    ax.set_ylim(79.5, 83.0)

    plt.tight_layout()
    plt.savefig(outdir / "fig2_robustness_curve.pdf")
    plt.savefig(outdir / "fig2_robustness_curve.png")
    plt.close()
    print(f"  Saved: fig2_robustness_curve.{{pdf,png}}")


def plot_training_trajectories(outdir: Path):
    """Figure 3: Effective rank evolution during training."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    trajectories_to_plot = [
        ("baseline", "SimCLR baseline", COLORS["baseline"], "o"),
        ("lambda_0.25", r"$\lambda=0.25$", COLORS["lambda_0.25"], "s"),
        ("lambda_0.5", r"$\lambda=0.5$", COLORS["lambda_0.5"], "^"),
    ]

    # --- Left: effective rank ---
    ax = axes[0]
    for key, label, color, marker in trajectories_to_plot:
        traj = TRAJECTORIES[key]
        ax.plot(traj["epochs"], traj["effective_rank"],
                marker=marker, linewidth=2.2, markersize=9,
                color=color, label=label, markeredgecolor="white",
                markeredgewidth=1.4, zorder=3)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Effective rank")
    ax.set_title("Effective rank throughout training", pad=10)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left")
    ax.axhline(512, color="#888888", linestyle=":", linewidth=1,
               alpha=0.6, zorder=1)
    ax.text(380, 505, "max = 512", fontsize=9, color="#666666",
            ha="right", va="top")

    # --- Right: test accuracy ---
    ax = axes[1]
    for key, label, color, marker in trajectories_to_plot:
        traj = TRAJECTORIES[key]
        acc_pct = [a * 100 for a in traj["test_acc"]]
        ax.plot(traj["epochs"], acc_pct,
                marker=marker, linewidth=2.2, markersize=9,
                color=color, label=label, markeredgecolor="white",
                markeredgewidth=1.4, zorder=3)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Linear probe test accuracy (%)")
    ax.set_title("Downstream accuracy throughout training", pad=10)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(outdir / "fig3_training_trajectories.pdf")
    plt.savefig(outdir / "fig3_training_trajectories.png")
    plt.close()
    print(f"  Saved: fig3_training_trajectories.{{pdf,png}}")


def plot_category_robustness(outdir: Path):
    """Figure 4: Per-category robustness comparison."""
    categories = ["noise", "blur", "weather", "digital"]
    cat_labels = ["Noise", "Blur", "Weather", "Digital"]

    configs = [
        ("simclr_baseline", "SimCLR", COLORS["baseline"]),
        ("simclr_dimreg_0.25", r"+ L$_{dim}$ ($\lambda$=0.25)", COLORS["lambda_0.25"]),
        ("simclr_dimreg_0.5", r"+ L$_{dim}$ ($\lambda$=0.5)", COLORS["lambda_0.5"]),
    ]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for i, (key, label, color) in enumerate(configs):
        values = [RESULTS[key]["category_means"][c] * 100 for c in categories]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color,
                      edgecolor="white", linewidth=1.2)
        # Value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3,
                    f"{height:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xlabel("Corruption category")
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_title("Robustness across corruption categories (CIFAR-100-C)",
                 pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.legend(loc="upper left", ncol=1)
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_ylim(0, 48)

    plt.tight_layout()
    plt.savefig(outdir / "fig4_category_robustness.pdf")
    plt.savefig(outdir / "fig4_category_robustness.png")
    plt.close()
    print(f"  Saved: fig4_category_robustness.{{pdf,png}}")


def plot_method_comparison(outdir: Path):
    """Figure 5: SimCLR vs VICReg — cross-method comparison."""
    methods = [
        ("SimCLR\nbaseline", RESULTS["simclr_baseline"], COLORS["baseline"]),
        ("SimCLR\n+ L$_{dim}$", RESULTS["simclr_dimreg_0.5"], COLORS["lambda_0.5"]),
        ("VICReg\nbaseline", RESULTS["vicreg_baseline"], COLORS["vicreg_base"]),
        ("VICReg\n+ L$_{dim}$", RESULTS["vicreg_dimreg"], COLORS["vicreg_dim"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # --- Clean accuracy ---
    ax = axes[0]
    labels = [m[0] for m in methods]
    values = [m[1]["clean_acc"] * 100 for m in methods]
    colors = [m[2] for m in methods]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Clean accuracy (%)")
    ax.set_title("Clean test accuracy", pad=10)
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_ylim(0, max(values) * 1.12)

    # --- Relative robustness ---
    ax = axes[1]
    values = [m[1]["relative_robustness"] * 100 for m in methods]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Relative robustness (%)")
    ax.set_title("Corruption robustness", pad=10)
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_ylim(78, max(values) * 1.02)

    # --- Effective rank ---
    ax = axes[2]
    values = [m[1]["effective_rank"] for m in methods]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Effective rank")
    ax.set_title("Representation dimensionality", pad=10)
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_ylim(0, max(values) * 1.15)

    plt.suptitle("Cross-method comparison: SimCLR vs VICReg",
                 fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "fig5_method_comparison.pdf")
    plt.savefig(outdir / "fig5_method_comparison.png")
    plt.close()
    print(f"  Saved: fig5_method_comparison.{{pdf,png}}")


def plot_rank_vs_entropy(outdir: Path):
    """Figure 6: Scatter of effective rank vs VN entropy, colored by accuracy."""
    fig, ax = plt.subplots(figsize=(7, 5))

    keys_order = [
        "simclr_baseline", "simclr_dimreg_0.1", "simclr_dimreg_0.25",
        "simclr_dimreg_0.5", "simclr_dimreg_1.0",
        "vicreg_baseline", "vicreg_dimreg",
    ]

    xs = [RESULTS[k]["effective_rank"] for k in keys_order]
    ys = [RESULTS[k]["vn_entropy"] for k in keys_order]
    accs = [RESULTS[k]["clean_acc"] * 100 for k in keys_order]

    scatter = ax.scatter(xs, ys, c=accs, s=250, cmap="viridis",
                         edgecolor="black", linewidth=1.5, zorder=3)

    # Annotate points
    annotations = {
        "simclr_baseline": ("SimCLR", (8, -12)),
        "simclr_dimreg_0.1": (r"$\lambda$=0.1", (8, 4)),
        "simclr_dimreg_0.25": (r"$\lambda$=0.25", (8, 4)),
        "simclr_dimreg_0.5": (r"$\lambda$=0.5", (8, 4)),
        "simclr_dimreg_1.0": (r"$\lambda$=1.0", (-35, -15)),
        "vicreg_baseline": ("VICReg", (-55, 4)),
        "vicreg_dimreg": ("VICReg+L$_{dim}$", (-15, -18)),
    }

    for key, (label, offset) in annotations.items():
        x = RESULTS[key]["effective_rank"]
        y = RESULTS[key]["vn_entropy"]
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                    fontsize=9, fontweight="bold")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Clean accuracy (%)")

    ax.set_xlabel("Effective rank")
    ax.set_ylabel("Von Neumann entropy")
    ax.set_title("Representation geometry across all configurations", pad=12)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(outdir / "fig6_rank_vs_entropy.pdf")
    plt.savefig(outdir / "fig6_rank_vs_entropy.png")
    plt.close()
    print(f"  Saved: fig6_rank_vs_entropy.{{pdf,png}}")


def plot_summary_table(outdir: Path):
    """Figure 7: Summary table as a figure (for insertion into thesis)."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")

    rows = [
        ["Method", "$\\lambda$", "Clean Acc", "Corrupt Acc", "Rel. Rob.",
         "Eff. Rank", "VN Ent."],
        ["SimCLR (baseline)",    "–",    "48.49%", "39.01%", "80.46%", "121.6", "4.80"],
        ["SimCLR + L$_{dim}$",   "0.1",  "48.21%", "38.68%", "80.23%", "210.0", "5.35"],
        ["SimCLR + L$_{dim}$",   "0.25", "45.49%", "37.22%", "81.83%", "302.1", "5.71"],
        ["SimCLR + L$_{dim}$",   "0.5",  "42.05%", "34.67%", "82.45%", "363.5", "5.90"],
        ["SimCLR + L$_{dim}$",   "1.0",  "34.09%", "27.64%", "81.08%", "411.8", "6.02"],
        ["VICReg (baseline)",    "–",    "47.20%", "38.70%", "81.98%", "159.1", "5.07"],
        ["VICReg + L$_{dim}$",   "0.05", "47.68%", "38.82%", "81.43%", "166.6", "5.12"],
    ]

    table = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center",
                     cellLoc="center", colWidths=[0.22, 0.08, 0.12, 0.12,
                                                  0.12, 0.12, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    # Style header
    for j in range(len(rows[0])):
        cell = table[(0, j)]
        cell.set_facecolor("#0f3460")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#333333")

    # Style data rows
    for i in range(1, len(rows)):
        for j in range(len(rows[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#f5f5f5")
            else:
                cell.set_facecolor("white")
            cell.set_edgecolor("#cccccc")

    # Highlight the peak relative robustness (SimCLR + L_dim λ=0.5)
    for j in range(len(rows[0])):
        cell = table[(4, j)]  # row index 4 corresponds to λ=0.5
        cell.set_facecolor("#fff4e6")
        cell.set_text_props(fontweight="bold")

    plt.title("Complete experimental results summary", pad=18,
              fontsize=13, fontweight="bold")
    plt.savefig(outdir / "fig7_summary_table.pdf")
    plt.savefig(outdir / "fig7_summary_table.png")
    plt.close()
    print(f"  Saved: fig7_summary_table.{{pdf,png}}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./figures")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_style()

    print(f"\nGenerating figures in: {outdir.absolute()}")
    print("=" * 60)

    plot_tradeoff_curve(outdir)
    plot_robustness_curve(outdir)
    plot_training_trajectories(outdir)
    plot_category_robustness(outdir)
    plot_method_comparison(outdir)
    plot_rank_vs_entropy(outdir)
    plot_summary_table(outdir)

    print("=" * 60)
    print(f"All figures saved to: {outdir.absolute()}")
    print("\nGenerated figures:")
    print("  Fig 1: Accuracy vs effective rank trade-off curve")
    print("  Fig 2: Relative robustness vs lambda")
    print("  Fig 3: Training trajectories (rank & accuracy)")
    print("  Fig 4: Per-category robustness comparison")
    print("  Fig 5: SimCLR vs VICReg cross-method comparison")
    print("  Fig 6: Geometry scatter (rank vs entropy)")
    print("  Fig 7: Summary results table")


if __name__ == "__main__":
    main()
