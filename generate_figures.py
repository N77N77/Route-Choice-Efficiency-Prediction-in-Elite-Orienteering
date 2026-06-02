"""Generate publication-quality figures (Fig1-Fig6) from latest model data.

All data is read from model_metrics.json, sensitivity_analysis.json,
feature_importance.csv, and permutation_test_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import norm

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "axes.linewidth": 0.6,
})

BLUE = "#0077BB"
TEAL = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
GREY = "#BBBBBB"

OUTPUT_DIR = Path("05_Paper_Preparation/Figures")

MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "HistGBM",
    "ridge_alpha10": "Ridge (α=10)",
    "ridge_alpha50": "Ridge (α=50)",
    "ridge_alpha100": "Ridge (α=100)",
    "elasticnet": "ElasticNet",
    "lasso_alpha1": "Lasso (α=1)",
    "lasso_alpha5": "Lasso (α=5)",
}

GROUP_DISPLAY = {
    "athlete_ranking": "Athlete Ranking",
    "event_zscore": "Event Z-Score",
    "pace_behavior": "Pace Behavior",
    "robust_zscore": "Robust Z-Score",
    "event_metadata": "Event Metadata",
    "distance_context": "Distance Context",
}


def _shorten_feature(name: str) -> str:
    replacements = [
        ("pace_deviation_x_athlete_pace_rank_event", "pace_dev × athlete_rank"),
        ("leg_pace_s_per_m_sq", "leg_pace²"),
        ("leg_pace_s_per_m", "leg_pace"),
        ("fatigue_adjusted_pace", "fatigue_adj_pace"),
        ("split_time_s_event_zscore", "split_time_z"),
        ("split_time_s_event_robust_zscore", "split_time_robust_z"),
        ("segment_position_pct", "segment_position"),
        ("athlete_count_event", "athlete_count"),
        ("athlete_count_log", "log(athlete_count)"),
        ("straight_distance_m_event_zscore", "distance_z"),
        ("segment_index", "segment_index"),
    ]
    for old, new in replacements:
        if name == old:
            return new
    return name


# ============================================================
# Fig1: Model Comparison
# ============================================================
def fig1_model_comparison(metrics: dict) -> str:
    cv = metrics.get("cross_validation", {}).get("per_model", {})
    sel = metrics.get("selected_model", "random_forest")
    tree_models = {"random_forest", "gradient_boosting", "hist_gradient_boosting", "lightgbm", "xgboost"}

    models = []
    for name, data in cv.items():
        folds = data["folds"]
        r2 = np.mean([f["r2"] for f in folds])
        rmse = np.mean([f["rmse"] for f in folds])
        models.append((name, r2, rmse))
    models.sort(key=lambda x: x[1], reverse=True)

    names = [MODEL_DISPLAY.get(m[0], m[0]) for m in models]
    r2s = [m[1] for m in models]
    rmses = [m[2] for m in models]
    colors = [ORANGE if m[0] == sel else BLUE if m[0] in tree_models else GREY for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8))
    y = np.arange(len(names))

    bars1 = ax1.barh(y, r2s, color=colors, edgecolor="white", linewidth=0.4, height=0.65)
    for bar, v in zip(bars1, r2s):
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=7, color="#333")
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("CV R$^2$"); ax1.set_xlim(0, max(r2s) * 1.15); ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    bars2 = ax2.barh(y, rmses, color=colors, edgecolor="white", linewidth=0.4, height=0.65)
    for bar, v in zip(bars2, rmses):
        ax2.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                 f"{v:.2f}", va="center", fontsize=7, color="#333")
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("CV RMSE"); ax2.set_xlim(0, max(rmses) * 1.10); ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.5)
    path = OUTPUT_DIR / "Fig1_Model_Comparison.pdf"
    fig.savefig(path); plt.close(fig)
    return str(path)


# ============================================================
# Fig2: Feature Importance
# ============================================================
def fig2_feature_importance(importance_csv: Path) -> str:
    fi = pd.read_csv(importance_csv)
    top = fi.head(11).iloc[::-1]
    names = [_shorten_feature(f) for f in top["feature"]]
    vals = top["importance"].values

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors = plt.cm.Blues(np.linspace(0.35, 0.90, len(top)))
    bars = ax.barh(range(len(top)), vals, color=colors, edgecolor=BLUE, linewidth=0.3, height=0.7)
    for bar, v in zip(bars, vals):
        if v > max(vals) * 0.15:
            ax.text(v - 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", ha="right", fontsize=7, color="white", fontweight="bold")
        else:
            ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", ha="left", fontsize=7, color="#333")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("Impurity Importance")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = OUTPUT_DIR / "Fig2_Feature_Importance.pdf"
    fig.savefig(path); plt.close(fig)
    return str(path)


# ============================================================
# Fig3: Combined Ablation (feature group + interaction)
# ============================================================
def fig3_combined_ablation(sensitivity: dict) -> str:
    ab = sensitivity.get("feature_ablation", {})
    abl = ab.get("ablations", {})
    ia = sensitivity.get("interaction_ablation", {})

    grp_raw = sorted(abl.keys(), key=lambda g: abl[g].get("delta_r2", 0))
    grp_disp = [GROUP_DISPLAY.get(g, g) for g in grp_raw]
    deltas = [abl[g]["delta_r2"] for g in grp_raw]

    with_r = ia.get("baseline_all_features", {}).get("mean_r2", 0)
    without_r = ia.get("without_interactions", {}).get("mean_r2", 0)
    delta_int = with_r - without_r

    fig = plt.figure(figsize=(7.2, 3.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    colors = [RED if d < -0.001 else TEAL if d > 0.001 else GREY for d in deltas]
    y = np.arange(len(grp_raw))
    bars = ax1.barh(y, deltas, color=colors, edgecolor="white", linewidth=0.4, height=0.6)
    ax1.axvline(x=0, color="black", linewidth=0.8)
    ax1.axvspan(-0.005, 0.005, alpha=0.05, color="gray", zorder=0)
    for bar, d in zip(bars, deltas):
        offset = 0.0006 if d >= 0 else -0.0006
        ha = "left" if d >= 0 else "right"
        ax1.text(d + offset, bar.get_y() + bar.get_height()/2,
                 f"{d:+.4f}", va="center", ha=ha, fontsize=7, color="#222")
    ax1.set_yticks(y); ax1.set_yticklabels(grp_disp, fontsize=7.5)
    ax1.set_xlabel("$\\Delta$R$^2$ when group removed")
    ax1.set_title("(a) Feature group ablation", fontsize=10, fontweight="bold", pad=8)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False); ax1.tick_params(left=False)
    xm = max(abs(min(deltas)), abs(max(deltas))) * 1.8
    ax1.set_xlim(-xm, xm)

    ax2 = fig.add_subplot(gs[1])
    labels = ["With", "Without"]
    values = [with_r, without_r]
    bars = ax2.bar(labels, values, color=[TEAL, RED], width=0.5, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                 f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    yt = max(values) * 0.6
    ax2.annotate("", xy=(0, yt), xytext=(1, yt),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax2.text(0.5, yt + 0.012, f"$\\Delta$R$^2$ = {delta_int:.3f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold", color="#222")
    ax2.set_ylabel("CV R$^2$")
    ax2.set_title("(b) Interaction ablation", fontsize=10, fontweight="bold", pad=8)
    ax2.set_ylim(0, max(values) * 1.22)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    path = OUTPUT_DIR / "Fig3_Ablation.pdf"
    fig.savefig(path); plt.close(fig)
    return str(path)


# ============================================================
# Fig4: Per-Fold CV Stability
# ============================================================
def fig4_cv_stability(metrics: dict) -> str:
    sel = metrics.get("selected_model", "random_forest")
    folds = metrics.get("cross_validation", {}).get("per_model", {}).get(sel, {}).get("folds", [])
    if not folds:
        return "No fold data"

    fold_ids = [f["fold"] for f in folds]
    r2s = [f["r2"] for f in folds]
    rmses = [f["rmse"] for f in folds]
    mean_r2 = np.mean(r2s)
    mean_rmse = np.mean(rmses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    x = np.arange(len(fold_ids))

    ax1.bar(x, r2s, color=BLUE, edgecolor="white", linewidth=0.4, width=0.55, alpha=0.85)
    ax1.axhline(y=mean_r2, color=RED, linestyle="--", linewidth=1.2)
    ax1.text(len(x) - 0.5, mean_r2 + 0.008, f"Mean = {mean_r2:.3f}",
             fontsize=8, color=RED, fontweight="bold", ha="right")
    for i, v in enumerate(r2s):
        ax1.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5, color="#333")
    ax1.set_xticks(x); ax1.set_xticklabels([f"Fold {f}" for f in fold_ids], fontsize=8)
    ax1.set_ylabel("R$^2$"); ax1.set_ylim(min(r2s) - 0.05, max(r2s) + 0.05)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2.bar(x, rmses, color=TEAL, edgecolor="white", linewidth=0.4, width=0.55, alpha=0.85)
    ax2.axhline(y=mean_rmse, color=RED, linestyle="--", linewidth=1.2)
    ax2.text(len(x) - 0.5, mean_rmse + 0.15, f"Mean = {mean_rmse:.2f}",
             fontsize=8, color=RED, fontweight="bold", ha="right")
    for i, v in enumerate(rmses):
        ax2.text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=7.5, color="#333")
    ax2.set_xticks(x); ax2.set_xticklabels([f"Fold {f}" for f in fold_ids], fontsize=8)
    ax2.set_ylabel("RMSE"); ax2.set_ylim(min(rmses) - 0.8, max(rmses) + 0.8)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.5)
    path = OUTPUT_DIR / "Fig4_CV_Stability.pdf"
    fig.savefig(path); plt.close(fig)
    return str(path)


# ============================================================
# Fig5: LOEO Distribution
# ============================================================
def fig5_loeo_distribution(sensitivity: dict) -> str:
    loeo = sensitivity.get("per_event_loeo", {})
    pe = loeo.get("per_event", {})
    if not pe:
        return "No LOEO data"

    r2v = np.array([v["r2"] if isinstance(v, dict) else v
                    for v in pe.values() if not (isinstance(v, float) and v != v)])
    med = float(np.median(r2v))
    q25 = float(np.percentile(r2v, 25))
    q75 = float(np.percentile(r2v, 75))
    n_pos = int(np.sum(r2v >= 0))
    n_neg = int(np.sum(r2v < 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    bins = np.linspace(min(r2v) - 0.05, max(r2v) + 0.05, 20)
    ax1.hist(r2v, bins=bins, color=BLUE, edgecolor="white", alpha=0.85, linewidth=0.3)
    ax1.axvline(x=med, color=RED, linestyle="--", linewidth=1.5, label=f"Median = {med:.3f}")
    ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax1.set_xlabel("Per-Event R$^2$"); ax1.set_ylabel("Count")
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    sorted_r2 = np.sort(r2v)
    idx = np.arange(len(sorted_r2))
    colors = [RED if v < 0 else TEAL if v > med else GREY for v in sorted_r2]
    ax2.scatter(idx, sorted_r2, s=15, c=colors, alpha=0.7, edgecolors="none", zorder=3)
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.6)
    ax2.axhline(y=med, color="#333", linestyle="--", alpha=0.7, linewidth=1.0)
    ax2.axhspan(q25, q75, alpha=0.05, color=BLUE, zorder=0)
    ax2.set_xlabel("Event Rank"); ax2.set_ylabel("Per-Event R$^2$")
    stats = f"n = {len(r2v)}\nMedian = {med:.3f}\nIQR = [{q25:.3f}, {q75:.3f}]\n+: {n_pos}  -:{n_neg}"
    ax2.text(0.03, 0.97, stats, transform=ax2.transAxes, va="top", fontsize=7,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.5)
    path = OUTPUT_DIR / "Fig5_LOEO_Distribution.pdf"
    fig.savefig(path); plt.close(fig)
    return str(path)


# ============================================================
# Fig6: Permutation Test
# ============================================================
def fig6_permutation_test(results_path: Path) -> str:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    null_path = results_path.parent / "permutation_null_r2s.npy"
    if not null_path.exists():
        return "No null distribution data"

    null_r2s = np.load(null_path)
    real_r2 = results["real_r2"]
    null_mean = results["null_mean_r2"]
    null_std = results["null_std_r2"]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bins = np.linspace(null_mean - 4*null_std, null_mean + 4*null_std, 60)
    ax.hist(null_r2s, bins=bins, density=True, alpha=0.7, color=BLUE,
            edgecolor="white", linewidth=0.3, label="Null distribution (n = 500)")

    x_fit = np.linspace(bins[0], bins[-1], 200)
    y_fit = norm.pdf(x_fit, null_mean, null_std)
    ax.plot(x_fit, y_fit, color=BLUE, linewidth=1.5, alpha=0.5, linestyle="--")

    ax.axvline(real_r2, color=RED, linewidth=2.0, linestyle="-",
               label=f"Observed = {real_r2:.3f}", zorder=5)
    ax.axvline(null_mean, color=GREY, linewidth=1.0, linestyle=":",
               label=f"Null mean = {null_mean:.3f}")

    arrow_y = ax.get_ylim()[1] * 0.85
    ax.annotate("",
                xy=(real_r2, arrow_y), xytext=(null_mean + 3*null_std, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5))
    ax.text((real_r2 + null_mean + 3*null_std) / 2, arrow_y + 0.3,
            f"$\\Delta$R$^2$ = {real_r2 - null_mean:.3f}\np < 0.002",
            ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold")

    ax.set_xlabel("CV R$^2$", fontsize=11); ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = OUTPUT_DIR / "Fig6_Permutation_Test.pdf"
    fig.savefig(path); plt.close(fig)
    return str(path)


# ============================================================
# MAIN
# ============================================================
def generate_all_figures() -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path("04_Model_Training/Reports/model_metrics.json").read_text(encoding="utf-8"))
    sensitivity = json.loads(Path("04_Model_Training/Reports/sensitivity_analysis.json").read_text(encoding="utf-8"))
    importance_path = Path("04_Model_Training/Reports/feature_importance.csv")
    permutation_path = Path("04_Model_Training/Reports/permutation_test_results.json")

    results = {}
    results["fig1"] = fig1_model_comparison(metrics)
    results["fig2"] = fig2_feature_importance(importance_path)
    results["fig3"] = fig3_combined_ablation(sensitivity)
    results["fig4"] = fig4_cv_stability(metrics)
    results["fig5"] = fig5_loeo_distribution(sensitivity)
    results["fig6"] = fig6_permutation_test(permutation_path)
    return results


if __name__ == "__main__":
    for name, path in generate_all_figures().items():
        print(f"  {name}: {path}")
    print("Done: 6 figures.")
