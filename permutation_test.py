"""Permutation test for M1 circularity check.

Shuffles RCE within each event group to build a null distribution of CV R².
If real R² >> null, the model captures behavioural signal beyond structural
circularity (shared d_straight component between pace features and RCE).

Usage:
    python permutation_test.py              # default 500 permutations
    python permutation_test.py --n 1000     # 1000 permutations
    python permutation_test.py --plot-only  # replot from saved results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "06_Code_Repository" / "src"))

from route_choice_efficiency.config import PATHS
from route_choice_efficiency.sensitivity import run_permutation_test


def plot_permutation_result(result: dict, output_path: Path) -> None:
    """Plot null distribution histogram with observed R² line."""
    null_r2s = np.array(result["null_r2s"])
    real_r2 = result["real_r2"]
    p_value = result["p_value"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram of null distribution
    ax.hist(null_r2s, bins=40, density=True, alpha=0.7, color="#4C72B0",
            edgecolor="white", linewidth=0.5, label="Null distribution (permuted RCE)")

    # Observed R2 line
    ax.axvline(real_r2, color="#C44E52", linewidth=2.5, linestyle="--",
               label=f"Observed CV R2 = {real_r2:.3f}")

    # Null mean
    ax.axvline(result["null_mean_r2"], color="gray", linewidth=1.5, linestyle=":",
               label=f"Null mean = {result['null_mean_r2']:.3f}")

    # Annotations
    ax.set_xlabel("Cross-Validated R2", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Permutation Test (n={result['n_permutations']})\n"
                 f"p-value = {p_value:.4f}", fontsize=13)

    # Effect size annotation
    effect = real_r2 - result["null_mean_r2"]
    ax.annotate(f"dR2 = {effect:.3f}\np = {p_value:.4f}",
                xy=(real_r2, ax.get_ylim()[1] * 0.85),
                fontsize=11, color="#C44E52", fontweight="bold",
                ha="left", va="top",
                xytext=(15, 0), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5))

    ax.legend(loc="upper left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Permutation test for circularity check (M1)")
    parser.add_argument("--n", type=int, default=500, help="Number of permutations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot-only", action="store_true", help="Replot from saved JSON results")
    args = parser.parse_args()

    results_dir = PATHS.model_reports
    json_path = results_dir / "permutation_test_results.json"
    plot_path = PATHS.figures / "Fig7_Permutation_Test.pdf"

    if args.plot_only:
        if not json_path.exists():
            print(f"ERROR: {json_path} not found. Run without --plot-only first.")
            sys.exit(1)
        result = json.loads(json_path.read_text(encoding="utf-8"))
        # Load null_r2s from separate file if available
        null_path = results_dir / "permutation_null_r2s.npy"
        if null_path.exists():
            result["null_r2s"] = np.load(null_path).tolist()
        else:
            print("WARNING: null_r2s not available, plot will be incomplete.")
            result["null_r2s"] = []
        plot_permutation_result(result, plot_path)
        return

    # Load data
    print("Loading data...")
    data = pd.read_csv(PATHS.terrain_features, dtype={"athlete_id": str})

    # Load selected features
    metrics = json.loads(PATHS.metrics.read_text(encoding="utf-8"))
    selected_features = metrics.get("selected_features", [])
    if not selected_features:
        print("ERROR: No selected features found in model_metrics.json")
        sys.exit(1)
    print(f"Using {len(selected_features)} features: {selected_features}")

    # Run permutation test
    print(f"\nRunning permutation test ({args.n} iterations, seed={args.seed})...")
    result = run_permutation_test(data, selected_features, n_permutations=args.n, seed=args.seed)

    # Save results
    save = {k: v for k, v in result.items() if k != "null_r2s"}
    json_path.write_text(json.dumps(save, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {json_path}")

    # Save full null distribution for later replotting
    np.save(results_dir / "permutation_null_r2s.npy", np.array(result["null_r2s"]))

    # Plot
    plot_permutation_result(result, plot_path)


if __name__ == "__main__":
    main()
