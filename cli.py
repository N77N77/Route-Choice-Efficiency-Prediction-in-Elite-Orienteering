"""CLI entry point — v2.0 stripped to essential commands only."""
from __future__ import annotations

import argparse
import json
import sys

from .config import PATHS


def _print_result(result: dict[str, str]) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(prog="route_choice_efficiency")
    subparsers = parser.add_subparsers(dest="command")

    # ── Core analysis pipeline ──
    features_parser = subparsers.add_parser("features", help="Generate terrain/behavioural features")
    features_parser.add_argument("--input", help="Optional CSV path to feature generation")

    train_parser = subparsers.add_parser("train", help="Train predictive models")
    train_parser.add_argument("--input", help="Optional CSV path to model training data")

    sensitivity_parser = subparsers.add_parser("sensitivity", help="Run sensitivity analysis")
    sensitivity_parser.add_argument("--input", help="Optional CSV path to model training data")

    subparsers.add_parser("paper-figures", help="Generate Fig1-6 from current model metrics")

    subparsers.add_parser("finalize", help="Run Nested CV + Bootstrap CI + SHAP analysis")

    perm_parser = subparsers.add_parser("permutation", help="Permutation test for circularity check (M1)")
    perm_parser.add_argument("--n", type=int, default=500, help="Number of permutations (default 500)")
    perm_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "features":
        from .features import calculate_terrain_features
        _print_result(calculate_terrain_features(input_csv=args.input))

    elif args.command == "train":
        from .modeling import train_models
        _print_result(train_models(input_csv=args.input))

    elif args.command == "sensitivity":
        from .sensitivity import run_sensitivity_analysis
        _print_result(run_sensitivity_analysis(input_csv=args.input))

    elif args.command == "paper-figures":
        from .paper_figures import generate_all_paper_figures
        for name, path in generate_all_paper_figures().items():
            print(f"  {name}: {path}")
        print("Done: 6 figures.")

    elif args.command == "finalize":
        from .finalize import run_all
        run_all()

    elif args.command == "permutation":
        from .sensitivity import run_permutation_test
        import pandas as pd
        source = PATHS.terrain_features
        data = pd.read_csv(source, dtype={"athlete_id": str})
        metrics = json.loads(PATHS.metrics.read_text(encoding="utf-8"))
        selected_features = metrics.get("selected_features", [])
        if not selected_features:
            print("ERROR: No selected features in model_metrics.json")
            sys.exit(1)
        result = run_permutation_test(data, selected_features, n_permutations=args.n, seed=args.seed)
        # Save results
        out_path = PATHS.model_reports / "permutation_test_results.json"
        # Exclude full null_r2s array from JSON (too large), keep summary
        save = {k: v for k, v in result.items() if k != "null_r2s"}
        out_path.write_text(json.dumps(save, indent=2), encoding="utf-8")
        print(f"\nResults saved to {out_path}")
