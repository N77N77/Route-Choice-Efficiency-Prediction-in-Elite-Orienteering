"""Sensitivity analysis for route choice efficiency model.

Provides feature ablation, CV stability, per-event analysis, and feature group
importance — all required for a rigorous SCI publication.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from .config import PATHS, MODEL_PARAMS


def _safe_r2(y_true, y_pred, sample_weight=None):
    if len(y_true) < 2 or pd.Series(y_true).nunique() < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred, sample_weight=sample_weight))


# ---------------------------------------------------------------------------
# Feature group definitions for ablation
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "event_zscore": [
        "leg_pace_s_per_m_event_zscore",
        "straight_distance_m_event_zscore",
        "split_time_s_event_zscore",
    ],
    "event_metadata": [
        "athlete_count_event",
        "athlete_count_log",
    ],
    "pace_behavior": [
        "leg_pace_s_per_m",
        "pace_deviation",
        "distance_pace_ratio",
        "fatigue_adjusted_pace",
        "pace_consistency_ratio",
    ],
    "athlete_ranking": [
        "athlete_pace_rank_event",
        "segment_position_pct",
    ],
    "robust_zscore": [
        "leg_pace_s_per_m_event_robust_zscore",
        "split_time_s_event_robust_zscore",
    ],
    "distance_context": [
        "straight_distance_m",
        "segment_index",
        "split_time_s",
    ],
}


def _prepare_features_from_model(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce the exact feature matrix from modeling._prepare_features."""
    from .modeling import _prepare_features
    return _prepare_features(data, add_interactions=True)


def _get_model():
    """Return a GradientBoosting model template for sensitivity analysis."""
    return GradientBoostingRegressor(**MODEL_PARAMS["gradient_boosting"])


def _cv_evaluate(features, target, groups, sample_weight, n_splits=5):
    """Run GroupKFold CV and return per-fold metrics."""
    group_series = groups.astype(str).reset_index(drop=True)
    n_splits = min(n_splits, group_series.nunique())
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_idx, valid_idx in splitter.split(features, target, group_series):
        model = _get_model()
        x_tr = features.iloc[train_idx]
        y_tr = target.iloc[train_idx]
        x_va = features.iloc[valid_idx]
        y_va = target.iloc[valid_idx]
        w_tr = sample_weight.iloc[train_idx]
        w_va = sample_weight.iloc[valid_idx]
        model.fit(x_tr, y_tr, sample_weight=w_tr)
        preds = np.clip(model.predict(x_va), 0, 100)
        folds.append({
            "r2": _safe_r2(y_va, preds, w_va),
            "rmse": float(np.sqrt(mean_squared_error(y_va, preds, sample_weight=w_va))),
            "mae": float(mean_absolute_error(y_va, preds, sample_weight=w_va)),
        })
    r2s = [f["r2"] for f in folds]
    rmses = [f["rmse"] for f in folds]
    maes = [f["mae"] for f in folds]
    return {
        "folds": folds,
        "mean_r2": float(np.nanmean(r2s)),
        "std_r2": float(np.nanstd(r2s)),
        "mean_rmse": float(np.nanmean(rmses)),
        "std_rmse": float(np.nanstd(rmses)),
        "mean_mae": float(np.nanmean(maes)),
        "std_mae": float(np.nanstd(maes)),
    }


# ---------------------------------------------------------------------------
# 1. Feature Ablation Study
# ---------------------------------------------------------------------------
def run_feature_ablation(data: pd.DataFrame, selected_features: list[str]) -> dict:
    """Remove each feature group one at a time and measure CV R² change."""
    x_all, y_all = _prepare_features_from_model(data)
    # v2.0: uniform weights — uncertainty_score and differential weighting removed
    if "quality_weight" in data.columns:
        sample_weight = data["quality_weight"]
    else:
        sample_weight = pd.Series(1.0, index=data.index)
    groups = data["event_id"]

    # Baseline with all selected features
    x_sel = x_all[selected_features]
    baseline = _cv_evaluate(x_sel, y_all, groups, sample_weight)

    ablation_results = {"baseline": baseline, "ablations": {}}

    for group_name, group_cols in FEATURE_GROUPS.items():
        # Features in selected_features that belong to this group
        cols_to_remove = [c for c in group_cols if c in selected_features]
        if not cols_to_remove:
            continue
        remaining = [c for c in selected_features if c not in cols_to_remove]
        if len(remaining) < 3:
            continue
        x_ablated = x_all[remaining]
        result = _cv_evaluate(x_ablated, y_all, groups, sample_weight)
        result["removed_features"] = cols_to_remove
        result["remaining_count"] = len(remaining)
        result["delta_r2"] = result["mean_r2"] - baseline["mean_r2"]
        ablation_results["ablations"][group_name] = result

    return ablation_results


# ---------------------------------------------------------------------------
# 2. Interaction Feature Ablation
# ---------------------------------------------------------------------------
def run_interaction_ablation(data: pd.DataFrame, selected_features: list[str]) -> dict:
    """Evaluate model with and without interaction/polynomial features."""
    x_all, y_all = _prepare_features_from_model(data)
    # v2.0: uniform weights — uncertainty_score and differential weighting removed
    if "quality_weight" in data.columns:
        sample_weight = data["quality_weight"]
    else:
        sample_weight = pd.Series(1.0, index=data.index)
    groups = data["event_id"]

    x_sel = x_all[selected_features]
    baseline = _cv_evaluate(x_sel, y_all, groups, sample_weight)

    # Identify interaction features (contain '_x_' or '_sq')
    interaction_cols = [c for c in selected_features if '_x_' in c or c.endswith('_sq')]
    base_cols = [c for c in selected_features if c not in interaction_cols]

    x_base_only = x_all[base_cols]
    base_result = _cv_evaluate(x_base_only, y_all, groups, sample_weight)

    return {
        "baseline_all_features": baseline,
        "without_interactions": base_result,
        "interaction_features": interaction_cols,
        "base_features": base_cols,
        "delta_r2": base_result["mean_r2"] - baseline["mean_r2"],
    }


# ---------------------------------------------------------------------------
# 3. CV Stability (Multiple Seeds / Fold Counts)
# ---------------------------------------------------------------------------
def run_cv_stability(data: pd.DataFrame, selected_features: list[str]) -> dict:
    """Assess CV stability across different random seeds and fold counts."""
    x_all, y_all = _prepare_features_from_model(data)
    # v2.0: uniform weights — uncertainty_score and differential weighting removed
    if "quality_weight" in data.columns:
        sample_weight = data["quality_weight"]
    else:
        sample_weight = pd.Series(1.0, index=data.index)
    groups = data["event_id"]
    x_sel = x_all[selected_features]

    results = {"by_seed": {}, "by_n_splits": {}}

    # Vary seed (GroupKFold is deterministic for a given group assignment,
    # so we vary subsample fraction to induce variation)
    for seed in [42, 123, 456, 789, 2026]:
        params = MODEL_PARAMS["gradient_boosting"].copy()
        params["random_state"] = seed
        model = GradientBoostingRegressor(**params)
        group_series = groups.astype(str).reset_index(drop=True)
        splitter = GroupKFold(n_splits=5)
        folds = []
        for train_idx, valid_idx in splitter.split(x_sel, y_all, group_series):
            m = clone(model)
            x_tr, y_tr = x_sel.iloc[train_idx], y_all.iloc[train_idx]
            x_va, y_va = x_sel.iloc[valid_idx], y_all.iloc[valid_idx]
            w_tr = sample_weight.iloc[train_idx]
            w_va = sample_weight.iloc[valid_idx]
            m.fit(x_tr, y_tr, sample_weight=w_tr)
            preds = np.clip(m.predict(x_va), 0, 100)
            folds.append({"r2": _safe_r2(y_va, preds, w_va)})
        r2s = [f["r2"] for f in folds]
        results["by_seed"][str(seed)] = {
            "mean_r2": float(np.nanmean(r2s)),
            "std_r2": float(np.nanstd(r2s)),
            "per_fold_r2": r2s,
        }

    # Vary number of folds
    for k in [3, 4, 5, 7, 10]:
        group_series = groups.astype(str).reset_index(drop=True)
        n_groups = group_series.nunique()
        if k > n_groups:
            continue
        splitter = GroupKFold(n_splits=k)
        model = _get_model()
        folds = []
        for train_idx, valid_idx in splitter.split(x_sel, y_all, group_series):
            m = clone(model)
            x_tr, y_tr = x_sel.iloc[train_idx], y_all.iloc[train_idx]
            x_va, y_va = x_sel.iloc[valid_idx], y_all.iloc[valid_idx]
            w_tr = sample_weight.iloc[train_idx]
            w_va = sample_weight.iloc[valid_idx]
            m.fit(x_tr, y_tr, sample_weight=w_tr)
            preds = np.clip(m.predict(x_va), 0, 100)
            folds.append({"r2": _safe_r2(y_va, preds, w_va)})
        r2s = [f["r2"] for f in folds]
        results["by_n_splits"][str(k)] = {
            "mean_r2": float(np.nanmean(r2s)),
            "std_r2": float(np.nanstd(r2s)),
            "per_fold_r2": r2s,
        }

    return results


# ---------------------------------------------------------------------------
# 4. Per-Event Performance Analysis
# ---------------------------------------------------------------------------
def run_per_event_analysis(data: pd.DataFrame, selected_features: list[str]) -> dict:
    """Train on all-but-one event and test on the held-out event."""
    x_all, y_all = _prepare_features_from_model(data)
    # v2.0: uniform weights — uncertainty_score and differential weighting removed
    if "quality_weight" in data.columns:
        sample_weight = data["quality_weight"]
    else:
        sample_weight = pd.Series(1.0, index=data.index)
    groups = data["event_id"].astype(str)
    x_sel = x_all[selected_features]

    unique_events = groups.unique()
    event_results = {}

    for event_id in unique_events:
        train_mask = groups != event_id
        test_mask = groups == event_id
        if test_mask.sum() < 5:
            continue
        model = _get_model()
        x_tr, y_tr = x_sel[train_mask], y_all[train_mask]
        x_te, y_te = x_sel[test_mask], y_all[test_mask]
        w_tr = sample_weight[train_mask]
        w_te = sample_weight[test_mask]
        model.fit(x_tr, y_tr, sample_weight=w_tr)
        preds = np.clip(model.predict(x_te), 0, 100)
        event_results[event_id] = {
            "n_test": int(test_mask.sum()),
            "r2": _safe_r2(y_te, preds, w_te),
            "rmse": float(np.sqrt(mean_squared_error(y_te, preds, sample_weight=w_te))),
            "mae": float(mean_absolute_error(y_te, preds, sample_weight=w_te)),
            "mean_actual": float(y_te.mean()),
            "mean_predicted": float(preds.mean()),
        }

    r2s = [v["r2"] for v in event_results.values() if not np.isnan(v["r2"])]
    return {
        "per_event": event_results,
        "summary": {
            "n_events": len(event_results),
            "mean_r2": float(np.nanmean(r2s)) if r2s else float("nan"),
            "std_r2": float(np.nanstd(r2s)) if r2s else float("nan"),
            "min_r2": float(np.nanmin(r2s)) if r2s else float("nan"),
            "max_r2": float(np.nanmax(r2s)) if r2s else float("nan"),
            "median_r2": float(np.nanmedian(r2s)) if r2s else float("nan"),
        },
    }


# ---------------------------------------------------------------------------
# 5. Feature Group Importance (aggregated)
# ---------------------------------------------------------------------------
def run_feature_group_importance(data: pd.DataFrame, selected_features: list[str]) -> dict:
    """Aggregate feature importance by feature group."""
    from .modeling import _select_features_by_importance

    x_all, y_all = _prepare_features_from_model(data)
    # v2.0: uniform weights — uncertainty_score and differential weighting removed
    if "quality_weight" in data.columns:
        sample_weight = data["quality_weight"]
    else:
        sample_weight = pd.Series(1.0, index=data.index)

    # Fit model and get importances
    model = _get_model()
    model.fit(x_all[selected_features], y_all, sample_weight=sample_weight)

    importance_dict = dict(zip(selected_features, model.feature_importances_))

    group_importance = {}
    for group_name, group_cols in FEATURE_GROUPS.items():
        group_imp = sum(importance_dict.get(c, 0) for c in group_cols)
        group_importance[group_name] = {
            "total_importance": float(group_imp),
            "features_in_model": [c for c in group_cols if c in selected_features],
            "n_features": sum(1 for c in group_cols if c in selected_features),
        }

    # Also account for interaction features
    interaction_cols = [c for c in selected_features if '_x_' in c or c.endswith('_sq')]
    interaction_imp = sum(importance_dict.get(c, 0) for c in interaction_cols)
    group_importance["interactions"] = {
        "total_importance": float(interaction_imp),
        "features_in_model": interaction_cols,
        "n_features": len(interaction_cols),
    }

    return group_importance


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_sensitivity_analysis(input_csv: str | None = None) -> dict[str, str]:
    """Run all sensitivity analyses and save results."""
    source = PATHS.terrain_features if input_csv is None else PATHS.project_root / input_csv
    data = pd.read_csv(source, dtype={"athlete_id": str})
    data["event_date"] = pd.to_datetime(data["event_date"])

    # Load selected features from previous training
    metrics = json.loads(PATHS.metrics.read_text(encoding="utf-8"))
    selected_features = metrics.get("selected_features", [])

    if not selected_features:
        raise ValueError("No selected features found in model_metrics.json")

    print("Running feature ablation...")
    ablation = run_feature_ablation(data, selected_features)

    print("Running interaction ablation...")
    interaction = run_interaction_ablation(data, selected_features)

    print("Running CV stability analysis...")
    stability = run_cv_stability(data, selected_features)

    print("Running per-event LOEO analysis...")
    per_event = run_per_event_analysis(data, selected_features)

    print("Running feature group importance...")
    group_importance = run_feature_group_importance(data, selected_features)

    results = {
        "feature_ablation": ablation,
        "interaction_ablation": interaction,
        "cv_stability": stability,
        "per_event_loeo": per_event,
        "feature_group_importance": group_importance,
    }

    # Save
    output_dir = PATHS.model_reports
    output_path = output_dir / "sensitivity_analysis.json"
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Sensitivity analysis saved to {output_path}")

    # Generate summary report
    report_path = output_dir / "Sensitivity_Analysis_Report.md"
    report_path.write_text(_generate_report(results, selected_features), encoding="utf-8")
    print(f"Report saved to {report_path}")

    return {"sensitivity_json": str(output_path), "sensitivity_report": str(report_path)}


def _generate_report(results: dict, selected_features: list[str]) -> str:
    """Generate a markdown sensitivity analysis report."""
    lines = ["# Sensitivity Analysis Report", ""]

    # Baseline
    baseline = results["feature_ablation"]["baseline"]
    lines.append("## 1. Baseline Model Performance")
    lines.append(f"- CV R²: {baseline['mean_r2']:.4f} ± {baseline['std_r2']:.4f}")
    lines.append(f"- CV RMSE: {baseline['mean_rmse']:.4f} ± {baseline['std_rmse']:.4f}")
    lines.append(f"- CV MAE: {baseline['mean_mae']:.4f} ± {baseline['std_mae']:.4f}")
    lines.append("")

    # Feature ablation
    lines.append("## 2. Feature Ablation Study")
    lines.append("| Feature Group | Removed | ΔR² | CV R² | Interpretation |")
    lines.append("|---|---|---|---|---|")
    for group, result in sorted(
        results["feature_ablation"]["ablations"].items(),
        key=lambda x: x[1].get("delta_r2", 0),
    ):
        delta = result.get("delta_r2", 0)
        interp = "Critical" if delta < -0.05 else "Important" if delta < -0.02 else "Minor"
        lines.append(
            f"| {group} | {result['remaining_count']} remaining | "
            f"{delta:+.4f} | {result['mean_r2']:.4f} | {interp} |"
        )
    lines.append("")

    # Interaction ablation
    lines.append("## 3. Interaction Feature Ablation")
    ia = results["interaction_ablation"]
    lines.append(f"- With interactions: R² = {ia['baseline_all_features']['mean_r2']:.4f}")
    lines.append(f"- Without interactions: R² = {ia['without_interactions']['mean_r2']:.4f}")
    lines.append(f"- ΔR² = {ia['delta_r2']:+.4f}")
    lines.append(f"- Interaction features ({len(ia['interaction_features'])}): {', '.join(ia['interaction_features'][:10])}")
    lines.append("")

    # CV stability
    lines.append("## 4. Cross-Validation Stability")
    lines.append("### By Random Seed")
    lines.append("| Seed | Mean R² | Std R² |")
    lines.append("|---|---|---|")
    for seed, vals in results["cv_stability"]["by_seed"].items():
        lines.append(f"| {seed} | {vals['mean_r2']:.4f} | {vals['std_r2']:.4f} |")
    lines.append("")
    lines.append("### By Number of Folds")
    lines.append("| K | Mean R² | Std R² |")
    lines.append("|---|---|---|")
    for k, vals in results["cv_stability"]["by_n_splits"].items():
        lines.append(f"| {k} | {vals['mean_r2']:.4f} | {vals['std_r2']:.4f} |")
    lines.append("")

    # Per-event LOEO
    lines.append("## 5. Leave-One-Event-Out (LOEO) Analysis")
    loeo = results["per_event_loeo"]["summary"]
    lines.append(f"- Events evaluated: {loeo['n_events']}")
    lines.append(f"- Mean R²: {loeo['mean_r2']:.4f} ± {loeo['std_r2']:.4f}")
    lines.append(f"- Range: [{loeo['min_r2']:.4f}, {loeo['max_r2']:.4f}]")
    lines.append(f"- Median R²: {loeo['median_r2']:.4f}")
    lines.append("")

    # Feature group importance
    lines.append("## 6. Feature Group Importance (Aggregated)")
    lines.append("| Group | Importance | # Features |")
    lines.append("|---|---|---|")
    for group, vals in sorted(
        results["feature_group_importance"].items(),
        key=lambda x: x[1]["total_importance"],
        reverse=True,
    ):
        lines.append(f"| {group} | {vals['total_importance']:.4f} | {vals['n_features']} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Permutation Test — Circularity Check (M1)
# ---------------------------------------------------------------------------
def _cv_evaluate_perm(features, target, groups, sample_weight, n_splits=5):
    """GroupKFold CV using GradientBoosting for permutation test."""
    group_series = groups.astype(str).reset_index(drop=True)
    n_splits = min(n_splits, group_series.nunique())
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_idx, valid_idx in splitter.split(features, target, group_series):
        model = GradientBoostingRegressor(**MODEL_PARAMS["gradient_boosting"])
        x_tr = features.iloc[train_idx]
        y_tr = target.iloc[train_idx]
        x_va = features.iloc[valid_idx]
        y_va = target.iloc[valid_idx]
        w_tr = sample_weight.iloc[train_idx]
        model.fit(x_tr, y_tr, sample_weight=w_tr)
        preds = np.clip(model.predict(x_va), 0, 100)
        folds.append({"r2": _safe_r2(y_va, preds)})
    r2s = [f["r2"] for f in folds]
    return {"mean_r2": float(np.nanmean(r2s)), "std_r2": float(np.nanstd(r2s)), "folds": folds}


def run_permutation_test(
    data: pd.DataFrame,
    selected_features: list[str],
    n_permutations: int = 500,
    seed: int = 42,
) -> dict:
    """Permutation test to assess whether model R² exceeds structural-circularity baseline.

    Within each event group, RCE values are randomly shuffled.  If the real
    model R² is far above the null distribution, the model captures genuine
    behavioural signal rather than merely exploiting the shared d_straight
    component between pace features and RCE.

    Parameters
    ----------
    data : DataFrame with terrain features
    selected_features : list of 11 locked feature names
    n_permutations : number of null iterations (default 500)
    seed : random seed for reproducibility

    Returns
    -------
    dict with real_r2, null distribution stats, p_value, and full null_r2s array.
    """
    x_all, y_all = _prepare_features_from_model(data)
    x_sel = x_all[selected_features]
    groups = data["event_id"]
    weights = pd.Series(1.0, index=data.index)

    # Real CV R²
    print(f"  [permutation] Running real CV (GradientBoosting)...")
    real_result = _cv_evaluate_perm(x_sel, y_all, groups, weights)
    real_r2 = real_result["mean_r2"]
    print(f"  [permutation] Real CV R2 = {real_r2:.4f}")

    # Null distribution: shuffle y within each event group
    rng = np.random.RandomState(seed)
    null_r2s = np.empty(n_permutations)
    group_values = groups.values
    unique_groups = np.unique(group_values)
    # Pre-compute group index arrays
    group_indices = {g: np.where(group_values == g)[0] for g in unique_groups}
    y_arr = y_all.values.copy()
    for i in range(n_permutations):
        y_perm_arr = y_arr.copy()
        for g in unique_groups:
            idx = group_indices[g]
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            y_perm_arr[idx] = y_arr[shuffled]
        y_perm = pd.Series(y_perm_arr, index=y_all.index)
        perm_result = _cv_evaluate_perm(x_sel, y_perm, groups, weights)
        null_r2s[i] = perm_result["mean_r2"]
        if (i + 1) % 50 == 0:
            print(f"  [permutation] {i+1}/{n_permutations} done, running null mean R2 = {null_r2s[:i+1].mean():.4f}")

    # p-value: proportion of null R² >= observed R²
    p_value = float((null_r2s >= real_r2).sum() / n_permutations)

    result = {
        "real_r2": real_r2,
        "null_mean_r2": float(null_r2s.mean()),
        "null_std_r2": float(null_r2s.std()),
        "null_95_ci": [
            float(np.percentile(null_r2s, 2.5)),
            float(np.percentile(null_r2s, 97.5)),
        ],
        "p_value": p_value,
        "n_permutations": n_permutations,
        "seed": seed,
        "null_r2s": null_r2s.tolist(),
    }

    print(f"\n  [permutation] === RESULTS ===")
    print(f"  Real R2:       {real_r2:.4f}")
    print(f"  Null mean R2:  {null_r2s.mean():.4f} +/- {null_r2s.std():.4f}")
    print(f"  Null 95% CI:   [{result['null_95_ci'][0]:.4f}, {result['null_95_ci'][1]:.4f}]")
    print(f"  p-value:       {p_value:.4f}")
    if p_value < 0.001:
        print("  Interpretation: Real R2 significantly exceeds null -> genuine behavioural signal.")
    elif p_value < 0.05:
        print("  Interpretation: Real R2 marginally above null -> partial behavioural signal.")
    else:
        print("  Interpretation: Real R2 not above null -> structural confounding dominates.")

    return result
