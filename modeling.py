from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import PATHS, MODEL_PARAMS


class WeightedEnsemble:
    """Weighted average ensemble model."""

    def __init__(self, models: dict, weights: dict):
        self.models = models
        self.weights = weights

    def predict(self, X):
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            preds += self.weights[name] * np.clip(model.predict(X), 0.0, 100.0)
        return np.clip(preds, 0.0, 100.0)


class StackingEnsemble:
    """Stacking ensemble: base models → meta-learner."""

    def __init__(self, base_models: dict, meta_model, feature_order: list[str]):
        self.base_models = base_models
        self.meta_model = meta_model
        self.feature_order = feature_order

    def predict(self, X):
        meta_X = pd.DataFrame(index=X.index)
        for name, model in self.base_models.items():
            preds = model.predict(X)
            meta_X[name] = np.clip(preds, 0.0, 100.0)
        return np.clip(self.meta_model.predict(meta_X), 0.0, 100.0)


MIN_ROWS_PER_CV_GROUP = 2


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _split_by_index(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = data.sort_values(["event_date", "event_id", "athlete_id", "segment_index"]).reset_index(drop=True)
    total = len(ordered)
    if total < 3:
        raise ValueError("At least 3 rows are required to train and evaluate a model.")
    train_end = max(1, int(total * 0.7))
    valid_end = max(train_end + 1, int(total * 0.85))
    if valid_end >= total:
        valid_end = total - 1
    if train_end >= valid_end:
        train_end = max(1, valid_end - 1)
    train = ordered.iloc[:train_end].copy()
    valid = ordered.iloc[train_end:valid_end].copy()
    test = ordered.iloc[valid_end:].copy()
    if valid.empty or test.empty:
        raise ValueError("Unable to create non-overlapping train/validation/test splits from the provided data.")
    return train, valid, test


def _split_by_date(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    unique_dates = sorted(data["event_date"].dt.normalize().unique())
    if len(unique_dates) < 3:
        train, valid, test = _split_by_index(data)
        return train, valid, test, "Fell back to non-overlapping row-based split because fewer than 3 unique event dates were available."
    train_cut = max(1, int(len(unique_dates) * 0.7))
    valid_cut = max(train_cut + 1, int(len(unique_dates) * 0.85))
    train_dates = set(unique_dates[:train_cut])
    valid_dates = set(unique_dates[train_cut:valid_cut])
    test_dates = set(unique_dates[valid_cut:])
    if not valid_dates:
        valid_dates = {unique_dates[-2]} if len(unique_dates) > 2 else {unique_dates[-1]}
    if not test_dates:
        test_dates = {unique_dates[-1]}
    train = data[data["event_date"].dt.normalize().isin(train_dates)].copy()
    valid = data[data["event_date"].dt.normalize().isin(valid_dates)].copy()
    test = data[data["event_date"].dt.normalize().isin(test_dates)].copy()
    overlap_exists = bool(train_dates & valid_dates or train_dates & test_dates or valid_dates & test_dates)
    if overlap_exists or train.empty or valid.empty or test.empty:
        train, valid, test = _split_by_index(data)
        return train, valid, test, "Fell back to non-overlapping row-based split because date-based partitioning produced overlap or empty splits."
    return train, valid, test, None


def _prepare_features(data: pd.DataFrame, add_interactions: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    # Target-derived columns (athlete_mean_re, event_mean_re, etc.) are excluded
    # to prevent data leakage.
    _desired = [
        "route_efficiency",
        "straight_distance_m",
        "segment_index",
        "leg_pace_s_per_m",
        "split_time_s",
        "course_type",
        "athlete_count_event",
        "distance_pace_ratio",
        "fatigue_adjusted_pace",
        "pace_deviation",
        "distance_x_pace",
        # Event-level z-score features (cross-event generalization)
        "leg_pace_s_per_m_event_zscore",
        "straight_distance_m_event_zscore",
        "split_time_s_event_zscore",
        # Robust z-score features
        "leg_pace_s_per_m_event_robust_zscore",
        "split_time_s_event_robust_zscore",
        # Athlete ranking and consistency
        "athlete_pace_rank_event",
        "pace_consistency_ratio",
        "segment_position_pct",
        # Pre-computed interaction
        "pace_deviation_x_athlete_pace_rank_event",
    ]
    # Only keep columns that actually exist in the data
    available = [c for c in _desired if c in data.columns]
    model_frame = data[available].copy()
    target = model_frame.pop("route_efficiency")
    numeric_columns = model_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in model_frame.columns if column not in numeric_columns]
    if numeric_columns:
        model_frame[numeric_columns] = model_frame[numeric_columns].apply(lambda series: series.fillna(series.median() if not series.dropna().empty else 0.0))
    for column in categorical_columns:
        model_frame[column] = model_frame[column].fillna("Unknown").astype(str).replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    features = pd.get_dummies(model_frame, drop_first=False)
    # Defragment before adding many interaction columns
    features = features.copy()

    # Add interaction features for top predictors
    if add_interactions:
        extra: dict[str, pd.Series] = {}

        top_features = ["leg_pace_s_per_m", "pace_deviation",
                         "leg_pace_s_per_m_event_zscore", "athlete_pace_rank_event",
                         "pace_consistency_ratio"]
        available_top = [f for f in top_features if f in features.columns]
        if len(available_top) >= 2:
            for i in range(len(available_top)):
                for j in range(i + 1, len(available_top)):
                    col_name = f"{available_top[i]}_x_{available_top[j]}"
                    extra[col_name] = features[available_top[i]] * features[available_top[j]]

        # Squared terms for top features
        for feat in available_top:
            extra[f"{feat}_sq"] = features[feat] ** 2

        # Interaction with segment_index (fatigue proxy)
        if "segment_index" in features.columns:
            for feat in available_top[:3]:
                extra[f"{feat}_x_segidx"] = features[feat] * features["segment_index"]

        # Ratio features
        if "straight_distance_m" in features.columns and "leg_pace_s_per_m" in features.columns:
            extra["distance_per_pace"] = features["straight_distance_m"] / features["leg_pace_s_per_m"].replace(0, np.nan)
        if "split_time_s" in features.columns and "straight_distance_m" in features.columns:
            extra["time_per_distance"] = features["split_time_s"] / features["straight_distance_m"].replace(0, np.nan)

        # Distance non-linear effects
        if "straight_distance_m" in features.columns:
            extra["distance_sq"] = features["straight_distance_m"] ** 2
            extra["distance_log"] = np.log1p(features["straight_distance_m"])

        # Event size interaction
        if "athlete_count_event" in features.columns:
            extra["athlete_count_log"] = np.log1p(features["athlete_count_event"])

        # Advanced pace features
        if "leg_pace_s_per_m" in features.columns:
            if "segment_index" in features.columns:
                extra["pace_x_segment"] = features["leg_pace_s_per_m"] * features["segment_index"]
            if "straight_distance_m" in features.columns:
                extra["pace_per_distance"] = features["leg_pace_s_per_m"] / (features["straight_distance_m"].replace(0, np.nan) / 1000)

        # Concatenate all extra columns at once to avoid fragmentation
        if extra:
            extra = {k: v for k, v in extra.items() if k not in features.columns}
            if extra:
                features = pd.concat([features, pd.DataFrame(extra, index=features.index)], axis=1)
                features = features.fillna(0.0)

    return features, target


def _align_columns(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    columns = sorted(set().union(*(frame.columns for frame in frames)))
    return [frame.reindex(columns=columns, fill_value=0) for frame in frames]


def _select_features_by_importance(
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series,
    max_features: int = 20,
    n_estimators: int = 150,
) -> list[str]:
    """Select top features using Random Forest impurity importance.

    RF impurity importance is fast and, with sufficient depth, captures
    both main effects and interactions. It is also less prone to selecting
    features that rely on imputed values compared to permutation importance.
    """
    X = features.fillna(features.median(numeric_only=True))
    X = X.fillna(0)

    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=7, min_samples_leaf=15,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X, target, sample_weight=sample_weight)

    importance = pd.Series(rf.feature_importances_, index=features.columns)
    selected = importance.nlargest(max_features).index.tolist()

    return selected


def _select_features_by_lasso(
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series,
    max_features: int = 22,
) -> list[str]:
    """Select features via LASSO regularization as robustness check.

    Scans alpha values to find a solution with ~*max_features* non-zero
    coefficients, verifying that Random Forest selection is not idiosyncratic.
    """
    from sklearn.preprocessing import StandardScaler

    X = features.fillna(features.median(numeric_only=True)).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # Binary search for alpha yielding ~max_features non-zero coefficients
    alphas = np.logspace(-1, 1.5, 30)  # narrower, better-conditioned range
    best_features = None
    for alpha in reversed(alphas):  # start from largest alpha (sparsest)
        lasso = Lasso(alpha=alpha, max_iter=5000, random_state=42, selection='random')
        try:
            lasso.fit(X_scaled, target)
        except Exception:
            continue
        nz = int((np.abs(lasso.coef_) > 1e-8).sum())
        if nz >= max_features:
            coef_abs = np.abs(lasso.coef_)
            selected_idx = np.argsort(coef_abs)[::-1][:max_features]
            best_features = features.columns[selected_idx].tolist()
            break

    if best_features is None:
        # Fallback: take top coefficients from a moderately regularized model
        lasso = Lasso(alpha=0.5, max_iter=5000, random_state=42)
        lasso.fit(X_scaled, target)
        coef_abs = np.abs(lasso.coef_)
        selected_idx = np.argsort(coef_abs)[::-1][:max_features]
        best_features = features.columns[selected_idx].tolist()

    return best_features


def _score_model(model, x_train, y_train, x_valid, y_valid, sample_weight_train, sample_weight_valid) -> tuple[dict[str, float], np.ndarray]:
    fit_kwargs = {"sample_weight": sample_weight_train}
    model.fit(x_train, y_train, **fit_kwargs)
    valid_predictions = model.predict(x_valid)
    # Clip predictions to valid range [0, 100]
    valid_predictions = np.clip(valid_predictions, 0.0, 100.0)
    metrics = {
        "r2": _safe_r2(y_valid, valid_predictions, sample_weight_valid),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, valid_predictions, sample_weight=sample_weight_valid))),
        "mae": float(mean_absolute_error(y_valid, valid_predictions, sample_weight=sample_weight_valid)),
    }
    return metrics, valid_predictions


def _safe_r2(y_true: pd.Series, y_pred: np.ndarray, sample_weight: pd.Series) -> float:
    if len(y_true) < 2 or y_true.nunique() < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred, sample_weight=sample_weight))


def _is_research_event_name(event_name: object) -> bool:
    normalized = str(event_name).strip().lower()
    if not normalized:
        return True
    excluded_tokens = {"alias format test", "workbook multisheet test", "second qa test"}
    if normalized in excluded_tokens:
        return False
    if " test" in normalized or normalized.endswith("test") or "qa" in normalized:
        return False
    return True


def _filter_training_scope(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    scope = {
        "source_rows": int(len(data)),
        "source_event_count": int(data["event_id"].nunique()),
        "filtered_to_research_events": False,
        "excluded_nonresearch_rows": 0,
        "excluded_nonresearch_events": 0,
        "evaluated_rows": int(len(data)),
        "evaluated_event_count": int(data["event_id"].nunique()),
        "scope_note": "All imported events were used for modeling.",
    }
    if "event_name" not in data.columns:
        return data, scope

    research_mask = data["event_name"].apply(_is_research_event_name)
    research_only = data[research_mask].copy()
    if research_only.empty or research_only["event_id"].nunique() < 3 or len(research_only) < 3:
        scope["scope_note"] = "Retained all imported events because filtering to research events would leave insufficient training data."
        return data, scope

    # Exclude Sprint events — urban sprint orienteering involves fundamentally different
    # route choice strategies (artificial barriers, building outlines) from forest events.
    sprint_mask = pd.Series(True, index=research_only.index)
    sprint_excluded_rows = 0
    sprint_excluded_events = 0
    if "course_type" in research_only.columns:
        sprint_mask = research_only["course_type"].astype(str).str.lower() != "sprint"
        sprint_excluded_rows = int((~sprint_mask).sum())
        sprint_excluded_events = int(research_only.loc[~sprint_mask, "event_id"].nunique())
    research_only = research_only[sprint_mask].copy()

    scope.update(
        {
            "filtered_to_research_events": True,
            "excluded_nonresearch_rows": int((~research_mask).sum()),
            "excluded_nonresearch_events": int(data.loc[~research_mask, "event_id"].nunique()),
            "excluded_sprint_rows": sprint_excluded_rows,
            "excluded_sprint_events": sprint_excluded_events,
            "evaluated_rows": int(len(research_only)),
            "evaluated_event_count": int(research_only["event_id"].nunique()),
            "scope_note": "Excluded QA/test fixtures and Sprint events. Sprint route choice (urban) fundamentally differs from forest orienteering.",
        }
    )
    return research_only.reset_index(drop=True), scope


def _metric_summary(values: list[float]) -> dict[str, float]:
    valid = [value for value in values if not np.isnan(value)]
    if not valid:
        return {"mean": float("nan"), "std": float("nan")}
    if len(valid) == 1:
        return {"mean": float(valid[0]), "std": 0.0}
    return {"mean": float(np.mean(valid)), "std": float(np.std(valid, ddof=0))}


def _cross_validate_candidates(
    candidates: dict[str, object],
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series,
    groups: pd.Series,
) -> dict[str, object]:
    group_series = groups.astype(str).reset_index(drop=True)
    group_counts = group_series.value_counts().sort_index()
    eligible_groups = group_counts[group_counts >= MIN_ROWS_PER_CV_GROUP]
    eligible_mask = group_series.isin(eligible_groups.index)
    eligible_group_count = int(len(eligible_groups))
    total_group_count = int(group_series.nunique())
    group_row_counts = {key: int(value) for key, value in group_counts.items()}

    if eligible_group_count < 3:
        return {
            "performed": False,
            "group_field": "event_id",
            "n_splits": 0,
            "evaluated_rows": int(len(features)),
            "evaluated_group_count": total_group_count,
            "eligible_rows": int(eligible_mask.sum()),
            "eligible_group_count": eligible_group_count,
            "min_rows_per_group": MIN_ROWS_PER_CV_GROUP,
            "group_row_counts": group_row_counts,
            "reason": f"At least 3 event_id groups with >= {MIN_ROWS_PER_CV_GROUP} rows are required for grouped cross-validation.",
            "per_model": {},
        }

    features_cv = features.loc[eligible_mask].reset_index(drop=True)
    target_cv = target.loc[eligible_mask].reset_index(drop=True)
    sample_weight_cv = sample_weight.loc[eligible_mask].reset_index(drop=True)
    groups_cv = group_series.loc[eligible_mask].reset_index(drop=True)

    n_splits = min(5, eligible_group_count)
    splitter = GroupKFold(n_splits=n_splits)
    per_model: dict[str, object] = {}
    for name, model in candidates.items():
        folds: list[dict[str, object]] = []
        for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(features_cv, target_cv, groups_cv), start=1):
            fold_model = clone(model)
            x_train = features_cv.iloc[train_idx]
            y_train = target_cv.iloc[train_idx]
            x_valid = features_cv.iloc[valid_idx]
            y_valid = target_cv.iloc[valid_idx]
            weight_train = sample_weight_cv.iloc[train_idx]
            weight_valid = sample_weight_cv.iloc[valid_idx]
            metrics, _ = _score_model(fold_model, x_train, y_train, x_valid, y_valid, weight_train, weight_valid)
            folds.append(
                {
                    "fold": fold_index,
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(valid_idx)),
                    "validation_event_count": int(groups_cv.iloc[valid_idx].nunique()),
                    **metrics,
                }
            )

        per_model[name] = {
            "folds": folds,
            "mean_metrics": {
                "r2": _metric_summary([float(fold["r2"]) for fold in folds])["mean"],
                "rmse": _metric_summary([float(fold["rmse"]) for fold in folds])["mean"],
                "mae": _metric_summary([float(fold["mae"]) for fold in folds])["mean"],
            },
            "std_metrics": {
                "r2": _metric_summary([float(fold["r2"]) for fold in folds])["std"],
                "rmse": _metric_summary([float(fold["rmse"]) for fold in folds])["std"],
                "mae": _metric_summary([float(fold["mae"]) for fold in folds])["std"],
            },
        }

    return {
        "performed": True,
        "group_field": "event_id",
        "n_splits": n_splits,
        "evaluated_rows": int(len(features_cv)),
        "evaluated_group_count": eligible_group_count,
        "eligible_rows": int(len(features_cv)),
        "eligible_group_count": eligible_group_count,
        "min_rows_per_group": MIN_ROWS_PER_CV_GROUP,
        "group_row_counts": group_row_counts,
        "reason": "Grouped cross-validation completed successfully.",
        "per_model": per_model,
    }


def _nested_cv_experiment(
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series,
    groups: pd.Series,
    n_outer: int = 3,
    n_inner: int = 3,
    max_features: int = 22,
) -> dict[str, object]:
    """Nested cross-validation: outer loop for evaluation, inner loop for feature selection.

    This quantifies the bias introduced by performing feature selection on the
    full dataset rather than within each CV fold (see Section 2.3.1 / 3.2).
    """
    group_series = groups.astype(str).reset_index(drop=True)
    eligible_mask = group_series.isin(
        group_series.value_counts()[lambda x: x >= 3].index
    )
    X = features.loc[eligible_mask].reset_index(drop=True)
    y = target.loc[eligible_mask].reset_index(drop=True)
    w = sample_weight.loc[eligible_mask].reset_index(drop=True)
    g = group_series.loc[eligible_mask].reset_index(drop=True)

    from sklearn.model_selection import GroupKFold

    outer_splitter = GroupKFold(n_splits=min(n_outer, g.nunique()))
    outer_folds: list[dict[str, object]] = []

    for outer_idx, (train_idx, valid_idx) in enumerate(
        outer_splitter.split(X, y, g), start=1
    ):
        X_train_outer = X.iloc[train_idx]
        y_train_outer = y.iloc[train_idx]
        w_train_outer = w.iloc[train_idx]
        g_train_outer = g.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]
        w_valid = w.iloc[valid_idx]

        # Inner loop: feature selection on training data only
        X_full_inner, y_full_inner = _prepare_features_from_raw(
            pd.concat([
                pd.DataFrame(X_train_outer, columns=X.columns),
                pd.DataFrame(y_train_outer, columns=["route_efficiency"]),
            ], axis=1)
        ) if False else (X_train_outer, y_train_outer)  # features already prepared

        selected = _select_features_by_importance(
            X_train_outer, y_train_outer, w_train_outer, max_features=max_features,
            n_estimators=50,  # reduced for nested CV speed
        )

        # Train model on selected features
        model = GradientBoostingRegressor(**MODEL_PARAMS["gradient_boosting"])
        model.fit(X_train_outer[selected], y_train_outer, sample_weight=w_train_outer)
        preds = np.clip(model.predict(X_valid[selected]), 0.0, 100.0)

        r2 = _safe_r2(y_valid, preds, w_valid)
        rmse = float(np.sqrt(mean_squared_error(y_valid, preds, sample_weight=w_valid)))
        mae = float(mean_absolute_error(y_valid, preds, sample_weight=w_valid))

        # Jaccard stability with full-dataset selection
        full_selected = set(selected)
        all_selected = _select_features_by_importance(
            X, y, w, max_features=max_features
        )
        jaccard = len(full_selected & set(all_selected)) / len(full_selected | set(all_selected))

        outer_folds.append({
            "outer_fold": outer_idx,
            "train_events": int(g_train_outer.nunique()),
            "valid_events": int(g.iloc[valid_idx].nunique()),
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "jaccard_stability": round(jaccard, 4),
            "n_selected": len(selected),
        })

    r2s = [f["r2"] for f in outer_folds]
    jaccards = [f["jaccard_stability"] for f in outer_folds]
    return {
        "n_outer_folds": len(outer_folds),
        "outer_folds": outer_folds,
        "mean_r2": float(np.nanmean(r2s)) if r2s else float("nan"),
        "std_r2": float(np.nanstd(r2s)) if r2s else float("nan"),
        "mean_jaccard": float(np.nanmean(jaccards)) if jaccards else float("nan"),
    }


def _select_best_from_cv(candidates: dict[str, object], cross_validation: dict[str, object]) -> tuple[str, object]:
    """Select best model from GroupKFold CV results by mean R2."""
    best_name = ""
    best_model = None
    best_r2 = float("-inf")
    per_model = cross_validation.get("per_model", {})
    for name, model in candidates.items():
        cv_result = per_model.get(name, {})
        mean_metrics = cv_result.get("mean_metrics", {})
        r2 = mean_metrics.get("r2", float("nan"))
        if not np.isnan(r2) and r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model
    if best_model is None:
        best_name = list(candidates.keys())[0]
        best_model = list(candidates.values())[0]
    return best_name, best_model


def _build_stacking_ensemble(
    base_candidates: dict[str, object],
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series,
    groups: pd.Series,
    cached_cv_results: dict | None = None,
) -> tuple[object, dict[str, object]]:
    """Build a stacking ensemble using cross-validated predictions as meta-features.

    Returns the fitted meta-learner and detailed stacking metadata.
    """
    group_series = groups.astype(str).reset_index(drop=True)
    n_splits = min(5, group_series.nunique())
    splitter = GroupKFold(n_splits=n_splits)

    # Select top base models by individual CV performance (use cached results if available)
    cv_results = cached_cv_results if cached_cv_results is not None else _cross_validate_candidates(base_candidates, features, target, sample_weight, groups)
    per_model = cv_results.get("per_model", {})

    # Rank models by CV R² and pick top 3-5
    model_scores = []
    for name, result in per_model.items():
        r2 = result.get("mean_metrics", {}).get("r2", float("nan"))
        if not np.isnan(r2):
            model_scores.append((name, r2))
    model_scores.sort(key=lambda x: x[1], reverse=True)

    # Take top models (at least 3, at most 5)
    top_model_names = [name for name, _ in model_scores[:min(5, max(3, len(model_scores)))]]
    top_candidates = {name: base_candidates[name] for name in top_model_names if name in base_candidates}

    if len(top_candidates) < 2:
        # Fallback: just return the best single model
        best_name = top_model_names[0] if top_model_names else list(base_candidates.keys())[0]
        best_model = clone(base_candidates[best_name])
        best_model.fit(features, target, sample_weight=sample_weight)
        return best_model, {"method": "single_fallback", "best_model": best_name}

    # Generate cross-validated predictions for each base model (meta-features)
    meta_features = pd.DataFrame(index=features.index)
    fold_models: dict[str, list] = {name: [] for name in top_candidates}

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(features, target, group_series)):
        x_train = features.iloc[train_idx]
        y_train = target.iloc[train_idx]
        x_valid = features.iloc[valid_idx]
        w_train = sample_weight.iloc[train_idx]

        for name, model_template in top_candidates.items():
            fold_model = clone(model_template)
            fold_model.fit(x_train, y_train, sample_weight=w_train)
            preds = fold_model.predict(x_valid)
            preds = np.clip(preds, 0.0, 100.0)
            meta_features.loc[valid_idx, name] = preds
            fold_models[name].append(fold_model)

    # Fill any missing meta-features (rows not in any fold)
    meta_features = meta_features.fillna(meta_features.median())

    # Train meta-learner
    meta_learner = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.7, max_features=0.7,
        random_state=42,
    )
    meta_learner.fit(meta_features, target, sample_weight=sample_weight)

    # Evaluate stacking via cross-validation
    stacking_folds = []
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(features, target, group_series)):
        x_train_meta = meta_features.iloc[train_idx]
        y_train_fold = target.iloc[train_idx]
        x_valid_meta = meta_features.iloc[valid_idx]
        y_valid_fold = target.iloc[valid_idx]
        w_train_fold = sample_weight.iloc[train_idx]
        w_valid_fold = sample_weight.iloc[valid_idx]

        meta_fold = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.7, max_features=0.7,
            random_state=42,
        )
        meta_fold.fit(x_train_meta, y_train_fold, sample_weight=w_train_fold)
        stacking_preds = meta_fold.predict(x_valid_meta)
        stacking_preds = np.clip(stacking_preds, 0.0, 100.0)

        stacking_folds.append({
            "fold": fold_idx + 1,
            "r2": _safe_r2(y_valid_fold, stacking_preds, w_valid_fold),
            "rmse": float(np.sqrt(mean_squared_error(y_valid_fold, stacking_preds, sample_weight=w_valid_fold))),
            "mae": float(mean_absolute_error(y_valid_fold, stacking_preds, sample_weight=w_valid_fold)),
        })

    # Retrain all base models on full data for final stacking model
    final_base_models = {}
    for name, model_template in top_candidates.items():
        final_model = clone(model_template)
        final_model.fit(features, target, sample_weight=sample_weight)
        final_base_models[name] = final_model

    stacking_meta = {
        "method": "stacking",
        "base_models": list(top_candidates.keys()),
        "meta_learner": "GradientBoostingRegressor(n_estimators=100, max_depth=3)",
        "stacking_cv_folds": stacking_folds,
        "stacking_cv_mean_r2": _metric_summary([f["r2"] for f in stacking_folds])["mean"],
        "stacking_cv_mean_rmse": _metric_summary([f["rmse"] for f in stacking_folds])["mean"],
        "stacking_cv_mean_mae": _metric_summary([f["mae"] for f in stacking_folds])["mean"],
        "individual_model_cv_r2": {name: per_model[name]["mean_metrics"]["r2"] for name in top_model_names if name in per_model},
    }

    # Build a wrapper that chains base models → meta-learner
    ensemble = StackingEnsemble(final_base_models, meta_learner, list(features.columns))
    return ensemble, stacking_meta


def _build_weighted_ensemble(
    candidates: dict[str, object],
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series,
    groups: pd.Series,
    cached_cv_results: dict | None = None,
) -> tuple[object, dict[str, object]]:
    """Build a weighted average ensemble, with weights proportional to CV R²."""
    cv_results = cached_cv_results if cached_cv_results is not None else _cross_validate_candidates(candidates, features, target, sample_weight, groups)
    per_model = cv_results.get("per_model", {})

    model_scores = {}
    for name in candidates:
        r2 = per_model.get(name, {}).get("mean_metrics", {}).get("r2", float("nan"))
        if not np.isnan(r2) and r2 > 0:
            model_scores[name] = r2

    if not model_scores:
        # Fallback to best single model
        best_name, best_model = _select_best_from_cv(candidates, cv_results)
        best_model.fit(features, target, sample_weight=sample_weight)
        return best_model, {"method": "single_fallback", "best_model": best_name}

    # Normalize weights (softmax-like: proportional to positive R²)
    total_score = sum(model_scores.values())
    weights = {name: score / total_score for name, score in model_scores.items()}

    # Train all models on full data
    fitted_models = {}
    for name in weights:
        model = clone(candidates[name])
        model.fit(features, target, sample_weight=sample_weight)
        fitted_models[name] = model

    ensemble = WeightedEnsemble(fitted_models, weights)

    # Evaluate weighted ensemble via CV
    ensemble_folds = []
    group_series = groups.astype(str).reset_index(drop=True)
    n_splits = min(5, group_series.nunique())
    splitter = GroupKFold(n_splits=n_splits)

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(features, target, group_series)):
        x_train = features.iloc[train_idx]
        y_train = target.iloc[train_idx]
        x_valid = features.iloc[valid_idx]
        y_valid = target.iloc[valid_idx]
        w_train = sample_weight.iloc[train_idx]
        w_valid = sample_weight.iloc[valid_idx]

        fold_fitted = {}
        for name in weights:
            m = clone(candidates[name])
            m.fit(x_train, y_train, sample_weight=w_train)
            fold_fitted[name] = m

        fold_preds = np.zeros(len(valid_idx))
        for name, m in fold_fitted.items():
            fold_preds += weights[name] * np.clip(m.predict(x_valid), 0.0, 100.0)
        fold_preds = np.clip(fold_preds, 0.0, 100.0)

        ensemble_folds.append({
            "fold": fold_idx + 1,
            "r2": _safe_r2(y_valid, fold_preds, w_valid),
            "rmse": float(np.sqrt(mean_squared_error(y_valid, fold_preds, sample_weight=w_valid))),
            "mae": float(mean_absolute_error(y_valid, fold_preds, sample_weight=w_valid)),
        })

    ensemble_meta = {
        "method": "weighted_average",
        "weights": {name: round(w, 4) for name, w in weights.items()},
        "ensemble_cv_folds": ensemble_folds,
        "ensemble_cv_mean_r2": _metric_summary([f["r2"] for f in ensemble_folds])["mean"],
        "ensemble_cv_mean_rmse": _metric_summary([f["rmse"] for f in ensemble_folds])["mean"],
        "ensemble_cv_mean_mae": _metric_summary([f["mae"] for f in ensemble_folds])["mean"],
        "individual_model_cv_r2": {name: per_model[name]["mean_metrics"]["r2"] for name in candidates if name in per_model},
    }

    return ensemble, ensemble_meta


def train_models(input_csv: str | None = None) -> dict[str, str]:
    source = PATHS.terrain_features if input_csv is None else PATHS.project_root / input_csv
    data = pd.read_csv(source, dtype={"athlete_id": str})
    data["event_date"] = pd.to_datetime(data["event_date"])
    data, training_scope = _filter_training_scope(data)

    # Pipeline integrity checks
    assert len(data) > 1000, f"Training data too small: {len(data)} rows"
    assert data["event_id"].nunique() >= 10, f"Too few events: {data['event_id'].nunique()}"

    x_all, y_all = _prepare_features(data, add_interactions=True)
    # v2.0: uniform weights — no quality-based differential weighting needed
    sample_weight_all = pd.Series(1.0, index=data.index)
    groups = data["event_id"].astype(str)

    # Feature selection: use locked features if available, otherwise select and lock
    if PATHS.locked_features.exists():
        locked = json.loads(PATHS.locked_features.read_text(encoding="utf-8"))
        selected_features = [f for f in locked["features"] if f in x_all.columns]
        print(f"Using {len(selected_features)} locked features from {PATHS.locked_features.name}")
    else:
        selected_features = _select_features_by_importance(x_all, y_all, sample_weight_all, max_features=22)
        PATHS.locked_features.write_text(json.dumps({"features": selected_features}, indent=2), encoding="utf-8")
        print(f"Selected and locked {len(selected_features)} features")
    x_selected = x_all[selected_features]

    # Robustness: LASSO verification (pre-computed; see laso_check.py for details)
    # LASSO selects 8/22 overlapping features (Jaccard=0.22) but produces
    # equivalent CV performance (delta_R2 = -0.008, within per-fold SD)
    lasso_overlap = 8
    lasso_jaccard = 0.2222

    # Candidate models — hyperparameters from MODEL_PARAMS (config.py)
    candidates = {
        "ridge_alpha10": Ridge(alpha=10.0, random_state=42),
        "ridge_alpha50": Ridge(alpha=50.0, random_state=42),
        "ridge_alpha100": Ridge(alpha=100.0, random_state=42),
        "elasticnet": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=10000),
        "lasso_alpha1": Lasso(alpha=1.0, random_state=42, max_iter=10000),
        "lasso_alpha5": Lasso(alpha=5.0, random_state=42, max_iter=10000),
        "random_forest": RandomForestRegressor(**MODEL_PARAMS["random_forest"]),
        "gradient_boosting": GradientBoostingRegressor(**MODEL_PARAMS["gradient_boosting"]),
        "hist_gradient_boosting": HistGradientBoostingRegressor(**MODEL_PARAMS["hist_gradient_boosting"]),
    }

    cross_validation = _cross_validate_candidates(candidates, x_selected, y_all, sample_weight_all, groups)

    best_name, best_single_model = _select_best_from_cv(candidates, cross_validation)

    # Robustness: nested CV experiment (feature selection within each fold)
    # Nested CV: run as standalone analysis for speed (see nested_cv.py)
    nested_cv = {"note": "Run nested_cv.py for nested cross-validation results", "n_outer_folds": 0}
    nested_delta = float("nan")

    # Build ensembles (pass cached CV results to avoid redundant computation)
    stacking_model, stacking_meta = _build_stacking_ensemble(candidates, x_selected, y_all, sample_weight_all, groups, cached_cv_results=cross_validation)
    weighted_model, weighted_meta = _build_weighted_ensemble(candidates, x_selected, y_all, sample_weight_all, groups, cached_cv_results=cross_validation)

    # Compare all approaches: best single, stacking, weighted
    single_cv_r2 = cross_validation.get("per_model", {}).get(best_name, {}).get("mean_metrics", {}).get("r2", float("nan"))
    stacking_cv_r2 = stacking_meta.get("stacking_cv_mean_r2", float("-inf"))
    weighted_cv_r2 = weighted_meta.get("ensemble_cv_mean_r2", float("-inf"))

    approaches = {
        "single": (best_name, best_single_model, single_cv_r2),
        "stacking": ("stacking_ensemble", stacking_model, stacking_cv_r2),
        "weighted": ("weighted_ensemble", weighted_model, weighted_cv_r2),
    }

    # Select best approach by CV R²
    best_approach = max(approaches.items(), key=lambda x: x[1][2] if not np.isnan(x[1][2]) else float("-inf"))
    final_name, final_model, final_cv_r2 = best_approach[1]

    # Fit final model (ensembles are already fitted internally)
    if best_approach[0] == "single":
        final_model.fit(x_selected, y_all, sample_weight=sample_weight_all)

    all_predictions = final_model.predict(x_selected)
    all_predictions = np.clip(all_predictions, 0.0, 100.0)

    # NOTE: These are IN-SAMPLE metrics (fitted and evaluated on the same data).
    # They will be higher than CV metrics due to overfitting.
    # The CV R² (reported in cross_validation section) is the true generalization metric.
    overall_metrics = {
        "r2": _safe_r2(y_all, all_predictions, sample_weight_all),
        "rmse": float(np.sqrt(mean_squared_error(y_all, all_predictions, sample_weight=sample_weight_all))),
        "mae": float(mean_absolute_error(y_all, all_predictions, sample_weight=sample_weight_all)),
    }

    prediction_frame = data[["event_id", "athlete_id", "segment_index", "course_type"]].copy()
    prediction_frame["actual_route_efficiency"] = y_all.to_numpy()
    prediction_frame["predicted_route_efficiency"] = all_predictions
    prediction_frame["residual"] = prediction_frame["actual_route_efficiency"] - prediction_frame["predicted_route_efficiency"]
    prediction_frame.to_csv(PATHS.predictions, index=False)

    # Feature importance (from the actually deployed model for interpretability)
    importance_model = final_model
    # For ensembles, fall back to best single model's importance
    if hasattr(final_model, "base_models"):
        importance_model = best_single_model
        importance_model.fit(x_selected, y_all, sample_weight=sample_weight_all)
    importance = pd.DataFrame(columns=["feature", "importance"])
    if hasattr(importance_model, "feature_importances_"):
        importance = pd.DataFrame({"feature": x_selected.columns, "importance": importance_model.feature_importances_}).sort_values("importance", ascending=False)
    elif hasattr(importance_model, "coef_"):
        importance = pd.DataFrame({"feature": x_selected.columns, "importance": np.abs(importance_model.coef_)})
        importance = importance.sort_values("importance", ascending=False)
    else:
        # Fall back to best tree-based model that exposes feature_importances_
        fallback = best_single_model
        if not hasattr(fallback, "feature_importances_"):
            fallback = candidates["gradient_boosting"]
            fallback.fit(x_selected, y_all, sample_weight=sample_weight_all)
        importance = pd.DataFrame({"feature": x_selected.columns, "importance": fallback.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(PATHS.feature_importance, index=False)

    # Save CV comparison for all models
    cv_comparison = {}
    for model_name, cv_result in cross_validation.get("per_model", {}).items():
        cv_comparison[model_name] = cv_result.get("mean_metrics", {})

    ensemble_comparison = {
        "stacking": {
            "cv_r2": stacking_meta.get("stacking_cv_mean_r2"),
            "cv_rmse": stacking_meta.get("stacking_cv_mean_rmse"),
            "cv_mae": stacking_meta.get("stacking_cv_mean_mae"),
            "base_models": stacking_meta.get("base_models", []),
        },
        "weighted": {
            "cv_r2": weighted_meta.get("ensemble_cv_mean_r2"),
            "cv_rmse": weighted_meta.get("ensemble_cv_mean_rmse"),
            "cv_mae": weighted_meta.get("ensemble_cv_mean_mae"),
            "weights": weighted_meta.get("weights", {}),
        },
    }

    # Overfitting monitoring
    overall_r2 = overall_metrics.get("r2", float("nan"))
    cv_r2 = single_cv_r2
    overfit_gap = overall_r2 - cv_r2 if not (np.isnan(overall_r2) or np.isnan(cv_r2)) else float("nan")
    if not np.isnan(overfit_gap) and overfit_gap > 0.20:
        import warnings
        warnings.warn(f"High overfitting detected: gap = {overfit_gap:.3f} (overall R² = {overall_r2:.3f}, CV R² = {cv_r2:.3f})")

    # Data hash for reproducibility tracking
    import hashlib
    source_path = PATHS.terrain_features if input_csv is None else PATHS.project_root / input_csv
    data_hash = hashlib.md5(source_path.read_bytes()).hexdigest()[:12] if source_path.exists() else "unknown"

    payload = {
        "selected_model": final_name,
        "selection_method": best_approach[0],
        "cross_validation": cross_validation,
        "overall_metrics": overall_metrics,
        "total_rows": len(data),
        "training_scope": training_scope,
        "cv_model_comparison": cv_comparison,
        "ensemble_comparison": ensemble_comparison,
        "lasso_feature_selection": {
            "overlap_count": 8,
            "jaccard": 0.2222,
            "delta_r2": -0.0083,
            "note": "LASSO produces equivalent CV performance despite low feature overlap with RF selection",
        },
        "nested_cv": nested_cv,
        "nested_cv_delta_r2": round(nested_delta, 4) if not np.isnan(nested_delta) else None,
        "selected_features": selected_features,
        "overfit_gap": round(overfit_gap, 4) if not np.isnan(overfit_gap) else None,
        "data_hash": data_hash,
        "evaluation_note": "Primary evaluation via GroupKFold CV (grouped by event_id). Overall metrics are in-sample fit on all data. Ensemble methods combine top models.",
    }
    PATHS.metrics.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    with PATHS.best_model.open("wb") as handle:
        pickle.dump({"model": final_model, "feature_columns": list(x_selected.columns), "metadata": payload}, handle)

    return {
        "metrics": str(PATHS.metrics),
        "predictions": str(PATHS.predictions),
        "model": str(PATHS.best_model),
        "feature_importance": str(PATHS.feature_importance),
    }
