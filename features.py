"""Feature engineering pipeline.

Reads the 7-column raw dataset and computes all derived features used by the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PATHS


def calculate_terrain_features(input_csv: str | None = None) -> dict[str, str]:
    """Compute all features from raw data and save to terrain_features.csv.

    Raw columns (7): event_id, course_type, athlete_id, segment_index,
                     split_time_s, straight_distance_m, actual_distance_m
    """
    source = PATHS.validated_dataset if input_csv is None else PATHS.project_root / input_csv
    data = pd.read_csv(source, dtype={"athlete_id": str})

    # ── Derived base columns ──────────────────────────────────
    data["route_efficiency"] = np.round(
        data["straight_distance_m"] / data["actual_distance_m"] * 100.0, 4
    )
    data["leg_pace_s_per_m"] = (
        data["split_time_s"] / data["straight_distance_m"].replace(0, np.nan)
    )
    data["speed_kmh"] = (
        data["actual_distance_m"] / data["split_time_s"].replace(0, np.nan) * 3.6
    )
    data["athlete_count_event"] = data.groupby("event_id")["athlete_id"].transform("nunique")

    data = data.sort_values(["event_id", "athlete_id", "segment_index"]).reset_index(drop=True)

    # ── Target-derived (analysis only, NOT used as features) ──
    data["athlete_mean_re"] = data.groupby("athlete_id")["route_efficiency"].transform("mean")
    data["event_mean_re"] = data.groupby("event_id")["route_efficiency"].transform("mean")
    data["course_type_mean_re"] = data.groupby("course_type")["route_efficiency"].transform("mean")
    data["athlete_re_std"] = data.groupby("athlete_id")["route_efficiency"].transform("std").fillna(0)
    data["event_re_std"] = data.groupby("event_id")["route_efficiency"].transform("std").fillna(0)
    data["is_high_re"] = (data["route_efficiency"] > 90).astype(int)
    data["is_low_re"] = (data["route_efficiency"] < 50).astype(int)

    # ── Behavioural features ──────────────────────────────────
    data["distance_pace_ratio"] = (
        data["straight_distance_m"] / data["leg_pace_s_per_m"].replace(0, np.nan)
    )
    data["fatigue_adjusted_pace"] = data["leg_pace_s_per_m"] * (1 + data["segment_index"] * 0.01)

    athlete_mean_pace = data.groupby("athlete_id")["leg_pace_s_per_m"].transform("mean")
    data["pace_deviation"] = data["leg_pace_s_per_m"] - athlete_mean_pace

    data["distance_x_pace"] = data["straight_distance_m"] * data["leg_pace_s_per_m"]

    # ── Event-level z-scores ──────────────────────────────────
    for col in ["leg_pace_s_per_m", "straight_distance_m", "split_time_s"]:
        event_mean = data.groupby("event_id")[col].transform("mean")
        event_std = data.groupby("event_id")[col].transform("std").replace(0, np.nan)
        data[f"{col}_event_zscore"] = ((data[col] - event_mean) / event_std).clip(-5, 5)

    # ── Robust z-scores (median/MAD) ──────────────────────────
    for col in ["leg_pace_s_per_m", "split_time_s"]:
        event_median = data.groupby("event_id")[col].transform("median")
        event_mad = data.groupby("event_id")[col].transform(
            lambda x: (x - x.median()).abs().median()
        ).replace(0, np.nan)
        data[f"{col}_event_robust_zscore"] = ((data[col] - event_median) / event_mad).clip(-5, 5)

    # ── Event-level statistics ────────────────────────────────
    data["event_pace_std"] = data.groupby("event_id")["leg_pace_s_per_m"].transform("std")
    data["event_distance_mean"] = data.groupby("event_id")["straight_distance_m"].transform("mean")

    # ── Athlete ranking ───────────────────────────────────────
    data["athlete_pace_rank_event"] = data.groupby(
        ["event_id", "segment_index"]
    )["leg_pace_s_per_m"].transform(lambda x: x.rank(pct=True))

    data["pace_deviation_x_athlete_pace_rank_event"] = (
        data["pace_deviation"] * data["athlete_pace_rank_event"]
    )

    athlete_pace_std = data.groupby("athlete_id")["leg_pace_s_per_m"].transform("std")
    event_avg_athlete_std = data.groupby("event_id")["leg_pace_s_per_m"].transform(
        lambda x: x.groupby(data["athlete_id"]).std().mean()
    )
    data["pace_consistency_ratio"] = (
        athlete_pace_std / event_avg_athlete_std.replace(0, np.nan)
    ).clip(0, 10)

    def _safe_pct(series):
        denom = series.max() - series.min()
        if denom == 0:
            return pd.Series(0.5, index=series.index)
        return (series - series.min()) / denom

    data["segment_position_pct"] = data.groupby("event_id")["segment_index"].transform(_safe_pct)

    # ── Integrity checks ──────────────────────────────────────
    for col in ["leg_pace_s_per_m", "pace_deviation", "split_time_s_event_zscore"]:
        assert data[col].isna().mean() < 0.5, f"Critical feature '{col}' has >50% NaN"

    # ── Save ──────────────────────────────────────────────────
    data.to_csv(PATHS.terrain_features, index=False)

    summary = (
        data.groupby(["course_type"])
        .agg(
            samples=("route_efficiency", "size"),
            mean_efficiency=("route_efficiency", "mean"),
            mean_speed=("speed_kmh", "mean"),
            mean_distance=("straight_distance_m", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(PATHS.terrain_summary, index=False)

    return {
        "terrain_features": str(PATHS.terrain_features),
        "terrain_summary": str(PATHS.terrain_summary),
    }
