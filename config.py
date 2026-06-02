"""Project paths and constants — v2.0."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "06_Code_Repository"


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path = PROJECT_ROOT
    code_root: Path = CODE_ROOT
    project_admin: Path = PROJECT_ROOT / "00_Project_Administration"
    data_acquisition: Path = PROJECT_ROOT / "02_Data_Acquisition"
    data_validation: Path = PROJECT_ROOT / "03_Data_Validation"
    validated_dataset: Path = PROJECT_ROOT / "03_Data_Validation" / "dataset_public.csv"
    model_training: Path = PROJECT_ROOT / "04_Model_Training"
    paper_preparation: Path = PROJECT_ROOT / "05_Paper_Preparation"
    figures: Path = PROJECT_ROOT / "05_Paper_Preparation" / "Figures"
    model_reports: Path = PROJECT_ROOT / "04_Model_Training" / "Reports"
    model_store: Path = PROJECT_ROOT / "04_Model_Training" / "Models"
    # Output paths
    terrain_features: Path = PROJECT_ROOT / "04_Model_Training" / "Reports" / "terrain_features.csv"
    terrain_summary: Path = PROJECT_ROOT / "04_Model_Training" / "Reports" / "terrain_feature_summary.csv"
    predictions: Path = PROJECT_ROOT / "04_Model_Training" / "Reports" / "model_predictions.csv"
    metrics: Path = PROJECT_ROOT / "04_Model_Training" / "Reports" / "model_metrics.json"
    feature_importance: Path = PROJECT_ROOT / "04_Model_Training" / "Reports" / "feature_importance.csv"
    best_model: Path = PROJECT_ROOT / "04_Model_Training" / "Models" / "best_model.pkl"
    locked_features: Path = PROJECT_ROOT / "04_Model_Training" / "Reports" / "locked_features.json"
    reports: Path = PROJECT_ROOT / "04_Model_Training" / "Reports"


PATHS = ProjectPaths()

# Hyperparameters from Optuna optimization
MODEL_PARAMS: dict[str, dict] = {
    "gradient_boosting": {
        "n_estimators": 345, "max_depth": 4, "learning_rate": 0.01408,
        "min_samples_leaf": 80, "min_samples_split": 86,
        "subsample": 0.502, "max_features": 0.419,
        "validation_fraction": 0.143, "n_iter_no_change": 17,
        "tol": 0.000447, "ccp_alpha": 0.0267, "random_state": 42,
    },
    "hist_gradient_boosting": {
        "max_iter": 400, "max_depth": 5, "learning_rate": 0.025,
        "min_samples_leaf": 25, "max_leaf_nodes": 60,
        "l2_regularization": 0.5, "early_stopping": True,
        "validation_fraction": 0.1, "n_iter_no_change": 15, "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 350, "max_depth": 6,
        "min_samples_leaf": 28, "min_samples_split": 30,
        "max_features": 0.3, "random_state": 42, "ccp_alpha": 0.0001,
    },
}


QUALITY_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0}  # v2.0: uniform — all events same quality tier


def ensure_directories() -> None:
    for d in [PATHS.model_reports, PATHS.model_store, PATHS.figures]:
        d.mkdir(parents=True, exist_ok=True)
