# Code: Event-Level Standardization Enables Cross-Event Route Choice Efficiency Prediction in Elite Orienteering

## Structure

```
code/
├── features.py          # Feature engineering (raw data → terrain_features.csv)
├── modeling.py          # Model training, CV, feature selection
├── sensitivity.py       # Ablation, LOEO, permutation test
├── config.py            # Paths and hyperparameters
├── cli.py               # Command-line interface
├── generate_figures.py  # Generate all 7 publication figures
├── permutation_test.py  # Standalone permutation test (n=500)
├── __init__.py
├── __main__.py
├── requirements.txt
└── README.md
```

## Reproduce All Results

```bash
pip install -r requirements.txt

# Step 1: Generate features
python -c "from features import calculate_terrain_features; calculate_terrain_features('path/to/dataset_public.csv')"

# Step 2: Train model
python -c "from modeling import train_models; train_models()"

# Step 3: Sensitivity analysis
python -c "from sensitivity import run_sensitivity_analysis; run_sensitivity_analysis()"

# Step 4: Permutation test (500 iterations)
python permutation_test.py --n 500

# Step 5: Generate figures
python generate_figures.py
```

## Key Design Decisions

- **No XGBoost/LightGBM dependency**: uses scikit-learn's GradientBoosting and RandomForest
- **44 candidate features** generated from 7 raw columns, **11 selected** via Random Forest impurity importance
- **GroupKFold by event_id**: ensures no event appears in both train and validation folds

## Environment

- Python 3.13
- scikit-learn 1.8.0
- NumPy, pandas, matplotlib, scipy (see requirements.txt)
