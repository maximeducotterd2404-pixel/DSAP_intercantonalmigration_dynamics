# Inter-Cantonal Migration Dynamics

## Overview
This project studies inter-cantonal migration dynamics in Switzerland and evaluates multiple predictive models using a time-based split. It includes linear baselines (OLS, Ridge), tree-based models (Random Forest, Gradient Boosting), a Decision Tree classifier, and K-means clustering for canton profiles.

## Setup
- Python 3.10+ recommended (project tested with Python 3.13).
- From the project root (e.g., `cd /path/to/DSAP_intercantonal_dynamics`).
- Install dependencies:
  - `pip install -r requirements.txt`
  - or `conda env create -f environment.yml`
- If using conda, activate the environment: `conda activate dsap_intercantonal_dynamics`.

## Running the project (main.py)
Entry point: `python main.py --model {ols|ridge|randomforest|gradientboosting|decisiontree|kmeans|all} [--data PATH] [--non-interactive]`

- Data defaults to `data/databasecsv.csv`.
- Override with `--data /path/to/file.csv`.

Examples:
- `python main.py --model all`
- `python main.py --model ridge --data /absolute/path/to/databasecsv.csv`
- `python main.py --model kmeans --non-interactive`

Console outputs:
- OLS: Train/Test R² + Test RMSE
- Ridge: Train/Val/Test R² + Test MSE (alpha selected on validation)
- RF/GB: Train/Test R² + Test RMSE + feature importances
- Decision Tree: accuracy (direction of migration change)
- KMeans: inertia + cluster profiles + radar plot (saved in `results/`)

Detailed results with coefficients: run the model-specific prediction scripts from the project root. 
- `python src/ML/ols/ols.py` 
- `python src/ML/ridge/ridge.py` 
- `python src/ML/ridge/randomforest.py`
- `python src/ML/ridge/gradientboosting.py`

## Data
- Cleaned dataset: `data/databasecsv.csv`
- Raw sources (OFS/BFS, SNB) used to build the dataset are included under `data/raw/` for reference.

## Tests
- Run: `pytest`
- Coverage: use `coverage run -m pytest` then `coverage report -m` (if installed)

## Project structure
```
.
├── README.md
├── PROPOSAL.md
├── AI_USAGE.md
├── environment.yml
├── requirements.txt
├── main.py
├── data/
│   ├── databasecsv.csv
│   └── raw/
├── results/
├── notebooks/
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── models.py
    ├── evaluation.py
    └── ML/
```

## Reproducibility
- Fixed random seeds where applicable (e.g., `random_state=0`).
- Time-based train/test split to avoid temporal leakage.
