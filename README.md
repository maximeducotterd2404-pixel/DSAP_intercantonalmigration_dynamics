# Inter-Cantonal Migration Dynamics

## Overview
This project studies inter-cantonal migration dynamics in Switzerland and evaluates multiple predictive models using a time-based split. It includes linear baselines (OLS, Ridge), tree-based models (Random Forest, Gradient Boosting), a Decision Tree classifier, and K-means clustering for canton profiles.

## Setup
- Python 3.10+ recommended (project tested with Python 3.13).
- Install dependencies:
  - `pip install -r requirements.txt`
  - or `conda env create -f environment.yml`

## Running the project (main.py)
Entry point: `python main.py --model {ols|ridge|randomforest|gradientboosting|decisiontree|all} [--data PATH]`

- Data defaults to `data/databasecsv.csv`.
- Override with `--data /path/to/file.csv`.

Examples:
- `python main.py --model all`
- `python main.py --model ridge --data /absolute/path/to/databasecsv.csv`

Console outputs:
- OLS: Train/Test R² + Test RMSE
- Ridge: Train/Val/Test R² + Test MSE (alpha selected on validation)
- RF/GB: Train/Test R² + Test RMSE + feature importances
- Decision Tree: accuracy (direction of migration change)

## Data
- Cleaned dataset: `data/databasecsv.csv`
- Raw sources (OFS/BFS, SNB) are not included; place any raw files under `data/raw/` if you want to rebuild the dataset.

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
│   ├── README.md
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
