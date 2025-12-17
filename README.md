# DSAP_intercantonalmigration_dynamics
Inter-cantonal migration dynamics and their sensitivity to housing and mortgage market conditions in Switzerland.

## Quick start
- Install dependencies from `requirements.txt`.
- Run any model via the unified entry point: `python main.py --model {ols|ridge|randomforest|gradientboosting|decisiontree|all}` (uses `data/databasecsv.csv` by default; override with `--data /path/to/file.csv`).
