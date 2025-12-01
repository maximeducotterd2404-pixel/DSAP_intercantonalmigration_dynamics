import pandas as pd
import numpy as np
from src.ML.decisiontree import (
    prepare_dataset, train_decision_tree, evaluate_model
)

def test_decision_tree_pipeline():
    # synthetic dataset
    df = pd.DataFrame({
        "canton": ["A", "A", "A", "B", "B", "B"],
        "year": [2020, 2021, 2022, 2020, 2021, 2022],
        "migration_rate": [0.1, 0.2, 0.15, -0.1, -0.15, -0.2],
        "log_rent_avg": [1,2,3,1,2,3],
        "log_avg_income": [5,6,7,4,5,6],
        "log_unemployment": [0.4,0.3,0.5,0.8,0.9,1.0],
        "log_schockexposure": [0.1,0.2,0.3,0.5,0.4,0.3],
        "housing_construction_pc": [10,20,30,5,15,25],
        "CLUSTER0": [1,1,1,0,0,0],
        "CLUSTER1": [0,0,0,1,1,1],
        "CLUSTER2": [0,0,0,0,0,0],
    })

    X_train, y_train, X_test, y_test = prepare_dataset(df)
    model = train_decision_tree(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)

    assert 0 <= acc <= 1
