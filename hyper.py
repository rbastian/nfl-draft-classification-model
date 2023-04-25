import numpy as np
import pandas as pd

from hyperopt import fmin, tpe, hp, Trials

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


# Load the dataset
csv_files = [
    "./data/2022QBCSV.csv",
    "./data/2021QBCSV.csv",
    "./data/2020QBCSV.csv",
    "./data/2019QBCSV.csv",
    "./data/2018QBCSV.csv",
    "./data/2017QBCSV.csv",
]
data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Create test and training splits
X = data.drop(
    ["NAME", "SCHOOL", "YEAR", "10-YD", "ARMS", "HANDS", "WING", "DRAFTED", "ROUND"],
    axis=1,
)
y = data["DRAFTED"]

# Define the objective function
def objective(params):

    model = LogisticRegression(
        solver=params["solver"],
        penalty=params["penalty"],
        C=params["C"],
        l1_ratio=params.get(
            "l1_ratio", None
        ),  # Include l1_ratio if it exists in params
        max_iter=50000,
        random_state=42,
    )

    # Cross-validation using stratified K-fold
    cv = StratifiedKFold(n_splits=5)
    score = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")
    return -np.mean(score)


# Define the hyperparameter search space
space = hp.choice(
    "model",
    [
        {
            "type": "logistic_regression",
            "solver": hp.choice(
                "solver_no_penalty", ["newton-cg", "lbfgs", "sag", "saga"]
            ),
            "penalty": "none",
            "C": hp.loguniform("C_no_penalty", -5, 2),
        },
        {
            "type": "logistic_regression",
            "solver": hp.choice("solver_l2", ["newton-cg", "lbfgs", "sag", "saga"]),
            "penalty": "l2",
            "C": hp.loguniform("C_l2", -5, 2),
        },
        {
            "type": "logistic_regression",
            "solver": hp.choice("solver_l1", ["liblinear", "saga"]),
            "penalty": "l1",
            "C": hp.loguniform("C_l1", -5, 2),
        },
        {
            "type": "logistic_regression",
            "solver": "saga",
            "penalty": "elasticnet",
            "C": hp.loguniform("C_elasticnet", -5, 2),
            "l1_ratio": hp.uniform("l1_ratio", 0, 1),
        },
    ],
)


# Run the optimization process
trials = Trials()
best = fmin(
    fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials, verbose=2
)

# Extract the best hyperparameters
best_model = best["model"]
if best_model == 0:
    best_solver = ["newton-cg", "lbfgs", "sag", "saga"][best["solver_no_penalty"]]
    best_penalty = "none"
    best_C = best["C_no_penalty"]
elif best_model == 1:
    best_solver = ["newton-cg", "lbfgs", "sag", "saga"][best["solver_l2"]]
    best_penalty = "l2"
    best_C = best["C_l2"]
elif best_model == 2:
    best_solver = ["liblinear", "saga"][best["solver_l1"]]
    best_penalty = "l1"
    best_C = best["C_l1"]
else:
    best_solver = "saga"
    best_penalty = "elasticnet"
    best_C = best["C_elasticnet"]
    best_l1_ratio = best["l1_ratio"]

# Print the best hyperparameters found
print("Best hyperparameters:")
print("Solver:", best_solver)
print("Penalty:", best_penalty)
print("C:", best_C)
if best_penalty == "elasticnet":
    print("L1_ratio:", best_l1_ratio)
