import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Load the dataset
csv_files = [
    "./data/2022QBCSV.csv",
    "./data/2021QBCSV.csv",
    "./data/2020QBCSV.csv",
    "./data/2019QBCSV.csv",
    "./data/2018QBCSV.csv",
]
data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Create test and training splits
X = data.drop(
    ["NAME", "SCHOOL", "YEAR", "10-YD", "ARMS", "HANDS", "WING", "DRAFTED", "ROUND"],
    axis=1,
)
y = data["DRAFTED"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("NFL Draft Classifier Experiment")


def run_experiment(params):
    with mlflow.start_run(run_name="Logistic_Regression: Fix 2019 Errors"):
        classifier = LogisticRegression(**params)
        classifier.fit(X_train, y_train)

        y_pred_train = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        # Calculate AUC score
        y_pred_train_prob = classifier.predict_proba(X_train)[:, 1]
        y_pred_test_prob = classifier.predict_proba(X_test)[:, 1]
        train_auc = roc_auc_score(y_train, y_pred_train_prob)
        test_auc = roc_auc_score(y_test, y_pred_test_prob)

        mlflow.log_params(params)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.sklearn.log_model(classifier, "model")


params = {"solver": "liblinear", "C": 1.0}

run_experiment(params)
