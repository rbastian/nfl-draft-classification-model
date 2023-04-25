import mlflow
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# MLFlow Run
logged_model = "runs:/951bd1d69b184405924d0314af777abf/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = loaded_model.predict(pd.DataFrame(X_test))

# Calculate the confusion matrix

# Check the lengths of y_test and y_pred
assert len(y_test) == len(y_pred), "y_test and y_pred lengths do not match"
cm = confusion_matrix(y_test, y_pred)

X_test["PREDICTION"] = y_pred
print(X_test)
print("Confusion Matrix:\n", cm)
