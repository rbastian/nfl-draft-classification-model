import mlflow
import pandas as pd

# MLFlow Run
logged_model = "runs:/951bd1d69b184405924d0314af777abf/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


# Load the dataset
csv_files = ["./data/2023QBCSV.csv"]
data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Create test and training splits
X = data.drop(
    ["NAME", "SCHOOL", "YEAR", "10-YD", "ARMS", "HANDS", "WING", "DRAFTED", "ROUND"],
    axis=1,
)
print(X)
predictions = loaded_model.predict(pd.DataFrame(X))
data = data.drop(["10-YD", "ARMS", "HANDS", "WING"], axis=1)
data["PREDICTION"] = predictions
print(data)
