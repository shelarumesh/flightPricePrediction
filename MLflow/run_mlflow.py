import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pandas as pd

# Step 1: Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "flight_price_prediction"

if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)


path = "D:\AlmaBetter\P01_travelPrice\data\preprocessed_data.csv"
df = pd.read_csv(path)
le = LabelEncoder()
df['from'] = le.fit_transform(df["from"])
df['to'] = le.fit_transform(df["to"])
df['flightType'] = le.fit_transform(df["flightType"])
df['agency'] = le.fit_transform(df["agency"])
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Step 3: Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Step 4: Start MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id

    # Log parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.3)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.sklearn.log_model(lr_model, "linear_regression_model")

    # Register model
    mlflow.register_model(
        f"runs:/{run_id}/linear_regression_model",
        "TravelPackagePriceModel"
    )

# Step 5: Load model back (optional)
model_uri = f"runs:/{run_id}/linear_regression_model"
loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)

# Step 6: End run (not needed if using context manager)
mlflow.end_run()