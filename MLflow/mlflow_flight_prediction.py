import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,LabelEncoder
import ast
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import logging
import os
import pickle

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Start an MLflow run
mlflow.start_run()

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

model = DecisionTreeRegressor()
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"mae {mae}")
print(f"mse {mse}")
print(f"rmse {rmse}")
print(f"r2 {r2}")

mlflow.log_param("test_size", 0.3)
mlflow.log_param("random_state", 42)
mlflow.log_metric("MAE", mae)
mlflow.log_metric("MSE", mse)
mlflow.log_metric("RMSE", rmse)
mlflow.log_metric("R2", r2)

# Save the trained model to MLflow

mlflow.sklearn.log_model(model, "regression_model")

predictions = model.predict(X_test)
signature = infer_signature(X_test, predictions)

tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

# Model registry does not work with file store
if tracking_url_type_store != "file":
    mlflow.sklearn.log_model(
        model, "model", registered_model_name= "dt_regression_model", signature=signature
    )
else:
    mlflow.sklearn.log_model(model, "model", signature=signature)

# End the MLflow run
mlflow.end_run()
