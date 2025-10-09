import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import logger_setup
import os
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import data_transformation


logger = logger_setup.setup_logger("ModelTraining", "D:\\AlmaBetter\\P01_travelPrice\\src\\logs\\model_training.log")
data_transformer = data_transformation.DataTransformation()

class ModelTraining:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, data: pd.DataFrame, target_column: str) -> None:
        
        try:
            if target_column not in data.columns:
                logger.error(f"Target column '{target_column}' not found in data.")
                raise ValueError(f"Target column '{target_column}' not found in data.")

            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Preprocess the features
            X = data_transformer.transform(X)
            X = data_transformer.preprocess(X)
            logger.info("Data preprocessing completed successfully. and X,y split started")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logger.info("Data split into training and testing sets successfully.")
            print(X_train[0:5])
            # Train the model
            logger.info("Model training started.")
            self.model.fit(X_train, y_train)
            logger.info("Model training completed successfully.")

            # Evaluate the model
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            logger.info(f"Model Evaluation Metrics - MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

        except Exception as e:
            logger.error(f"An error occurred during model training: {e}")
            raise Exception(f"An error occurred during model training: {e}")

    def save_model(self, model_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved successfully at {model_path}")
        except Exception as e:
            logger.error(f"An error occurred while saving the model: {e}")
            raise Exception(f"An error occurred while saving the model: {e}")