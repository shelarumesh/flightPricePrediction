import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import logger_setup as logger_setup
import os
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import data_transformation as data_transformation
from sklearn.metrics import r2_score, accuracy_score


logger = logger_setup.setup_logger("ModelTraining", "D:\\AlmaBetter\\P01_travelPrice\\src\\logs\\model_training.log")
data_transformer = data_transformation.DataTransformation()

class ModelTraining:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=5, random_state=42)

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


            logger.info("Model evaluation started.")
            # Evaluate the model
            y_pred = self.model.predict(X_test)
            logger.info("Model evaluation started. yPred calculated")
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            logger.info("Model evaluation started. mae and mse completed")
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            logger.info("Model evaluation started. rmse and r2 calculated")
            print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")
            logger.info(f"Model Evaluation Metrics - \nMAE: {mae}, \nMSE: {mse}, \nRMSE: {rmse}, \nR2: {r2}")

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

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'price': [100, 200, 300, 400, 500]
    })
    model_trainer = ModelTraining()
    model_trainer.train(data, 'price')

    