import pandas as pd
import numpy as np
import os
import logger_setup as logger_setup
import data_ingstion as data_ingstion
import data_transformation as data_transformation
import model_training as model_training

logger = logger_setup.setup_logger("data_log", "logs/data_log.log")


def main():
    file_path = 'D:\\AlmaBetter\\P01_travelPrice\\data\\flights.csv'
    
    # Initialize DataIngestion class
    data_ingestor = data_ingstion.DataIngestion(file_path)
    data_transformer = data_transformation.DataTransformation()
    model_trainer = model_training.ModelTraining()
    
    try:
        # Read data
        data = data_ingestor.read_data()
        logger.info("Data read successfully.")
        print(data.head())
        # Transform data
        transformed_data = data_transformer.transform(data)
        preprocessed_data = data_transformer.preprocess(transformed_data)
        print(transformed_data.head())
        print(preprocessed_data)
        logger.info("Data transformed successfully.")

        # Train model
        target_column = 'price'  # Replace with the actual target column name
        model_trainer.train(transformed_data, target_column)
        model_trainer.save_model('D:\\AlmaBetter\\P01_travelPrice\\src\\models\\travel_price_model.pkl')
        logger.info("Model trained and saved successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

    

if __name__ == "__main__":
    main()


