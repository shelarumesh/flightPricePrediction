import pandas as pd
import numpy as np
import os
import logger_setup
import data_transformation
import model_training
from datetime import datetime, timedelta


logger = logger_setup.setup_logger("data_log", "logs/data_log.log")


def get_data():
    file_path = 'D:\\AlmaBetter\\P01_travelPrice\\data\\flights.csv'
    
    # Initialize DataIngestion class
    data_ingestor = data_ingstion.DataIngestion(file_path)
     # Read data
    data = data_ingestor.read_data()
    logger.info("Data read successfully.")
    return data

def data_transform(data):

    data_transformer = data_transformation.DataTransformation()
    # Transform data
    # Transform data
    transformed_data = data_transformer.transform(data)
    preprocessed_data = data_transformer.preprocess(transformed_data)
    print(transformed_data.head())
    print(preprocessed_data)
    logger.info("Data transformed successfully.")
    return transformed_data

def model_train(data):
    model_trainer = model_training.ModelTraining()
 # Train model
    target_column = 'price'  # Replace with the actual target column name
    model_trainer.train(transformed_data, target_column)
    model_trainer.save_model('D:\\AlmaBetter\\P01_travelPrice\\src\\models\\travel_price_model.pkl')
    logger.info("Model trained and saved successfully.")