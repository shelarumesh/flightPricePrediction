import pandas as pd
import os 
import numpy as np
import AlmaBetter.P01_travelPrice.src.airflow.dags.logger_setup as logger_setup

logger = logger_setup.setup_logger("DataIngestion", "D:\\AlmaBetter\\P01_travelPrice\\src\\logs\\data_ingestion.log")

class DataIngestion:
    
    def __init__(self, file_path):
        self.file_path = file_path

    def read_data(self):
        if not os.path.exists(self.file_path):
            logger.error(f"The file {self.file_path} does not exist.")
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        try:
            data = pd.read_csv(self.file_path)
            logger.info(f"Data successfully read from {self.file_path}")
            return data
        except Exception as e:
            logger.error(f"An error occurred while reading the file: {e}")
            raise Exception(f"An error occurred while reading the file: {e}")