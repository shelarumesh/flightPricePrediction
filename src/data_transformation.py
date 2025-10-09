import pandas as pd 
import numpy as np
import logger_setup
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


logger = logger_setup.setup_logger("DataTransformation", "D:\\AlmaBetter\\P01_travelPrice\\src\\logs\\data_transformation.log")


class DataTransformation:
    def __init__(self):
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Example transformation: Fill missing values and convert date columns
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data.fillna(method='ffill', inplace=True)
            logger.info("Data transformation successful.")
            return data
        except Exception as e:
            logger.error(f"An error occurred during data transformation: {e}")
            raise Exception(f"An error occurred during data transformation: {e}")
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Example preprocessing: Scaling numerical features and encoding categorical features
            numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = data.select_dtypes(include=['object']).columns.tolist()

            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

            data_preprocessed = pipeline.fit_transform(data)
            logger.info("Data preprocessing successful.")
            return data_preprocessed
        except Exception as e:
            logger.error(f"An error occurred during data preprocessing: {e}")
            raise Exception(f"An error occurred during data preprocessing: {e}")