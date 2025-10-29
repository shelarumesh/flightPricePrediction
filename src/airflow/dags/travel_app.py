import pandas as pd
import numpy as np
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta


class dataIngestion:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_data(self):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.file_path)
        return df

class dataTransformation:
    def __init__(self, df):
        self.df = df

    def transform_data(self):
        # Example transformation: Fill missing values and encode categorical variables
        self.df.fillna(method='ffill', inplace=True)
        self.df = pd.get_dummies(self.df, drop_first=True)
        return self.df

class dataEncoding:
    def __init__(self, df):
        self.df = df

    def encode_data(self):
        # Example encoding: Normalize numerical features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = (self.df[numeric_cols] - self.df[numeric_cols].mean()) / self.df[numeric_cols].std()
        return self.df