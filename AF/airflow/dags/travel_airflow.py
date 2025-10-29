from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from Etl import ExtractLoadTransformSave
import time
start = time.time()
# your code
import numpy as np

from airflow.decorators import dag, task
from datetime import datetime

@dag(start_date=datetime(2023, 1, 1), catchup=False)
def travel_pipeline():

    @task
    def run_loader():
        etl = ExtractLoadTransformSave()
        etl.data_loader()

    @task
    def run_transform():
        etl = ExtractLoadTransformSave()
        etl.data_transformation()

    @task
    def run_model():
        etl = ExtractLoadTransformSave()
        etl.model_training()

    run_loader() >> run_transform() >> run_model()

travel_pipeline()

