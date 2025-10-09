from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
import requests
import pandas as pd


default_args = {
    'owner': 'airflow_umesh',
    'Start_date': '2023,01,01'}

das = DAG(
    'travel_price_dag',
    default_args=default_args,
    description='A simple travel price DAG',
    schedule_interval='17 * * *',
    catchup=False
)

def det_data():
    print("Data Extraction Started")

def transform_data():
    print("Data Transformation Started")

def load_data():
    print("Data Loading Started")


get_data = PythonOperator(
    task_id='extract_data',
    python_callable=det_data,
    dag=dag
)

transform_data = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)
get_data >> transform_data >> load_data