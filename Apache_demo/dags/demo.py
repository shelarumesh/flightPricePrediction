import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'demo_dag',
    default_args=default_args,
    description='A simple demo DAG',
    schedule_interval=timedelta(days=1),
)

def fetch_data():
    response = requests.get("http://api.worldbank.org/v2/region?format=json")
    data = response.json()
    df = pd.DataFrame(data)
    df.to_csv("/path/to/save/data.csv", index=False)

def process_data():
    df = pd.read_csv("/path/to/save/data.csv")
    # Perform some data processing
    df['new_column'] = df['existing_column'] * 2
    df.to_csv("/path/to/save/processed_data.csv", index=False)

def upload_data():
    files = {'file': open("/path/to/save/processed_data.csv", 'rb')}
    response = requests.post("https://api.example.com/upload", files=files)
    print(response.text)

fetch_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='upload_data',
    python_callable=upload_data,
    dag=dag,
)

fetch_task >> process_task >> upload_task