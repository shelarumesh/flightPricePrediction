from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'simple_dag',
    default_args=default_args,
    description='A simple DAG',
    
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    t1 = EmptyOperator(
        task_id='start',
    )

    t2 = EmptyOperator(
        task_id='end',
    )

    t1 >> t2