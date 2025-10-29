
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def my_name():
    name = "Shelar Umesh Prakash"
    print(f"My name is {name}")
    return name

def my_address():
    address = "82, Tandlekar wadi, \n Post Kotheri, \n Tal - Mahad, \n Dist - Raigad, \n Pin - 402301"
    print(f"My address is {address}")
    return address

def my_mobile():
    mobile = "7057279787"
    print(f"My mobile number is {mobile}")
    return mobile

def my_info(name, address, mobile):
    info = f"Name: {name}\nAddress: {address}\nMobile: {mobile}"
    print(f"My complete information is:\n{info}")
    return info


with DAG(
    'umesh_info',
    default_args=default_args,
    description='A simple DAG',
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id='my_name',
        python_callable=my_name,
    )

    t2 = PythonOperator(
        task_id='my_address',
        python_callable=my_address,
    )

    t3 = PythonOperator(
        task_id='my_mobile',
        python_callable=my_mobile,
    )

    t4 = PythonOperator(
        task_id='m_info',
        python_callable=my_info,
        op_kwargs={
            'name': "{{ task_instance.xcom_pull(task_ids='my_name') }}",
            'address': "{{ task_instance.xcom_pull(task_ids='my_address') }}",
            'mobile': "{{ task_instance.xcom_pull(task_ids='my_mobile') }}"
        }
    )

    t1 >> t2 >> t3 >> t4
