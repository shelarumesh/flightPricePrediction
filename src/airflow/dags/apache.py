import pandas as pd
import numpy as np
import os
# from airflow import DAG
# from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def data_ingestion(**kwargs):
    file_path = '/opt/airflow/data/flights.csv'
    data = pd.read_csv(file_path)
    return data

def data_processor(data):
    # Perform data processing steps
    data = data.dropna()
    data = data[data['price'] > 0]
    data['date'] = pd.to_datetime(data['date'])
    return data

def data_encoding(data):
    # Perform data encoding steps
    le= LabelEncoder()
    data['from'] = le.fit_transform(data['from'])
    data['to'] = le.fit_transform(data['to'])
    data['flightType'] = le.fit_transform(data['flightType'])
    data['agency'] = le.fit_transform(data['agency'])
    data.drop(['date','time'], axis=1, inplace=True)
    X=data.drop('price', axis=1)
    y=data['price']
    return X, y

def model_trainer(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    return model


def evaluate(model):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    score = model.score(X_test, y_test)

    result = {
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'R^2 Score': r2,
        'Mean Absolute Error': mae,
        'Model Score': score
      
    }
    print( pd.DataFrame(result, index=[0]).T)
    return result



def save_model(model):
    # Save the model
    import joblib
    joblib.dump(model, '/opt/airflow/travel_price_model.pkl')


data = data_ingestion()
print("Data Ingested")
processed_data = data_processor(data)
print("Data Processed")
X, y = data_encoding(processed_data)
print("Data Encoded")
model = model_trainer(X, y)
print("Model Trained")
evaluate(model)
print("Model Evaluated")
save_model(model)   
print("Model Saved")



# starting DAG Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    'owner': 'airflow',
    'start_date': 'days_ago(1)'
}

dag = DAG(
    'travel_price_prediction',
    default_args=default_args,
    description='A DAG for training and evaluating a travel price prediction model',
    schedule_interval=timedelta(days=1)
)

ingest_task = PythonOperator(
    task_id='data_ingestion',
    python_callable=data_ingestion,
    dag=dag
)

process_task = PythonOperator(
    task_id='data_processing',
    python_callable=data_processor,
    dag=dag
)

encode_task = PythonOperator(
    task_id='data_encoding',
    python_callable=data_encoding,
    dag=dag
)

train_task = PythonOperator(
    task_id='model_training',
    python_callable=model_trainer,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='model_evaluation',
    python_callable=evaluate,
    dag=dag
)

save_task = PythonOperator(
    task_id='model_saving',
    python_callable=save_model,
    dag=dag
)

ingest_task >> process_task >> encode_task >> train_task >> evaluate_task >> save_task
