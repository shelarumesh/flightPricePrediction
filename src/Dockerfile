FROM apache/airflow:latest

USER root

RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean

# Switch to airflow user before installing Python packages
USER airflow

RUN pip install --no-cache-dir openpyxl
RUN pip install --no-cache-dir scikit-learn
RUN pip install --no-cache-dir joblib

EXPOSE 8080
