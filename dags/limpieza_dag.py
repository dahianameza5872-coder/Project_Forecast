from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

# Rutas dentro del contenedor Docker
RAW_DATA = '/opt/airflow/data/raw/online_retail_II.csv'
CLEAN_DATA = '/opt/airflow/data/clean/dataset_limpio.csv'
# Archivos intermedios (usamos /tmp que existe en Linux/Docker)
TMP_NULOS = '/tmp/nulos_report.csv'
TMP_OUTLIERS = '/tmp/outliers_limpio.csv'
TMP_FECHAS = '/tmp/fechas_procesadas.csv'

default_args = {
    'owner': 'dahiana_meza',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def detectar_nulos():
    df = pd.read_csv(RAW_DATA)
    print("Nulos por columna:")
    print(df.isnull().sum())
    df.to_csv(TMP_NULOS, index=False)

def tratar_outliers():
    df = pd.read_csv(RAW_DATA)
    # Trabajamos con la columna 'Quantity'
    z_scores = np.abs((df['Quantity'] - df['Quantity'].mean()) / df['Quantity'].std())
    df_clean = df[z_scores < 3]
    print(f"Registros originales: {len(df)}, después de quitar outliers: {len(df_clean)}")
    df_clean.to_csv(TMP_OUTLIERS, index=False)

def procesar_fechas():
    df = pd.read_csv(TMP_OUTLIERS)
    # La columna de fecha es 'InvoiceDate'
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['year'] = df['InvoiceDate'].dt.year
    df['month'] = df['InvoiceDate'].dt.month
    df.to_csv(TMP_FECHAS, index=False)

def exportar():
    df = pd.read_csv(TMP_FECHAS)
    # Crear carpeta destino si no existe
    os.makedirs(os.path.dirname(CLEAN_DATA), exist_ok=True)
    df.to_csv(CLEAN_DATA, index=False)
    print(f"Datos limpios exportados a {CLEAN_DATA}")

with DAG(
    'limpieza_automatica',
    default_args=default_args,
    description='Limpieza de datos del proyecto de forecast',
    schedule_interval=None,
    start_date=datetime(2026, 4, 23),
    catchup=False
) as dag:

    t1 = PythonOperator(task_id='detectar_nulos', python_callable=detectar_nulos)
    t2 = PythonOperator(task_id='tratar_outliers', python_callable=tratar_outliers)
    t3 = PythonOperator(task_id='procesar_fechas', python_callable=procesar_fechas)
    t4 = PythonOperator(task_id='exportar', python_callable=exportar)

    t1 >> t2 >> t3 >> t4