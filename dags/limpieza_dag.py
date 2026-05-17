from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

# Obtener la ruta base del proyecto (subiendo desde la carpeta dags/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROYECTO_DIR = os.path.dirname(BASE_DIR)  # sube un nivel para llegar a proyecto_forecast

# Rutas de los archivos
RAW_DATA = os.path.join(PROYECTO_DIR, 'data', 'raw', 'online_retail_II.csv')
CLEAN_DATA = os.path.join(PROYECTO_DIR, 'data', 'clean', 'dataset_limpio.csv')
TMP_NULOS = os.path.join('C:\Temp', 'nulos_report.csv')
TMP_OUTLIERS = os.path.join('C:\Temp', 'outliers_limpio.csv')
TMP_FECHAS = os.path.join('C:\Temp', 'fechas_procesadas.csv')

default_args = {
    'owner': 'dahiana_meza',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def detectar_nulos():
    df = pd.read_csv(RAW_DATA)
    print("Nulos por columna:")
    print(df.isnull().sum())
    # Guarda reporte
    df.to_csv(TMP_NULOS, index=False)

def tratar_outliers():
    df = pd.read_csv(RAW_DATA)
    z_scores = np.abs((df['Quantity'] - df['Quantity'].mean()) / df['Quantity'].std())
    df_clean = df[z_scores < 3]
    df_clean.to_csv(TMP_OUTLIERS, index=False)
    print(f"Registros originales: {len(df)}, después de quitar outliers: {len(df_clean)}")

def procesar_fechas():
    df = pd.read_csv(TMP_OUTLIERS)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['year'] = df['Order Date'].dt.year
    df['month'] = df['Order Date'].dt.month
    df.to_csv(TMP_FECHAS, index=False)

def exportar():
    df = pd.read_csv(TMP_FECHAS)
    # Crear carpeta data/clean si no existe
    os.makedirs(os.path.dirname(CLEAN_DATA), exist_ok=True)
    df.to_csv(CLEAN_DATA, index=False)
    print(f"Datos limpios exportados a {CLEAN_DATA}")

with DAG(
    'limpieza_automatica',
    default_args=default_args,
    description='Limpieza de datos del proyecto de forecast',
    schedule_interval=None,   # Solo manual por ahora
    start_date=datetime(2026, 4, 23),
    catchup=False
) as dag:

    t1 = PythonOperator(task_id='detectar_nulos', python_callable=detectar_nulos)
    t2 = PythonOperator(task_id='tratar_outliers', python_callable=tratar_outliers)
    t3 = PythonOperator(task_id='procesar_fechas', python_callable=procesar_fechas)
    t4 = PythonOperator(task_id='exportar', python_callable=exportar)

    t1 >> t2 >> t3 >> t4