from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import re

# ─── Rutas ────────────────────────────────────────────────────────────────────
RAW_DATA   = '/opt/airflow/data/raw/online_retail_II.csv'
CLEAN_DATA = '/opt/airflow/data/clean/dataset_limpio.csv'

TMP_CARGADO    = '/tmp/01_cargado.csv'
TMP_DUPLICADOS = '/tmp/02_sin_duplicados.csv'
TMP_NULOS      = '/tmp/03_sin_nulos.csv'
TMP_CANCELADOS = '/tmp/04_sin_cancelados.csv'
TMP_STOCKCODES = '/tmp/05_stockcodes_limpios.csv'
TMP_NUMERICOS  = '/tmp/06_numericos_limpios.csv'
TMP_OUTLIERS   = '/tmp/07_sin_outliers.csv'
TMP_FECHAS     = '/tmp/08_fechas_procesadas.csv'
TMP_DESCRIP    = '/tmp/09_descriptions_limpias.csv'

default_args = {
    'owner': 'dahiana_meza',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# ─── TAREA 1: Cargar el CSV con encoding correcto ─────────────────────────────
def cargar_datos():
    """
    Carga el CSV ignorando líneas malformadas.
    El dataset usa UTF-8; on_bad_lines='skip' descarta filas corruptas.
    """
    df = pd.read_csv(RAW_DATA, encoding='utf-8', on_bad_lines='skip')
    print(f"[CARGA] Filas leídas: {len(df):,}  |  Columnas: {list(df.columns)}")
    df.to_csv(TMP_CARGADO, index=False)


# ─── TAREA 2: Eliminar duplicados exactos ─────────────────────────────────────
def eliminar_duplicados():
    """
    El dataset tiene 34,335 filas completamente duplicadas (3.2 %).
    Se eliminan conservando la primera ocurrencia.
    """
    df = pd.read_csv(TMP_CARGADO)
    antes = len(df)
    df = df.drop_duplicates()
    print(f"[DUPLICADOS] Eliminados: {antes - len(df):,}  |  Quedan: {len(df):,}")
    df.to_csv(TMP_DUPLICADOS, index=False)


# ─── TAREA 3: Tratar nulos ────────────────────────────────────────────────────
def tratar_nulos():
    """
    Problemas encontrados:
      - Description:  4,382 nulos  (0.4%) → se eliminan (sin descripción no tiene sentido el registro)
      - Customer ID: 243,007 nulos (22.8%) → se imputa con 0 como 'Cliente desconocido'
                     (se conservan porque muchas transacciones válidas no tienen ID)
    """
    df = pd.read_csv(TMP_DUPLICADOS)

    # Description: eliminar nulos
    antes = len(df)
    df = df.dropna(subset=['Description'])
    print(f"[NULOS] Description nulos eliminados: {antes - len(df):,}")

    # Customer ID: imputar con 0 (cliente anónimo)
    nulos_cid = df['Customer ID'].isna().sum()
    df['Customer ID'] = df['Customer ID'].fillna(0).astype(int)
    print(f"[NULOS] Customer ID imputados con 0: {nulos_cid:,}")

    print(f"[NULOS] Filas restantes: {len(df):,}")
    df.to_csv(TMP_NULOS, index=False)


# ─── TAREA 4: Eliminar facturas canceladas ────────────────────────────────────
def eliminar_canceladas():
    """
    Las facturas que empiezan con 'C' son devoluciones/cancelaciones (19,494 registros).
    También tienen Quantity negativa. Se eliminan porque distorsionan el forecast de ventas.
    """
    df = pd.read_csv(TMP_NULOS)
    antes = len(df)
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    print(f"[CANCELADAS] Facturas canceladas eliminadas: {antes - len(df):,}  |  Quedan: {len(df):,}")
    df.to_csv(TMP_CANCELADOS, index=False)


# ─── TAREA 5: Filtrar StockCodes no estándar ──────────────────────────────────
def filtrar_stockcodes():
    """
    StockCodes válidos siguen el patrón: 5 dígitos + letras opcionales (ej. '85048', '22041A').
    Se eliminan códigos administrativos como POST, DOT, M, C2, D, BANK CHARGES, TEST001, etc.
    Estos 6,094 registros no son productos reales y distorsionan el análisis.
    """
    df = pd.read_csv(TMP_CANCELADOS)
    antes = len(df)
    patron_valido = r'^\d{5}[A-Za-z]*$'
    df = df[df['StockCode'].astype(str).str.match(patron_valido)]
    print(f"[STOCKCODES] Registros no estándar eliminados: {antes - len(df):,}  |  Quedan: {len(df):,}")
    df.to_csv(TMP_STOCKCODES, index=False)


# ─── TAREA 6: Limpiar valores numéricos inválidos ─────────────────────────────
def limpiar_numericos():
    """
    Problemas encontrados:
      - Quantity < 0:  22,950 registros  (después de eliminar canceladas, quedan residuales)
      - Price <= 0:     6,207 registros  (Price = 0 sin sentido comercial; Price < 0 es error)
      - Country = 'Unspecified': 756 registros → se reemplaza por 'Unknown'
    """
    df = pd.read_csv(TMP_STOCKCODES)

    # Quantity: eliminar negativos residuales (no capturados por el prefijo 'C')
    antes = len(df)
    df = df[df['Quantity'] > 0]
    print(f"[NUMERICOS] Quantity <= 0 eliminados: {antes - len(df):,}")

    # Price: eliminar cero o negativos
    antes = len(df)
    df = df[df['Price'] > 0]
    print(f"[NUMERICOS] Price <= 0 eliminados: {antes - len(df):,}")

    # Country 'Unspecified' → 'Unknown'
    n_unspec = (df['Country'] == 'Unspecified').sum()
    df['Country'] = df['Country'].replace('Unspecified', 'Unknown')
    print(f"[NUMERICOS] Country 'Unspecified' reemplazados: {n_unspec:,}")

    print(f"[NUMERICOS] Filas restantes: {len(df):,}")
    df.to_csv(TMP_NUMERICOS, index=False)


# ─── TAREA 7: Eliminar outliers ───────────────────────────────────────────────
def eliminar_outliers():
    """
    Se aplica el método IQR (más robusto que z-score para distribuciones sesgadas como ventas).
    IQR = Q3 - Q1; se eliminan valores fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    
    Columnas tratadas: Quantity y Price.
    
    PROBLEMA DEL DAG ORIGINAL: usaba z-score sobre el CSV raw, ignorando que había
    negativos, cancelaciones y StockCodes administrativos que inflaban la media y desviación.
    Ahora se aplica DESPUÉS de todos los filtros previos para mayor precisión.
    """
    df = pd.read_csv(TMP_NUMERICOS)
    antes = len(df)

    for col in ['Quantity', 'Price']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_antes = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f"[OUTLIERS] {col}: [{lower:.2f}, {upper:.2f}]  |  Eliminados: {n_antes - len(df):,}")

    print(f"[OUTLIERS] Total eliminados: {antes - len(df):,}  |  Quedan: {len(df):,}")
    df.to_csv(TMP_OUTLIERS, index=False)


# ─── TAREA 8: Procesar fechas ─────────────────────────────────────────────────
def procesar_fechas():
    """
    Convierte InvoiceDate a datetime y extrae:
      - year, month, day, hour, day_of_week (para el forecast por período)
      - week_of_year (útil para patrones semanales)
    
    PROBLEMA DEL DAG ORIGINAL: solo extraía year y month, insuficiente para forecast.
    """
    df = pd.read_csv(TMP_OUTLIERS)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], infer_datetime_format=True)

    df['year']        = df['InvoiceDate'].dt.year
    df['month']       = df['InvoiceDate'].dt.month
    df['day']         = df['InvoiceDate'].dt.day
    df['hour']        = df['InvoiceDate'].dt.hour
    df['day_of_week'] = df['InvoiceDate'].dt.dayofweek   # 0=lunes, 6=domingo
    df['week_of_year']= df['InvoiceDate'].dt.isocalendar().week.astype(int)

    # Columna de revenue por fila (útil directamente para el forecast)
    df['revenue'] = df['Quantity'] * df['Price']

    print(f"[FECHAS] Columnas generadas: year, month, day, hour, day_of_week, week_of_year, revenue")
    print(f"[FECHAS] Rango de fechas: {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
    df.to_csv(TMP_FECHAS, index=False)


# ─── TAREA 9: Limpiar Descriptions ───────────────────────────────────────────
def limpiar_descriptions():
    """
    Problemas encontrados en Description:
      - Strings con '???', '??', 'lost', 'wet', 'check?' → son notas internas, no productos.
      - Espacios extra al inicio/final.
      - Descriptions en minúsculas mezcladas con mayúsculas.
    
    Se eliminan registros cuya descripción contiene solo signos de interrogación,
    palabras como 'lost', 'damages', 'adjust', o tiene menos de 3 caracteres reales.
    El resto se normaliza a mayúsculas y se limpian espacios.
    """
    df = pd.read_csv(TMP_FECHAS)
    antes = len(df)

    # Normalizar: strip + upper
    df['Description'] = df['Description'].astype(str).str.strip().str.upper()

    # Eliminar descriptions basura
    patron_basura = r'^[\?\*\!\s]+$|^(LOST|DAMAGES|ADJUST|WET|CHECK|MISSING|THROWN|RUSTY)'
    mask_basura = df['Description'].str.match(patron_basura, na=False)
    df = df[~mask_basura]

    # Eliminar descriptions demasiado cortas (menos de 3 caracteres)
    df = df[df['Description'].str.len() >= 3]

    print(f"[DESCRIPTIONS] Registros basura eliminados: {antes - len(df):,}  |  Quedan: {len(df):,}")
    df.to_csv(TMP_DESCRIP, index=False)


# ─── TAREA 10: Exportar dataset final ─────────────────────────────────────────
def exportar():
    """
    Exporta el dataset limpio y genera un reporte de resumen.
    Customer ID se convierte a int (ya no tiene NaN).
    """
    df = pd.read_csv(TMP_DESCRIP)

    # Asegurar tipos correctos
    df['Customer ID'] = df['Customer ID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Crear carpeta destino si no existe
    os.makedirs(os.path.dirname(CLEAN_DATA), exist_ok=True)
    df.to_csv(CLEAN_DATA, index=False)

    # Reporte final
    print("=" * 55)
    print("DATASET LIMPIO - REPORTE FINAL")
    print("=" * 55)
    print(f"  Filas finales:      {len(df):,}")
    print(f"  Columnas:           {list(df.columns)}")
    print(f"  Nulos restantes:    {df.isnull().sum().sum()}")
    print(f"  Duplicados:         {df.duplicated().sum()}")
    print(f"  Quantity mín/máx:   {df['Quantity'].min()} / {df['Quantity'].max()}")
    print(f"  Price mín/máx:      {df['Price'].min()} / {df['Price'].max()}")
    print(f"  Revenue total:      {df['revenue'].sum():,.2f}")
    print(f"  Rango de fechas:    {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
    print(f"  Países:             {df['Country'].nunique()}")
    print(f"  Productos únicos:   {df['StockCode'].nunique()}")
    print(f"  Clientes únicos:    {df['Customer ID'].nunique()}")
    print("=" * 55)
    print(f"Exportado en: {CLEAN_DATA}")


# ─── DEFINICIÓN DEL DAG ───────────────────────────────────────────────────────
with DAG(
    'limpieza_automatica',
    default_args=default_args,
    description='Limpieza completa del dataset Online Retail II para forecast',
    schedule_interval=None,
    start_date=datetime(2026, 4, 23),
    catchup=False,
    tags=['limpieza', 'forecast', 'ml'],
) as dag:

    t1  = PythonOperator(task_id='cargar_datos',          python_callable=cargar_datos)
    t2  = PythonOperator(task_id='eliminar_duplicados',   python_callable=eliminar_duplicados)
    t3  = PythonOperator(task_id='tratar_nulos',          python_callable=tratar_nulos)
    t4  = PythonOperator(task_id='eliminar_canceladas',   python_callable=eliminar_canceladas)
    t5  = PythonOperator(task_id='filtrar_stockcodes',    python_callable=filtrar_stockcodes)
    t6  = PythonOperator(task_id='limpiar_numericos',     python_callable=limpiar_numericos)
    t7  = PythonOperator(task_id='eliminar_outliers',     python_callable=eliminar_outliers)
    t8  = PythonOperator(task_id='procesar_fechas',       python_callable=procesar_fechas)
    t9  = PythonOperator(task_id='limpiar_descriptions',  python_callable=limpiar_descriptions)
    t10 = PythonOperator(task_id='exportar',              python_callable=exportar)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8 >> t9 >> t10