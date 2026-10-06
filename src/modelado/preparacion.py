"""
Paso 1 — Preparación: carga del CSV y conversión de categóricas a dtype
'category' (SQLite no preserva ese dtype, se hace al cargar en pandas).
"""

import pandas as pd

from . import config


def parsear_fechas(serie: pd.Series) -> pd.Series:
    """
    Detecta el formato de fecha (ISO 'YYYY-MM-DD' o 'D/M/YYYY'). Evita
    dayfirst+mixed, que invierte las fechas ISO cuando día y mes son <= 12.
    """
    serie = serie.astype(str)
    if serie.str.contains("/").any():
        return pd.to_datetime(serie, dayfirst=True)
    return pd.to_datetime(serie, format="ISO8601")


def cargar_dataset(ruta_csv: str) -> pd.DataFrame:
    """Carga el CSV, parsea FECHA y convierte las categóricas fijas."""
    df = pd.read_csv(ruta_csv)
    df["FECHA"] = parsear_fechas(df["FECHA"])

    # PRODUCTO_ID/HOSPITAL_ID a veces se leen como número si en algún punto
    # de la base perdieron el tipo TEXT — se fuerzan a string con ceros a
    # la izquierda para que el cruce posterior con CPFR_FORECAST_RESULTADO
    # (baseline.py) nunca falle por tipos, sea cual sea el lado afectado.
    df["PRODUCTO_ID"] = df["PRODUCTO_ID"].astype(str).str.zfill(9)
    df["HOSPITAL_ID"] = df["HOSPITAL_ID"].astype(str).str.zfill(4)

    for col in config.CATEGORICAS_FIJAS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
