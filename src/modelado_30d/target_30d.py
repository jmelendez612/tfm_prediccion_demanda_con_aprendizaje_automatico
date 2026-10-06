"""
Cálculo de target_30d: promedio de la demanda de los 30 días siguientes a
cada fila. Se descarta (NaN) si la ventana no cumple el umbral de
completitud de días reales.
"""

import pandas as pd

from . import config_30d as config


def _promedio_y_completitud_hacia_adelante(demanda: pd.Series, dia_real: pd.Series,
                                            ventana: int) -> tuple[pd.Series, pd.Series]:
    """
    Para cada posición i (con la serie YA ordenada por fecha), calcula:
      - el promedio de `demanda` en los próximos `ventana` días (i+1 .. i+ventana)
      - la fracción de esos días que son genuinamente reales

    Técnica: desplazar 1 día hacia atrás (shift(-1), así la posición i
    queda alineada con el día i+1), invertir la serie, aplicar una
    ventana móvil "hacia atrás" normal de pandas sobre la serie invertida
    (que equivale a "hacia adelante" en el orden original), y volver a
    invertir. min_periods=ventana exige que existan los 30 días
    calendario completos -- si no, NaN (no se puede promediar lo que no
    existe todavía, ej. el final de la serie).
    """
    demanda_manana = demanda.shift(-1)
    real_manana = dia_real.shift(-1).astype(float)

    promedio = demanda_manana[::-1].rolling(ventana, min_periods=ventana).mean()[::-1]
    completitud = real_manana[::-1].rolling(ventana, min_periods=ventana).mean()[::-1]

    return promedio, completitud


def calcular_target_30d(df: pd.DataFrame, columna_demanda: str) -> pd.DataFrame:
    """
    Agrega dos columnas al df: target_30d y completitud_30d.
    target_30d queda en NaN si completitud_30d < config.UMBRAL_COMPLETITUD,
    o si no hay 30 días calendario disponibles hacia adelante (final de
    cada serie -- esperado, no error).
    """
    for col in ("stock_flag", "STOCK_ES_REAL", "CONSUMO_ES_REAL"):
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}', requerida para calcular target_30d.")

    df = df.sort_values(["PRODUCTO_ID", "HOSPITAL_ID", "FECHA"]).copy()
    df["_dia_real"] = (
        (df["stock_flag"] == 1) & (df["STOCK_ES_REAL"] == 1) & (df["CONSUMO_ES_REAL"] == 1)
    )

    promedios, completitudes = [], []
    for _, g in df.groupby(["PRODUCTO_ID", "HOSPITAL_ID"], observed=True):
        promedio, completitud = _promedio_y_completitud_hacia_adelante(
            g[columna_demanda], g["_dia_real"], config.VENTANA_DIAS
        )
        promedios.append(promedio)
        completitudes.append(completitud)

    df["completitud_30d"] = pd.concat(completitudes)
    target_crudo = pd.concat(promedios)
    df["target_30d"] = target_crudo.where(df["completitud_30d"] >= config.UMBRAL_COMPLETITUD)

    df = df.drop(columns=["_dia_real"])
    return df
