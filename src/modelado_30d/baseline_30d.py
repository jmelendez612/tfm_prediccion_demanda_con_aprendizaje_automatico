"""
Baseline Nivel 1 (30 días): QTY_FORECAST (réplica real del CPFR).

═══════════════════════════════════════════════════════════════════════
CORRECCIÓN — Disponibilidad temporal real del baseline (dictamen de
jurado simulado, punto crítico #2)
───────────────────────────────────────────────────────────────────────
ANTES: baseline_30d(D) se calculaba como el promedio de
QTY_FORECAST(D+1)...QTY_FORECAST(D+30) -- pero QTY_FORECAST(D+k)
necesita datos reales hasta D+k-1 para existir, información que en el
día D (el momento real de la predicción) todavía no ocurrió. Era una
fuga de información real, no solo teórica: el "baseline" usaba hasta
29 días de futuro real para construirse.

AHORA: baseline_30d(D) = QTY_FORECAST(D) sin promediar nada. Es lo
único que el CPFR genuinamente tiene disponible el día D (calculado con
datos hasta D-1), proyectado de forma plana como estimación de la tasa
diaria para los 30 días siguientes -- el CPFR no tiene mecanismo nativo
para generar 30 pronósticos distintos sin información futura.
═══════════════════════════════════════════════════════════════════════
"""

import pandas as pd

from src.modelado import baseline as baseline_1d


def cruzar_y_calcular_baseline_30d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza QTY_FORECAST (igual que el modelo de 1 día) y define
    baseline_30d = QTY_FORECAST(D) -- la única tasa diaria genuinamente
    disponible en el día D, sin usar ningún dato posterior a D-1.
    """
    df = baseline_1d.cruzar_cpfr_real(df)
    df["baseline_30d"] = df["QTY_FORECAST"]
    return df


def evaluar_fold(df: pd.DataFrame, test_inicio, test_fin, columna_demanda: str) -> dict:
    test = df[(df["FECHA"] >= test_inicio) & (df["FECHA"] <= test_fin)].copy()
    test = test.dropna(subset=["target_30d", "baseline_30d"])

    mae = (test["target_30d"] - test["baseline_30d"]).abs().mean()
    return {"n_filas_evaluadas": len(test), "mae": mae}


def evaluar_todos_los_folds(df: pd.DataFrame, metadata: pd.DataFrame, columna_demanda: str) -> pd.DataFrame:
    resultados = []
    for _, fila in metadata.iterrows():
        r = evaluar_fold(df, fila["test_inicio"], fila["test_fin"], columna_demanda)
        r["fold"] = int(fila["fold"])
        resultados.append(r)
    return pd.DataFrame(resultados)
