"""
Baseline nivel 1: QTY_FORECAST (réplica del CPFR, tabla CPFR_FORECAST_RESULTADO).

Se obtiene por SQL (cruce con la tabla de la muestra) o desde un CSV
(ver config.CPFR_DESDE_CSV).

Estados de cálculo:
  - OK                  -> valor real
  - MEDIANA_NO_POSITIVA -> QTY_FORECAST = 0
  - SIN_BLOQUES_VALIDOS -> QTY_FORECAST = NaN; se excluye del MAE
"""

import sqlite3

import pandas as pd

from . import config


def _cruzar_desde_sqlite(df: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """JOIN directo en SQL: CPFR_FORECAST_RESULTADO (5,266) x MUESTRA (358)."""
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT C.FECHA_FORECAST AS FECHA, C.PRODUCTO_ID, C.HOSPITAL_ID,
               C.QTY_FORECAST, C.ESTADO_CALCULO
        FROM {config.TABLA_CPFR_REAL} C
        INNER JOIN {config.TABLA_MUESTRA} M
            ON C.PRODUCTO_ID = M.PRODUCTO_ID AND C.HOSPITAL_ID = M.HOSPITAL_ID
    """
    cpfr = pd.read_sql(query, conn)
    conn.close()
    cpfr["FECHA"] = pd.to_datetime(cpfr["FECHA"])

    # PRODUCTO_ID/HOSPITAL_ID pueden haberse guardado como número en SQLite
    # (tipado dinámico), perdiendo los ceros a la izquierda pese a la
    # columna estar declarada TEXT. Se rellenan de nuevo con el largo fijo
    # observado en el resto del proyecto (PRODUCTO_ID=9, HOSPITAL_ID=4).
    cpfr["PRODUCTO_ID"] = cpfr["PRODUCTO_ID"].astype(str).str.zfill(9)
    cpfr["HOSPITAL_ID"] = cpfr["HOSPITAL_ID"].astype(str).str.zfill(4)

    return cpfr


def _cruzar_desde_csv(ruta_cpfr: str) -> pd.DataFrame:
    # dtype=str es necesario: si no se fuerza, pandas interpreta PRODUCTO_ID/
    # HOSPITAL_ID como números y borra los ceros a la izquierda (ej.
    # "000450016" -> 450016), rompiendo el cruce contra el resto del dataset
    # (que sí conserva esos ceros). Ver mismo tratamiento en
    # _cruzar_desde_sqlite() (corrección de la auditoría de agosto de 2026).
    cpfr = pd.read_csv(ruta_cpfr, dtype={"PRODUCTO_ID": str, "HOSPITAL_ID": str})
    cpfr["PRODUCTO_ID"] = cpfr["PRODUCTO_ID"].str.zfill(9)
    cpfr["HOSPITAL_ID"] = cpfr["HOSPITAL_ID"].str.zfill(4)
    cpfr["FECHA_FORECAST"] = pd.to_datetime(cpfr["FECHA_FORECAST"])
    return cpfr.rename(columns={"FECHA_FORECAST": "FECHA"})[
        ["FECHA", "PRODUCTO_ID", "HOSPITAL_ID", "QTY_FORECAST", "ESTADO_CALCULO"]
    ]


def cruzar_cpfr_real(df: pd.DataFrame) -> pd.DataFrame:
    """Une QTY_FORECAST al dataset por FECHA/PRODUCTO_ID/HOSPITAL_ID."""
    if config.CPFR_DESDE_CSV:
        cpfr = _cruzar_desde_csv(config.RUTA_CPFR_REAL)
    else:
        cpfr = _cruzar_desde_sqlite(df, config.DB_PATH)

    merged_check_df = df.copy()
    merged_check_df["PRODUCTO_ID"] = merged_check_df["PRODUCTO_ID"].astype(str).str.zfill(9)
    merged_check_df["HOSPITAL_ID"] = merged_check_df["HOSPITAL_ID"].astype(str).str.zfill(4)

    merged = merged_check_df.merge(cpfr, on=["FECHA", "PRODUCTO_ID", "HOSPITAL_ID"], how="left")

    sin_match = merged["QTY_FORECAST"].isna().sum() - (merged["ESTADO_CALCULO"] == "SIN_BLOQUES_VALIDOS").sum()
    if sin_match > 0:
        print(f"  {sin_match:,} filas sin match en CPFR_FORECAST_RESULTADO (revisar cruce, "
              f"nombre de tabla de muestra, o cobertura de fechas)")

    return merged


def evaluar_fold(df: pd.DataFrame, test_inicio, test_fin, columna_demanda: str) -> dict:
    test = df[(df["FECHA"] >= test_inicio) & (df["FECHA"] <= test_fin)].copy()
    n_total = len(test)
    sin_forecast = test["QTY_FORECAST"].isna().sum()
    test = test.dropna(subset=["QTY_FORECAST"])

    mae = (test[columna_demanda] - test["QTY_FORECAST"]).abs().mean()

    return {
        "n_filas_evaluadas": len(test),
        "n_filas_sin_baseline": sin_forecast,
        "n_filas_total": n_total,
        "mae": mae,
    }


def evaluar_todos_los_folds(df: pd.DataFrame, metadata: pd.DataFrame, columna_demanda: str) -> pd.DataFrame:
    resultados = []
    for _, fila in metadata.iterrows():
        r = evaluar_fold(df, fila["test_inicio"], fila["test_fin"], columna_demanda)
        r["fold"] = int(fila["fold"])
        resultados.append(r)
    return pd.DataFrame(resultados)
