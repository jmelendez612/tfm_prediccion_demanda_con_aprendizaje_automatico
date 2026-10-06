"""
Rolling Windows para el horizonte de 30 días.

Diferencia clave frente al modelo de 1 día (src/modelado/rolling_windows.py):
  1. Los folds terminan 30 días antes que el máximo del dataset, porque
     target_30d de un día D necesita los 30 días SIGUIENTES para existir
     -- el último día evaluable es (fecha_max - 30 días).
  2. El corte de entrenamiento para cada fold se retrasa 30 días
     adicionales (fecha_corte_train = fecha_corte - 30 días), para
     excluir del entrenamiento las filas cuya ventana de 30 días hacia
     ADELANTE se asome al bloque de test de ese mismo fold -- fuga
     simétrica a la ya resuelta en adi/cv2 (que mira hacia atrás).

adi/cv2/clasificacion_demanda/pct_censura_historica SÍ pueden seguir
usando fecha_corte tal cual (sin el retraso adicional), porque esas son
sobre información pasada -- no tienen el problema de fuga hacia adelante.
"""

import pandas as pd

from src.modelado import rolling_windows as rw_1d
from . import config_30d as config
from . import target_30d as t30


def procesar(df: pd.DataFrame, columna_demanda: str, clasificar_intermitencia,
             n_cortes: int = config.N_CORTES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Punto de entrada. Devuelve (df_enriquecido, metadata_folds).
    metadata_folds tiene una columna extra frente al modelo de 1 día:
    fecha_corte_train (el corte real usado para armar el set de train,
    30 días más temprano que fecha_corte).
    """
    print("  Calculando target_30d (esto puede tardar un poco más que el de 1 día)...")
    df = t30.calcular_target_30d(df, columna_demanda)
    n_validos = df["target_30d"].notna().sum()
    print(f"  target_30d válido en {n_validos:,} de {len(df):,} filas "
          f"({100*n_validos/len(df):.1f}%, resto NaN por umbral de completitud o fin de serie)")

    for col in ("STOCK_ES_REAL", "CONSUMO_ES_REAL", "CANTIDAD_STOCK"):
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}', requerida para el paso 2.")

    df = rw_1d.agregar_stock_ayer(df)

    fecha_max_evaluable = df["FECHA"].max() - pd.Timedelta(days=config.VENTANA_DIAS)
    cortes = rw_1d.generar_cortes_mensuales(fecha_max_evaluable, n_cortes)
    for c in cortes:
        c["fecha_corte_train"] = c["fecha_corte"] - pd.Timedelta(days=config.VENTANA_DIAS)

    for c in cortes:
        i = c["fold"]
        val_inicio = c["fecha_corte_train"] - pd.DateOffset(months=1)

        # 1. Historial pre-validación
        df_hasta_val = df[df["FECHA"] < val_inicio]
        tabla_val = rw_1d.recalcular_intermitencia_hasta_corte(df_hasta_val, columna_demanda, clasificar_intermitencia)
        tabla_val = tabla_val.rename(columns={
            "adi": f"adi_val_fold{i}",
            "cv2": f"cv2_val_fold{i}",
            "clasificacion_demanda": f"clasificacion_demanda_val_fold{i}",
            "pct_censura_historica": f"pct_censura_historica_val_fold{i}",
        })
        df = df.merge(tabla_val, on=["PRODUCTO_ID", "HOSPITAL_ID"], how="left")

        # 2. Historial pre-test
        df_hasta_test = df[df["FECHA"] < c["test_inicio"]]
        tabla_test = rw_1d.recalcular_intermitencia_hasta_corte(df_hasta_test, columna_demanda, clasificar_intermitencia)
        tabla_test = tabla_test.rename(columns={
            "adi": f"adi_test_fold{i}",
            "cv2": f"cv2_test_fold{i}",
            "clasificacion_demanda": f"clasificacion_demanda_test_fold{i}",
            "pct_censura_historica": f"pct_censura_historica_test_fold{i}",
        })
        df = df.merge(tabla_test, on=["PRODUCTO_ID", "HOSPITAL_ID"], how="left")

    metadata = pd.DataFrame(cortes)
    return df, metadata
