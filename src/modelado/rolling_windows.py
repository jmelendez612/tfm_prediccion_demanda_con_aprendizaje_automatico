"""
Rolling windows (walk-forward, ventana expansiva).

Genera N_CORTES bloques mensuales y, para cada uno, recalcula adi, cv2,
clasificacion_demanda y pct_censura_historica con datos anteriores al corte.
Agrega stock_ayer (CANTIDAD_STOCK desplazado un día).
"""

import pandas as pd

from . import config


def generar_cortes_mensuales(fecha_max, n_cortes: int) -> list[dict]:
    """Bloques mensuales consecutivos terminando en el último mes disponible."""
    periodos = pd.period_range(end=fecha_max.to_period("M"), periods=n_cortes, freq="M")
    cortes = []
    for i, p in enumerate(periodos, start=1):
        cortes.append({
            "fold": i,
            "fecha_corte": p.start_time,
            "test_inicio": p.start_time,
            "test_fin": p.end_time,
        })
    return cortes


def _dias_reales(g: pd.DataFrame) -> pd.DataFrame:
    """Filtro estricto de demanda genuinamente real (sin días imputados)."""
    return g[
        (g["stock_flag"] == 1) & (g["STOCK_ES_REAL"] == 1) & (g["CONSUMO_ES_REAL"] == 1)
    ]


def recalcular_intermitencia_hasta_corte(df_hasta_corte: pd.DataFrame,
                                          columna_demanda: str,
                                          clasificar_intermitencia) -> pd.DataFrame:
    """
    Recalcula adi/cv2/clasificacion_demanda/pct_censura_historica por
    SKU x Hospital, usando solo filas anteriores a la fecha de corte.
    clasificar_intermitencia: función de src.featuring.DataFeaturing.
    """
    filas = []
    for (sku, hospital), g in df_hasta_corte.groupby(["PRODUCTO_ID", "HOSPITAL_ID"], observed=True):
        dias = _dias_reales(g)
        adi, cv2, clasificacion = clasificar_intermitencia(dias[columna_demanda])
        pct_censura = (g["stock_flag"] == 0).mean() * 100
        filas.append({
            "PRODUCTO_ID": sku, "HOSPITAL_ID": hospital,
            "adi": adi, "cv2": cv2,
            "clasificacion_demanda": clasificacion,
            "pct_censura_historica": pct_censura,
        })
    return pd.DataFrame(filas)


def agregar_stock_ayer(df: pd.DataFrame) -> pd.DataFrame:
    """CANTIDAD_STOCK es stock de CIERRE del día (confirmado con negocio) —
    desplazado 1 día para que sea seguro en el tiempo."""
    df = df.sort_values(["PRODUCTO_ID", "HOSPITAL_ID", "FECHA"])
    df["stock_ayer"] = df.groupby(["PRODUCTO_ID", "HOSPITAL_ID"], observed=True)["CANTIDAD_STOCK"].shift(1)
    return df


def procesar(df: pd.DataFrame, columna_demanda: str, clasificar_intermitencia,
             n_cortes: int = config.N_CORTES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Punto de entrada del paso 2. Devuelve (df_enriquecido, metadata_folds).
    df_enriquecido tiene 4*n_cortes columnas nuevas (adi_foldN, cv2_foldN,
    clasificacion_demanda_foldN, pct_censura_historica_foldN) + stock_ayer.
    """
    for col in ("STOCK_ES_REAL", "CONSUMO_ES_REAL", "CANTIDAD_STOCK"):
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}', requerida para el paso 2.")

    df = agregar_stock_ayer(df)
    cortes = generar_cortes_mensuales(df["FECHA"].max(), n_cortes)

    for c in cortes:
        i = c["fold"]
        val_inicio = c["fecha_corte"] - pd.DateOffset(months=1)

        # 1. Fotografía limpia para Train y Val (antes de la validación)
        df_hasta_val = df[df["FECHA"] < val_inicio]
        tabla_val = recalcular_intermitencia_hasta_corte(df_hasta_val, columna_demanda, clasificar_intermitencia)
        tabla_val = tabla_val.rename(columns={
            "adi": f"adi_val_fold{i}",
            "cv2": f"cv2_val_fold{i}",
            "clasificacion_demanda": f"clasificacion_demanda_val_fold{i}",
            "pct_censura_historica": f"pct_censura_historica_val_fold{i}",
        })
        df = df.merge(tabla_val, on=["PRODUCTO_ID", "HOSPITAL_ID"], how="left")

        # 2. Fotografía limpia para Test (antes del inicio de test)
        df_hasta_test = df[df["FECHA"] < c["fecha_corte"]]
        tabla_test = recalcular_intermitencia_hasta_corte(df_hasta_test, columna_demanda, clasificar_intermitencia)
        tabla_test = tabla_test.rename(columns={
            "adi": f"adi_test_fold{i}",
            "cv2": f"cv2_test_fold{i}",
            "clasificacion_demanda": f"clasificacion_demanda_test_fold{i}",
            "pct_censura_historica": f"pct_censura_historica_test_fold{i}",
        })
        df = df.merge(tabla_test, on=["PRODUCTO_ID", "HOSPITAL_ID"], how="left")

    metadata = pd.DataFrame(cortes)
    return df, metadata
