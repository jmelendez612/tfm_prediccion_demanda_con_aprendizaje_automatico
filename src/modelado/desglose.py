"""
Paso 5 — Desglose del MAE (baseline real vs LightGBM) por categoría de
intermitencia, combinando los N folds. Confirma/refuta si el error se
concentra en las categorías de mayor dispersión (erratic/lumpy).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from . import lightgbm_model as lgbm


def _procesar_fold(df: pd.DataFrame, fold: int, columna_demanda: str,
                    fecha_corte, test_inicio, test_fin) -> pd.DataFrame:
    X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat = lgbm.preparar_datos_fold(
        df, fold, columna_demanda, fecha_corte, test_inicio, test_fin
    )

    modelo = lgb.LGBMRegressor(**config.PARAMS_LGBM)
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=columnas_cat,
        callbacks=[lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    pred_lgbm = modelo.predict(X_test)

    # QTY_FORECAST no es feature de entrenamiento; se recupera del df
    # original con los mismos índices del test
    qty_forecast_test = df.loc[X_test.index, "QTY_FORECAST"]
    
    # CORRECCIÓN: Se lee la columna de la fotografía de test para el fold actual
    col_clasif_test = f"{config.CATEGORICA_POR_FOLD}_test_fold{fold}"
    
    # Si por alguna razón la columna renombrada está dentro de X_test bajo el nombre estándar:
    if config.CATEGORICA_POR_FOLD in X_test.columns:
        categorias = X_test[config.CATEGORICA_POR_FOLD].values
    else:
        categorias = df.loc[X_test.index, col_clasif_test].values

    detalle = pd.DataFrame({
        "categoria": categorias,
        "actual": y_test.values,
        "pred_baseline": qty_forecast_test.values,
        "pred_lightgbm": pred_lgbm,
    })
    detalle = detalle.dropna(subset=["pred_baseline"])  # SIN_BLOQUES_VALIDOS -> excluir
    detalle["error_baseline"] = (detalle["actual"] - detalle["pred_baseline"]).abs()
    detalle["error_lightgbm"] = (detalle["actual"] - detalle["pred_lightgbm"]).abs()
    detalle["fold"] = fold
    return detalle


def calcular_desglose(df: pd.DataFrame, metadata: pd.DataFrame, columna_demanda: str) -> pd.DataFrame:
    detalles = [
        _procesar_fold(df, int(f["fold"]), columna_demanda, f["fecha_corte"], f["test_inicio"], f["test_fin"])
        for _, f in metadata.iterrows()
    ]
    todo = pd.concat(detalles, ignore_index=True)

    resumen = todo.groupby("categoria").agg(
        n_filas=("actual", "size"),
        media_demanda_real=("actual", "mean"),
        mae_baseline=("error_baseline", "mean"),
        mae_lightgbm=("error_lightgbm", "mean"),
    ).round(2)
    resumen["mejora_pct"] = (100 * (1 - resumen["mae_lightgbm"] / resumen["mae_baseline"])).round(1)
    return resumen