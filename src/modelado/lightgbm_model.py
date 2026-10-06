"""
Paso 4 — LightGBM (Nivel 2), entrenado y evaluado por fold (walk-forward).

Features excluidas explícitamente (fuga de información, ver contexto
del proyecto sección 11.3): DEMANDA_IMPUTADA/_SBA (target),
CONSUMO_REGISTRADO, CANTIDAD_STOCK sin desplazar, stock_flag,
STOCK_ES_REAL, CONSUMO_ES_REAL, CURVA_PQR, QTY_FORECAST (es el baseline).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config


def preparar_datos_fold(df: pd.DataFrame, fold: int, columna_demanda: str,
                         fecha_corte, test_inicio, test_fin):
    
    val_inicio = fecha_corte - pd.DateOffset(months=1)

    # Definimos los sufijos diferenciados
    cols_num_val = [f"{c}_val_fold{fold}" for c in config.NUMERICAS_POR_FOLD]
    cols_num_test = [f"{c}_test_fold{fold}" for c in config.NUMERICAS_POR_FOLD]
    col_cat_val = f"{config.CATEGORICA_POR_FOLD}_val_fold{fold}"
    col_cat_test = f"{config.CATEGORICA_POR_FOLD}_test_fold{fold}"

    # Features para Train/Val vs Features para Test
    features_val = config.CATEGORICAS_FIJAS + [col_cat_val] + config.NUMERICAS_FIJAS + cols_num_val
    features_test = config.CATEGORICAS_FIJAS + [col_cat_test] + config.NUMERICAS_FIJAS + cols_num_test

    # Creamos un renombre dinámico para que X_train y X_test tengan las mismas columnas exactas
    mapa_renombre = {col_cat_test: col_cat_val}
    for c_test, c_val in zip(cols_num_test, cols_num_val):
        mapa_renombre[c_test] = c_val

    cols_a_extraer = list(set(["FECHA", columna_demanda, "QTY_FORECAST"] + features_val + features_test))
    df_fold = df[df["FECHA"] <= test_fin][cols_a_extraer].copy()

    # Convertimos categóricas
    df_fold[col_cat_val] = df_fold[col_cat_val].astype("category")
    df_fold[col_cat_test] = df_fold[col_cat_test].astype("category")

    # Separación
    train = df_fold[df_fold["FECHA"] < val_inicio]
    val = df_fold[(df_fold["FECHA"] >= val_inicio) & (df_fold["FECHA"] < fecha_corte)]
    test = df_fold[(df_fold["FECHA"] >= test_inicio) & (df_fold["FECHA"] <= test_fin) & (df_fold["QTY_FORECAST"].notna())]

    # Preparamos X con sus nombres estandarizados
    X_train = train[features_val]
    X_val = val[features_val]
    X_test = test[features_test].rename(columns=mapa_renombre)

    columnas_cat = config.CATEGORICAS_FIJAS + [col_cat_val]

    return (X_train, train[columna_demanda],
            X_val, val[columna_demanda],
            X_test, test[columna_demanda],
            columnas_cat)


def entrenar_y_evaluar_fold(df: pd.DataFrame, fold: int, columna_demanda: str,
                             fecha_corte, test_inicio, test_fin) -> dict:
    X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat = preparar_datos_fold(
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

    pred = modelo.predict(X_test)
    mae = np.abs(y_test.values - pred).mean()

    return {
        "fold": fold, "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "mejor_iteracion": modelo.best_iteration_, "mae": mae,
        "_modelo": modelo,  # se antepone "_" para que no aparezca en tablas exportadas a CSV
    }


def evaluar_todos_los_folds(df: pd.DataFrame, metadata: pd.DataFrame, columna_demanda: str) -> pd.DataFrame:
    resultados = []
    for _, fila in metadata.iterrows():
        fold = int(fila["fold"])
        r = entrenar_y_evaluar_fold(df, fold, columna_demanda, fila["fecha_corte"], fila["test_inicio"], fila["test_fin"])
        del r["_modelo"]
        resultados.append(r)
    return pd.DataFrame(resultados)
