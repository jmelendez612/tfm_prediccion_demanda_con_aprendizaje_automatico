"""
Paso 4 — Modelo LightGBM Nivel 2 a 30 días.

Corrige de raíz el conflicto de columnas duplicadas por colisión con el dataset base,
asegurando que X_train, X_val y X_test sean matrices con Series de 1 dimensión.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config_30d as config


def preparar_datos_fold(df: pd.DataFrame, fold: int, fecha_corte_train, test_inicio, test_fin):
    sufijo_val = f"_val_fold{fold}"
    sufijo_test = f"_test_fold{fold}"

    col_clasif_val = f"{config.CATEGORICA_POR_FOLD}{sufijo_val}"
    col_clasif_test = f"{config.CATEGORICA_POR_FOLD}{sufijo_test}"

    cols_num_val = [f"{c}{sufijo_val}" for c in config.NUMERICAS_POR_FOLD]
    cols_num_test = [f"{c}{sufijo_test}" for c in config.NUMERICAS_POR_FOLD]

    col_clasif_std = config.CATEGORICA_POR_FOLD
    val_inicio = fecha_corte_train - pd.DateOffset(months=1)
    # EMBARGO: como target_30d de una fila con fecha D usa demanda de
    # D+1...D+30, las filas de train con fecha inmediatamente anterior a
    # val_inicio tendrían etiquetas que caen dentro de la propia ventana
    # de validación. Se retira ese margen de train, igual que ya se hace
    # entre train y test vía fecha_corte_train (ver rolling_windows_30d.py).
    train_fin_embargo = val_inicio - pd.Timedelta(days=config.VENTANA_DIAS)

    # 1. TRAIN & VAL (usan la fotografía _val)
    df_train_val = df[(df["FECHA"] <= fecha_corte_train) & df["target_30d"].notna()].copy()
    
    # ELIMINAR del DF original la columna base si ya existía para evitar duplicación tras el rename
    if col_clasif_std in df_train_val.columns:
        df_train_val = df_train_val.drop(columns=[col_clasif_std])
    for c_base in config.NUMERICAS_POR_FOLD:
        if c_base in df_train_val.columns:
            df_train_val = df_train_val.drop(columns=[c_base])

    # Renombrado seguro para TRAIN / VAL
    mapa_val = {col_clasif_val: col_clasif_std}
    for c_val, c_base in zip(cols_num_val, config.NUMERICAS_POR_FOLD):
        mapa_val[c_val] = c_base
    
    df_train_val = df_train_val.rename(columns=mapa_val)

    # 2. TEST (usa la fotografía _test)
    # Se exige también baseline_30d.notna() para que la muestra evaluada
    # coincida exactamente con la del baseline (mismo criterio aplicado
    # en el modelo de 1 día, ver src/modelado/lightgbm_model.py).
    df_test = df[(df["FECHA"] >= test_inicio) & (df["FECHA"] <= test_fin)
                 & df["target_30d"].notna() & df["baseline_30d"].notna()].copy()
    
    if col_clasif_std in df_test.columns:
        df_test = df_test.drop(columns=[col_clasif_std])
    for c_base in config.NUMERICAS_POR_FOLD:
        if c_base in df_test.columns:
            df_test = df_test.drop(columns=[c_base])

    # Renombrado seguro para TEST
    mapa_test = {col_clasif_test: col_clasif_std}
    for c_test, c_base in zip(cols_num_test, config.NUMERICAS_POR_FOLD):
        mapa_test[c_test] = c_base

    df_test = df_test.rename(columns=mapa_test)

    # Construcción de la lista de features
    raw_features = config.CATEGORICAS_FIJAS + [col_clasif_std] + config.NUMERICAS_FIJAS + config.NUMERICAS_POR_FOLD
    features_std = []
    for c in raw_features:
        if c in df_train_val.columns and c in df_test.columns and c not in features_std:
            features_std.append(c)

    # Subsets finales
    train = df_train_val[df_train_val["FECHA"] < train_fin_embargo]
    val = df_train_val[(df_train_val["FECHA"] >= val_inicio) & (df_train_val["FECHA"] < fecha_corte_train)]
    test = df_test

    X_train = train[features_std].copy()
    y_train = train["target_30d"]

    X_val = val[features_std].copy()
    y_val = val["target_30d"]

    X_test = test[features_std].copy()
    y_test = test["target_30d"]

    # Deduplicación preventiva en los DataFrames finales si persistieran etiquetas duplicadas en el index de columnas
    X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()
    X_val = X_val.loc[:, ~X_val.columns.duplicated()].copy()
    X_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()

    # Identificar e imponer categoricas sobre Series garantizadas
    columnas_cat = [c for c in config.CATEGORICAS_FIJAS if c in X_train.columns]
    if col_clasif_std in X_train.columns and col_clasif_std not in columnas_cat:
        columnas_cat.append(col_clasif_std)

    for col in columnas_cat:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    return X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat


def evaluar_fold(df: pd.DataFrame, fold: int, fecha_corte_train, test_inicio, test_fin) -> dict:
    X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat = preparar_datos_fold(
        df, fold, fecha_corte_train, test_inicio, test_fin
    )

    modelo = lgb.LGBMRegressor(**config.PARAMS_LGBM)
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    pred_test = modelo.predict(X_test)
    mae = np.mean(np.abs(y_test - pred_test))

    return {
        "fold": fold,
        "n_filas_evaluadas": len(X_test),
        "mae": mae,
        "best_iteration": modelo.best_iteration_,
    }


def evaluar_todos_los_folds(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    resultados = []
    for _, fila in metadata.iterrows():
        r = evaluar_fold(
            df,
            int(fila["fold"]),
            fila["fecha_corte_train"],
            fila["test_inicio"],
            fila["test_fin"]
        )
        resultados.append(r)
    return pd.DataFrame(resultados)