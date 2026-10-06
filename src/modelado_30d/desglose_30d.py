"""
Desglose del MAE (baseline vs LightGBM) por categoría de intermitencia,
para el modelo de 30 días -- mismo propósito que el desglose del modelo
de 1 día: ver si un promedio chico esconde una historia mejor (o peor)
por categoría.

Uso:
    python desglose_30d.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.modelado import preparacion, baseline as base_1d
from src.modelado_30d import config_30d as config
from src.modelado_30d import rolling_windows_30d as rw30
from src.modelado_30d import baseline_30d as base30
from src.modelado_30d import lightgbm_model_30d as lgbm30
from src.featuring import DataFeaturing
from src import rutas

DIRECTORIO_SALIDA = rutas.DIR_RESULTADOS_30D


def procesar_fold(df, fold, fecha_corte_train, test_inicio, test_fin):
    X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat = lgbm30.preparar_datos_fold(
        df, fold, fecha_corte_train, test_inicio, test_fin
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

    col_clasif = config.CATEGORICA_POR_FOLD
    baseline_test = df.loc[X_test.index, "baseline_30d"]

    detalle = pd.DataFrame({
        "categoria": X_test[col_clasif].values,
        "actual": y_test.values,
        "pred_baseline": baseline_test.values,
        "pred_lightgbm": pred_lgbm,
    })
    detalle["error_baseline"] = (detalle["actual"] - detalle["pred_baseline"]).abs()
    detalle["error_lightgbm"] = (detalle["actual"] - detalle["pred_lightgbm"]).abs()
    detalle["fold"] = fold
    return detalle


def main():
    for target in config.TARGETS_30D:
        print(f"\n{'='*70}")
        print(f"DESGLOSE POR CATEGORÍA (30 días) — target: {target.nombre}")
        print(f"{'='*70}")

        df = preparacion.cargar_dataset(target.ruta_csv)
        fe = DataFeaturing()
        df, metadata = rw30.procesar(df, target.columna_demanda, fe.clasificar_intermitencia)
        df = base30.cruzar_y_calcular_baseline_30d(df)

        detalles = []
        for _, fila in metadata.iterrows():
            fold = int(fila["fold"])
            print(f"  Procesando fold {fold}...")
            detalles.append(procesar_fold(df, fold, fila["fecha_corte_train"], fila["test_inicio"], fila["test_fin"]))

        todo = pd.concat(detalles, ignore_index=True)
        resumen = todo.groupby("categoria").agg(
            n_filas=("actual", "size"),
            media_demanda_real=("actual", "mean"),
            mae_baseline=("error_baseline", "mean"),
            mae_lightgbm=("error_lightgbm", "mean"),
        ).round(2)
        resumen["mejora_pct"] = (100 * (1 - resumen["mae_lightgbm"] / resumen["mae_baseline"])).round(1)

        print(resumen.to_string())
        resumen.to_csv(DIRECTORIO_SALIDA / f"desglose_30d_categoria_{target.nombre}.csv")
        print(f"Guardado en desglose_30d_categoria_{target.nombre}.csv")


if __name__ == "__main__":
    main()
