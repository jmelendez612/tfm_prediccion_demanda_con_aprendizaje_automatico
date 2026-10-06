"""
Desglose por categoría de intermitencia para el experimento de residual
learning (30 días) -- para confirmar si la mejora está repartida en
erratic/lumpy (a diferencia del enfoque directo, donde esas categorías
perdían) o si sigue sostenida solo por intermittent.

No modifica nada existente -- reutiliza residual_learning_30d.py.

Uso:
    python desglose_residual_30d.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.modelado import preparacion
from src.modelado_30d import config_30d as config
from src.modelado_30d import rolling_windows_30d as rw30
from src.modelado_30d import baseline_30d as base30
from src.featuring import DataFeaturing
from src.modelado_30d.residual_learning_30d import preparar_datos_fold_residual

from pathlib import Path
from src import rutas
DIRECTORIO_SALIDA = rutas.DIR_RESULTADOS_30D

def procesar_fold(df, fold, fecha_corte_train, test_inicio, test_fin):
    train, val, test, columnas_features, columnas_cat = preparar_datos_fold_residual(
        df, fold, fecha_corte_train, test_inicio, test_fin
    )

    modelo = lgb.LGBMRegressor(**config.PARAMS_LGBM)
    modelo.fit(
        train[columnas_features], train["residual_30d"],
        eval_set=[(val[columnas_features], val["residual_30d"])],
        eval_metric="mae",
        categorical_feature=columnas_cat,
        callbacks=[lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    residuo_predicho = modelo.predict(test[columnas_features])
    prediccion_final = test["baseline_30d"].values + residuo_predicho

    #col_clasif = f"{config.CATEGORICA_POR_FOLD}_fold{fold}"
    col_clasif = f"{config.CATEGORICA_POR_FOLD}_val_fold{fold}"  # Renombrada durante preparar_datos_fold_residual

    detalle = pd.DataFrame({
        "categoria": test[col_clasif].values,
        "actual": test["target_30d"].values,
        "pred_baseline": test["baseline_30d"].values,
        "pred_residual_learning": prediccion_final,
    })
    detalle["error_baseline"] = (detalle["actual"] - detalle["pred_baseline"]).abs()
    detalle["error_residual_learning"] = (detalle["actual"] - detalle["pred_residual_learning"]).abs()
    detalle["fold"] = fold
    return detalle


def main():
    for target in config.TARGETS_30D:
        print(f"\n{'='*70}")
        print(f"DESGLOSE POR CATEGORÍA — Residual Learning (30 días) — target: {target.nombre}")
        print(f"{'='*70}")

        df = preparacion.cargar_dataset(target.ruta_csv)
        fe = DataFeaturing()
        df, metadata = rw30.procesar(df, target.columna_demanda, fe.clasificar_intermitencia)
        df = base30.cruzar_y_calcular_baseline_30d(df)
        df["residual_30d"] = df["target_30d"] - df["baseline_30d"]

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
            mae_residual_learning=("error_residual_learning", "mean"),
        ).round(2)
        resumen["mejora_pct"] = (100 * (1 - resumen["mae_residual_learning"] / resumen["mae_baseline"])).round(1)

        print(resumen.to_string())
        resumen.to_csv(DIRECTORIO_SALIDA / f"desglose_residual_30d_categoria_{target.nombre}.csv")
        print(f"Guardado en desglose_residual_30d_categoria_{target.nombre}.csv")


if __name__ == "__main__":
    main()
