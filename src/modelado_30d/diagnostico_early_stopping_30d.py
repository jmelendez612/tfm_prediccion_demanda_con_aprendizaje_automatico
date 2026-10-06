"""
Diagnóstico de early stopping (30 días, enfoque directo).

Reentrena los 6 folds x 2 targets con la configuración de config_30d.py y
guarda la curva de MAE de validación por iteración. Es una auditoría, no
forma parte del pipeline normal.

Salidas (en DIRECTORIO_SALIDA):
  - curvas_early_stopping_directo.csv: MAE de validación por (target, fold, iteración).
  - curvas_early_stopping_resumen.csv: % de iteraciones con aumento del MAE y
    ruido máximo tras el mejor punto, por (target, fold).

Uso:  python -m src.modelado_30d.diagnostico_early_stopping_30d
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from src.modelado import preparacion
from src.modelado_30d import config_30d as config
from src.modelado_30d import rolling_windows_30d as rw30
from src.modelado_30d import baseline_30d as base30
from src.modelado_30d import lightgbm_model_30d as lgbm30
from src.featuring import DataFeaturing
from src import rutas

DIRECTORIO_SALIDA = rutas.DIR_RESULTADOS_30D


def main():
    filas = []

    for target in config.TARGETS_30D:
        print(f"=== {target.nombre} ===")
        df = preparacion.cargar_dataset(target.ruta_csv)
        fe = DataFeaturing()
        df, metadata = rw30.procesar(df, target.columna_demanda, fe.clasificar_intermitencia)
        df = base30.cruzar_y_calcular_baseline_30d(df)

        for _, fila in metadata.iterrows():
            fold = int(fila["fold"])
            X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat = lgbm30.preparar_datos_fold(
                df, fold, fila["fecha_corte_train"], fila["test_inicio"], fila["test_fin"]
            )
            evals_result = {}
            modelo = lgb.LGBMRegressor(**config.PARAMS_LGBM)
            modelo.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mae",
                categorical_feature=columnas_cat,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.record_evaluation(evals_result),
                ],
            )
            mae_hist = evals_result["valid_0"]["l1"]
            best = modelo.best_iteration_
            for it, mae in enumerate(mae_hist, start=1):
                filas.append({
                    "target": target.nombre, "fold": fold, "iteracion": it,
                    "mae_val": mae, "best_iteration": best,
                })
            print(f"  fold {fold}: best_iteration={best} n_rounds_ran={len(mae_hist)}")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(DIRECTORIO_SALIDA / "curvas_early_stopping_directo.csv", index=False)
    print(f"\nGuardado: {len(tabla)} filas -> curvas_early_stopping_directo.csv")

    resumen = []
    for (t, f), g in tabla.groupby(["target", "fold"]):
        g = g.sort_values("iteracion")
        mae = g["mae_val"].values
        best_it = int(g["best_iteration"].iloc[0])
        subidas = sum(1 for i in range(1, len(mae)) if mae[i] > mae[i - 1])
        mae_best = g.loc[g["iteracion"] == best_it, "mae_val"].values[0]
        cola = g[g["iteracion"] > best_it]
        max_cola = cola["mae_val"].max() if len(cola) else mae_best
        resumen.append({
            "target": t, "fold": f, "n_iter": len(mae),
            "pct_subidas_iter_a_iter": round(100 * subidas / (len(mae) - 1), 1),
            "ruido_max_ventana_paciencia_pct": round(100 * (max_cola / mae_best - 1), 1),
        })
    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(DIRECTORIO_SALIDA / "curvas_early_stopping_resumen.csv", index=False)
    print(f"Guardado: curvas_early_stopping_resumen.csv")
    print(resumen_df.to_string())


if __name__ == "__main__":
    main()
