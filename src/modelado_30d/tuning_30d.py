"""
Búsqueda aleatoria de hiperparámetros de LightGBM para 30 días.

Se usa el fold 1, el único cuya ventana de validación no se solapa con ningún
test reportado. Evalúa siempre sobre validación, nunca sobre test.

Uso:  python -m src.modelado_30d.tuning_30d
"""

from pathlib import Path

import random
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.modelado import preparacion
from src.modelado_30d import config_30d as config
from src.modelado_30d import rolling_windows_30d as rw30
from src.modelado_30d import baseline_30d as base30
from src.modelado_30d import lightgbm_model_30d as lgbm30
from src.featuring import DataFeaturing
from src import rutas

DIRECTORIO_SALIDA = rutas.DIR_RESULTADOS_30D

TARGET = config.TARGETS_30D[0]  # MEDIANA -- tuneamos sobre este, asumimos que transfiere razonablemente a SBA
FOLD_PARA_TUNING = 1  # único fold cuya validación no se solapa con ningún test reportado (ver docstring)

random.seed(42)

GRILLA = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "min_child_samples": [20, 50, 100, 200],
    "max_depth": [-1, 4, 6, 8],
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
}

N_COMBINACIONES = 15


def muestrear_combinacion():
    combo = {k: random.choice(v) for k, v in GRILLA.items()}
    combo["objective"] = "mae"
    combo["n_estimators"] = 2000
    combo["random_state"] = 42
    combo["verbosity"] = -1
    if combo["bagging_fraction"] < 1.0:
        combo["bagging_freq"] = 1  # requerido por LightGBM para que bagging_fraction tenga efecto
    return combo


def evaluar_combinacion(X_train, y_train, X_val, y_val, columnas_cat, params):
    modelo = lgb.LGBMRegressor(**params)
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=columnas_cat,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    pred = modelo.predict(X_val)
    return np.abs(y_val.values - pred).mean(), modelo.best_iteration_


def main():
    print(f"Cargando {TARGET.ruta_csv}...")
    df = preparacion.cargar_dataset(TARGET.ruta_csv)

    print("Calculando target_30d y folds...")
    fe = DataFeaturing()
    df, metadata = rw30.procesar(df, TARGET.columna_demanda, fe.clasificar_intermitencia)

    # preparar_datos_fold exige baseline_30d para armar el test
    print("Calculando baseline CPFR (30 días)...")
    df = base30.cruzar_y_calcular_baseline_30d(df)

    fila = metadata[metadata["fold"] == FOLD_PARA_TUNING].iloc[0]
    print(f"\nUsando fold {FOLD_PARA_TUNING} para la búsqueda "
          f"(train < {fila['fecha_corte_train'].date()})")

    X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat = lgbm30.preparar_datos_fold(
        df, FOLD_PARA_TUNING, fila["fecha_corte_train"], fila["test_inicio"], fila["test_fin"]
    )
    print(f"n_train={len(X_train):,} | n_val={len(X_val):,}\n")

    resultados = []
    for i in range(N_COMBINACIONES):
        params = muestrear_combinacion()
        mae_val, mejor_iter = evaluar_combinacion(X_train, y_train, X_val, y_val, columnas_cat, params)
        resultados.append({**params, "mae_val": mae_val, "mejor_iteracion": mejor_iter})
        print(f"[{i+1}/{N_COMBINACIONES}] MAE val = {mae_val:.4f} | "
              f"num_leaves={params['num_leaves']} lr={params['learning_rate']} "
              f"min_child={params['min_child_samples']} max_depth={params['max_depth']} "
              f"feat_frac={params['feature_fraction']} bag_frac={params['bagging_fraction']}")

    tabla = pd.DataFrame(resultados).sort_values("mae_val")
    tabla.to_csv(DIRECTORIO_SALIDA / "tuning_30d_resultados.csv", index=False)

    print(f"\n{'='*70}")
    print("MEJOR COMBINACIÓN ENCONTRADA:")
    print(f"{'='*70}")
    mejor = tabla.iloc[0]
    print(mejor.to_string())
    print(f"\nResultados completos guardados en 'tuning_30d_resultados.csv'")


if __name__ == "__main__":
    main()
