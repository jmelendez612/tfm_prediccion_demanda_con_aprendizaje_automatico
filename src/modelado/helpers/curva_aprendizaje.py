"""
Curva de aprendizaje del modelo diario (fold 6).

Entrena con fracciones crecientes de combinaciones producto x hospital
(20/40/60/80/100 %, muestreadas por combinación, no por fila) y evalúa
siempre contra el mismo bloque de test.
"""
import sys
from pathlib import Path
RAIZ_PROYECTO = Path(__file__).resolve().parents[3]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.modelado import preparacion, rolling_windows as rw, baseline, lightgbm_model as lgbm1d, config as config1d
from src.featuring import DataFeaturing
from src import rutas

FRACCIONES = [0.2, 0.4, 0.6, 0.8, 1.0]
FOLD_ANALIZADO = 6
SEMILLA = 42
DIRECTORIO_BASE = rutas.DIR_RESULTADOS_1D

def evaluar_con_fraccion(X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat,
                          skus_train: pd.Series, fraccion: float, rng) -> dict:
    skus_unicos = skus_train.unique()
    n_sample = max(1, int(len(skus_unicos) * fraccion))
    skus_elegidos = rng.choice(skus_unicos, size=n_sample, replace=False)
    mascara = skus_train.isin(skus_elegidos)

    modelo = lgb.LGBMRegressor(**config1d.PARAMS_LGBM)
    modelo.fit(
        X_train[mascara.values], y_train[mascara.values],
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=columnas_cat,
        callbacks=[lgb.early_stopping(stopping_rounds=config1d.EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    pred = modelo.predict(X_test)
    mae = np.abs(y_test.values - pred).mean()

    return {"fraccion": fraccion, "n_skus": n_sample, "n_filas_train": int(mascara.sum()), "mae": mae}


def main():
    resultados_todos = []

    for target in config1d.TARGETS:
        print(f"\n{'='*70}")
        print(f"CURVA DE APRENDIZAJE — target: {target.nombre}")
        print(f"{'='*70}")

        df = preparacion.cargar_dataset(target.ruta_csv)
        fe = DataFeaturing()
        df, metadata = rw.procesar(df, target.columna_demanda, fe.clasificar_intermitencia)
        df = baseline.cruzar_cpfr_real(df)

        fila = metadata[metadata["fold"] == FOLD_ANALIZADO].iloc[0]
        X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat = lgbm1d.preparar_datos_fold(
            df, FOLD_ANALIZADO, target.columna_demanda, fila["fecha_corte"], fila["test_inicio"], fila["test_fin"]
        )

        # Necesitamos el PRODUCTO_ID x HOSPITAL_ID de cada fila de train
        # para poder muestrear por SKU, no por fila
        skus_train = df.loc[X_train.index, "PRODUCTO_ID"].astype(str) + "_" + df.loc[X_train.index, "HOSPITAL_ID"].astype(str)

        rng = np.random.RandomState(SEMILLA)
        resultados = []
        for frac in FRACCIONES:
            r = evaluar_con_fraccion(X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat,
                                      skus_train, frac, rng)
            r["target"] = target.nombre
            resultados.append(r)
            print(f"  {int(frac*100)}% de SKUs ({r['n_skus']} SKUs, {r['n_filas_train']:,} filas): "
                  f"MAE = {r['mae']:.2f}")

        resultados_todos.extend(resultados)
        tabla_target = pd.DataFrame(resultados)
        tabla_target.to_csv(DIRECTORIO_BASE/f"curva_aprendizaje_{target.nombre}.csv", index=False)

    tabla_completa = pd.DataFrame(resultados_todos)
    tabla_completa.to_csv(DIRECTORIO_BASE/"curva_aprendizaje_completa.csv", index=False)

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    for target_nombre, grupo in tabla_completa.groupby("target"):
        ax.plot(grupo["n_skus"], grupo["mae"], marker="o", label=target_nombre)
    ax.set_xlabel("Cantidad de SKU x Hospital en el entrenamiento")
    ax.set_ylabel("MAE en el bloque de test (fold 6)")
    ax.set_title("Curva de aprendizaje — Modelo de 1 día")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DIRECTORIO_BASE/"curva_aprendizaje.png", dpi=150)
    print("\nGuardado: curva_aprendizaje_MEDIANA.csv, curva_aprendizaje_SBA.csv, "
          "curva_aprendizaje_completa.csv, curva_aprendizaje.png")


if __name__ == "__main__":
    main()