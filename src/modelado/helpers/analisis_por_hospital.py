"""
EXPERIMENTO ADICIONAL — Análisis del MAE (baseline vs LightGBM) por
hospital, para el modelo de 1 día. No modifica nada de src/modelado/
existente -- reutiliza sus módulos tal cual.

Objetivo: confirmar si la mejora de LightGBM es pareja en los 20
hospitales de la red 05, o se concentra en unos pocos.

Uso:
    python analisis_por_hospital.py
"""
import sys
from pathlib import Path
RAIZ_PROYECTO = Path(__file__).resolve().parents[3]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.modelado import preparacion, rolling_windows as rw, baseline, lightgbm_model as lgbm1d, config as config1d
from src.featuring import DataFeaturing
from src import rutas

DIRECTORIO_BASE = rutas.DIR_RESULTADOS_1D

def procesar_fold(df, fold, columna_demanda, fecha_corte, test_inicio, test_fin):
    X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat = lgbm1d.preparar_datos_fold(
        df, fold, columna_demanda, fecha_corte, test_inicio, test_fin
    )

    modelo = lgb.LGBMRegressor(**config1d.PARAMS_LGBM)
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=columnas_cat,
        callbacks=[lgb.early_stopping(stopping_rounds=config1d.EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    pred_lgbm = modelo.predict(X_test)

    # HOSPITAL_ID y QTY_FORECAST (baseline) no son features de entrenamiento,
    # se recuperan del df original con los mismos índices del test
    hospital_test = df.loc[X_test.index, "HOSPITAL_ID"]
    qty_forecast_test = df.loc[X_test.index, "QTY_FORECAST"]

    detalle = pd.DataFrame({
        "hospital": hospital_test.values,
        "actual": y_test.values,
        "pred_baseline": qty_forecast_test.values,
        "pred_lightgbm": pred_lgbm,
    })
    detalle = detalle.dropna(subset=["pred_baseline"])  # SIN_BLOQUES_VALIDOS -> excluir
    detalle["error_baseline"] = (detalle["actual"] - detalle["pred_baseline"]).abs()
    detalle["error_lightgbm"] = (detalle["actual"] - detalle["pred_lightgbm"]).abs()
    detalle["fold"] = fold
    return detalle


def main():
    for target in config1d.TARGETS:
        print(f"\n{'='*70}")
        print(f"ANÁLISIS POR HOSPITAL — target: {target.nombre}")
        print(f"{'='*70}")

        df = preparacion.cargar_dataset(target.ruta_csv)
        fe = DataFeaturing()
        df, metadata = rw.procesar(df, target.columna_demanda, fe.clasificar_intermitencia)
        df = baseline.cruzar_cpfr_real(df)

        detalles = []
        for _, fila in metadata.iterrows():
            fold = int(fila["fold"])
            print(f"  Procesando fold {fold}...")
            detalles.append(procesar_fold(
                df, fold, target.columna_demanda, fila["fecha_corte"], fila["test_inicio"], fila["test_fin"]
            ))

        todo = pd.concat(detalles, ignore_index=True)
        resumen = todo.groupby("hospital").agg(
            n_filas=("actual", "size"),
            media_demanda_real=("actual", "mean"),
            mae_baseline=("error_baseline", "mean"),
            mae_lightgbm=("error_lightgbm", "mean"),
        ).round(2)
        resumen["mejora_pct"] = (100 * (1 - resumen["mae_lightgbm"] / resumen["mae_baseline"])).round(1)
        resumen = resumen.sort_values("mejora_pct")

        print(f"\nMAE por hospital (ordenado de peor a mejor mejora):")
        print(resumen.to_string())

        n_hospitales_pierde = (resumen["mejora_pct"] < 0).sum()
        print(f"\nHospitales donde LightGBM NO supera al baseline: {n_hospitales_pierde} de {len(resumen)}")

        resumen.to_csv(DIRECTORIO_BASE/f"analisis_hospital_{target.nombre}.csv")
        print(f"Guardado en analisis_hospital_{target.nombre}.csv")


if __name__ == "__main__":
    main()