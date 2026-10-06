"""
Importancia de variables y SHAP del modelo diario.

1. Importancia nativa de LightGBM (tipo split), promediada en los 6 folds.
2. SHAP sobre el fold 6.
"""
import sys
import re
from pathlib import Path
RAIZ_PROYECTO = Path(__file__).resolve().parents[3]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

from src.modelado import (preparacion, rolling_windows as rw, baseline, lightgbm_model as lgbm1d, config as config1d)
from src.featuring import DataFeaturing
from src import rutas

DIRECTORIO_BASE = rutas.DIR_RESULTADOS_1D

def calcular_importancia_gain(df, metadata, columna_demanda):
    """
    Entrena los 6 folds y promedia la importancia 'gain' entre ellos.

    IMPORTANTE: adi_foldN, cv2_foldN, clasificacion_demanda_foldN y
    pct_censura_historica_foldN son columnas DISTINTAS en cada fold
    (cada modelo usa la que le corresponde a su propio corte). Antes de
    promediar, se renombran a un nombre genérico (adi, cv2,
    clasificacion_demanda, pct_censura_historica) para que representen
    la MISMA variable conceptual medida en los 6 folds -- sin este
    renombrado, pandas promedia con NaN en 5 de 6 folds para cada una,
    lo que mostraba el valor de un solo fold como si fuera un promedio.
    """
    importancias_por_fold = []
    modelo_final, X_test_final = None, None

    def normalizar_nombre(col):
        return re.sub(r"_fold\d+$", "", col)

    for _, fila in metadata.iterrows():
        fold = int(fila["fold"])
        X_train, y_train, X_val, y_val, X_test, y_test, columnas_cat = lgbm1d.preparar_datos_fold(
            df, fold, columna_demanda, fila["fecha_corte"], fila["test_inicio"], fila["test_fin"]
        )

        modelo = lgb.LGBMRegressor(**config1d.PARAMS_LGBM)
        modelo.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            categorical_feature=columnas_cat,
            callbacks=[lgb.early_stopping(stopping_rounds=config1d.EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        nombres_normalizados = [normalizar_nombre(c) for c in X_train.columns]
        importancia = pd.Series(modelo.feature_importances_, index=nombres_normalizados, name=f"fold_{fold}")
        importancias_por_fold.append(importancia)

        if fold == int(metadata["fold"].max()):
            modelo_final, X_test_final = modelo, X_test

    tabla = pd.concat(importancias_por_fold, axis=1)
    tabla["promedio"] = tabla.mean(axis=1)
    tabla = tabla.sort_values("promedio", ascending=False)

    return tabla, modelo_final, X_test_final


def calcular_shap(modelo, X_test, top_n=15):
    """SHAP sobre el fold con más datos de entrenamiento (el último)."""
    import re
    explainer = shap.TreeExplainer(modelo)
    muestra = X_test.sample(min(3000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(muestra)

    nombres_normalizados = [re.sub(r"_fold\d+$", "", c) for c in muestra.columns]
    importancia_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=nombres_normalizados, name="importancia_shap_abs"
    ).sort_values(ascending=False)

    return importancia_shap.head(top_n), shap_values, muestra


def main():
    for target in config1d.TARGETS:
        print(f"\n{'='*70}")
        print(f"IMPORTANCIA DE VARIABLES — target: {target.nombre}")
        print(f"{'='*70}")

        df = preparacion.cargar_dataset(target.ruta_csv)
        fe = DataFeaturing()
        df, metadata = rw.procesar(df, target.columna_demanda, fe.clasificar_intermitencia)
        df = baseline.cruzar_cpfr_real(df)

        print("Entrenando los 6 folds para promediar importancia...")
        tabla_gain, modelo_final, X_test_final = calcular_importancia_gain(df, metadata, target.columna_demanda)

        print(f"\nTop 15 features por importancia 'gain' (promedio de 6 folds):")
        print(tabla_gain[["promedio"]].head(15).to_string())
        tabla_gain.to_csv(DIRECTORIO_BASE/f"importancia_gain_{target.nombre}.csv")

        print(f"\nCalculando SHAP sobre el fold 6 (más historia de entrenamiento)...")
        importancia_shap, shap_values, muestra = calcular_shap(modelo_final, X_test_final)
        print(f"\nTop 15 features por importancia SHAP (|valor SHAP| promedio):")
        print(importancia_shap.to_string())
        importancia_shap.to_csv(DIRECTORIO_BASE/f"importancia_shap_{target.nombre}.csv")

        np.save(DIRECTORIO_BASE/f"shap_values_{target.nombre}.npy", shap_values)
        muestra.to_csv(DIRECTORIO_BASE/f"shap_muestra_features_{target.nombre}.csv", index=False)

        print(f"\nGuardado: importancia_gain_{target.nombre}.csv, "
              f"importancia_shap_{target.nombre}.csv, shap_values_{target.nombre}.npy")


if __name__ == "__main__":
    main()