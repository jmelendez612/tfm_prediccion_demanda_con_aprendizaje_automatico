"""
Residual Learning para el horizonte de 30 días.

LightGBM predice el residual (target_30d - baseline_30d) y la predicción
final es baseline_30d + residual predicho. Reutiliza los módulos existentes.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.modelado import preparacion
from src.modelado_30d import config_30d as config
from src.modelado_30d import rolling_windows_30d as rw30
from src.modelado_30d import baseline_30d as base30
from src.featuring import DataFeaturing

from pathlib import Path
from src import rutas
DIRECTORIO_SALIDA = rutas.DIR_RESULTADOS_30D

def preparar_datos_fold_residual(df: pd.DataFrame, fold: int, fecha_corte_train, test_inicio, test_fin):
    """
    Igual que lgbm30.preparar_datos_fold, pero el target es 'residual_30d'
    (target_30d - baseline_30d) en vez de target_30d directo.
    Aplica las fotografías dinámicas (_val para train/val y _test para test)
    generadas por la versión corregida de rolling_windows_30d.py.
    """
    sufijo_val = f"_val_fold{fold}"
    sufijo_test = f"_test_fold{fold}"

    col_clasif_val = f"{config.CATEGORICA_POR_FOLD}{sufijo_val}"
    col_clasif_test = f"{config.CATEGORICA_POR_FOLD}{sufijo_test}"

    cols_num_val = [f"{c}{sufijo_val}" for c in config.NUMERICAS_POR_FOLD]
    cols_num_test = [f"{c}{sufijo_test}" for c in config.NUMERICAS_POR_FOLD]

    features_val = config.CATEGORICAS_FIJAS + [col_clasif_val] + config.NUMERICAS_FIJAS + cols_num_val
    features_test = config.CATEGORICAS_FIJAS + [col_clasif_test] + config.NUMERICAS_FIJAS + cols_num_test

    # Excluimos columnas faltantes si no existen en el df
    faltantes_val = [c for c in features_val if c not in df.columns]
    faltantes_test = [c for c in features_test if c not in df.columns]

    features_val = [c for c in features_val if c not in faltantes_val]
    features_test = [c for c in features_test if c not in faltantes_test]

    cols_necesarias = list(set(["FECHA", "target_30d", "baseline_30d", "residual_30d"] + features_val + features_test))
    df_fold = df[df["FECHA"] <= test_fin][cols_necesarias].copy()
    df_fold = df_fold.dropna(subset=["residual_30d"])

    # Conversión a categóricas
    if col_clasif_val in df_fold.columns:
        df_fold[col_clasif_val] = df_fold[col_clasif_val].astype("category")
    if col_clasif_test in df_fold.columns:
        df_fold[col_clasif_test] = df_fold[col_clasif_test].astype("category")

    for col in config.CATEGORICAS_FIJAS:
        if col in df_fold.columns:
            df_fold[col] = df_fold[col].astype("category")

    val_inicio = fecha_corte_train - pd.DateOffset(months=1)
    # EMBARGO: mismo criterio que lightgbm_model_30d.preparar_datos_fold --
    # se retira de train el margen de VENTANA_DIAS inmediatamente anterior
    # a val_inicio, porque target_30d (y por tanto residual_30d) de esas
    # filas usa demanda que cae dentro de la propia ventana de validación.
    train_fin_embargo = val_inicio - pd.Timedelta(days=config.VENTANA_DIAS)
    train = df_fold[df_fold["FECHA"] < train_fin_embargo]
    val = df_fold[(df_fold["FECHA"] >= val_inicio) & (df_fold["FECHA"] < fecha_corte_train)]
    test = df_fold[(df_fold["FECHA"] >= test_inicio) & (df_fold["FECHA"] <= test_fin)]

    # Mapeo para homologar nombres de columnas entre train/val y test
    mapa_renombre = {col_clasif_test: col_clasif_val}
    for c_test, c_val in zip(cols_num_test, cols_num_val):
        mapa_renombre[c_test] = c_val

    test_aligned = test.copy()
    # Evitar columnas duplicadas: quitamos las _val_fold antes de renombrar
    # las _test_fold, ya que ambas coexisten en df_fold.
    cols_a_eliminar = [c for c in ([col_clasif_val] + cols_num_val) if c in test_aligned.columns]
    test_aligned = test_aligned.drop(columns=cols_a_eliminar)
    test_aligned = test_aligned.rename(columns=mapa_renombre)

    columnas_cat = [c for c in config.CATEGORICAS_FIJAS if c in df_fold.columns] + [col_clasif_val]

    return train, val, test_aligned, features_val, columnas_cat


def entrenar_y_evaluar_fold_residual(df, fold, fecha_corte_train, test_inicio, test_fin) -> dict:
    train, val, test, columnas_features, columnas_cat = preparar_datos_fold_residual(
        df, fold, fecha_corte_train, test_inicio, test_fin
    )

    modelo = lgb.LGBMRegressor(**config.PARAMS_LGBM)  # mismos hiperparámetros ya tuneados
    modelo.fit(
        train[columnas_features], train["residual_30d"],
        eval_set=[(val[columnas_features], val["residual_30d"])],
        eval_metric="mae",
        categorical_feature=columnas_cat,
        callbacks=[lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    residuo_predicho = modelo.predict(test[columnas_features])
    prediccion_final = test["baseline_30d"].values + residuo_predicho

    mae = np.abs(test["target_30d"].values - prediccion_final).mean()
    mae_baseline_solo = np.abs(test["target_30d"].values - test["baseline_30d"].values).mean()

    return {
        "fold": fold, "n_train": len(train), "n_test": len(test),
        "mae_residual_learning": mae, "mae_baseline_en_este_fold": mae_baseline_solo,
        "mejora_pct": 100 * (1 - mae / mae_baseline_solo),
    }


def main():
    for target in config.TARGETS_30D:
        print(f"\n{'='*70}")
        print(f"RESIDUAL LEARNING (30 días) — target: {target.nombre}")
        print(f"{'='*70}")

        df = preparacion.cargar_dataset(target.ruta_csv)
        fe = DataFeaturing()
        df, metadata = rw30.procesar(df, target.columna_demanda, fe.clasificar_intermitencia)
        df = base30.cruzar_y_calcular_baseline_30d(df)
        df["residual_30d"] = df["target_30d"] - df["baseline_30d"]

        resultados = []
        for _, fila in metadata.iterrows():
            fold = int(fila["fold"])
            print(f"  Procesando fold {fold}...")
            r = entrenar_y_evaluar_fold_residual(df, fold, fila["fecha_corte_train"], fila["test_inicio"], fila["test_fin"])
            resultados.append(r)
            print(f"    MAE residual learning = {r['mae_residual_learning']:.2f} | "
                  f"MAE baseline = {r['mae_baseline_en_este_fold']:.2f} | mejora = {r['mejora_pct']:.1f}%")

        tabla = pd.DataFrame(resultados)
        mae_final = tabla["mae_residual_learning"].mean()
        mae_baseline_final = tabla["mae_baseline_en_este_fold"].mean()
        mejora_final = 100 * (1 - mae_final / mae_baseline_final)

        print(f"\n{'-'*70}")
        print(f"MAE promedio residual learning: {mae_final:.2f}")
        print(f"MAE promedio baseline:          {mae_baseline_final:.2f}")
        print(f"Mejora: {mejora_final:.1f}%")
        print(f"{'-'*70}")

        tabla.to_csv(DIRECTORIO_SALIDA / f"residual_learning_30d_resultados_{target.nombre}.csv", index=False)
        print(f"Guardado en residual_learning_30d_resultados_{target.nombre}.csv")


if __name__ == "__main__":
    main()