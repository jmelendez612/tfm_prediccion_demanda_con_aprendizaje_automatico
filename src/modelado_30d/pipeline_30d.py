"""
Modelado a 30 días: ejecuta los pasos para MEDIANA y SBA y guarda los resultados.
Requiere los datasets extendidos hasta junio de 2026.

Uso:  python -m src.modelado_30d.pipeline_30d
"""

from pathlib import Path

import pandas as pd

from src.modelado import preparacion
from src.featuring import DataFeaturing

from . import config_30d as config
from . import rolling_windows_30d as rw30
from . import baseline_30d as base30
from . import lightgbm_model_30d as lgbm30
from src import rutas

DIRECTORIO_SALIDA = rutas.DIR_RESULTADOS_30D


def ejecutar_para_target(target: config.TargetConfig30d) -> dict:
    print(f"\n{'='*70}")
    print(f"TARGET (30 días): {target.nombre}")
    print(f"{'='*70}")

    print("\n[1/4] Preparación...")
    df = preparacion.cargar_dataset(target.ruta_csv)
    print(f"  {len(df):,} filas cargadas, rango {df['FECHA'].min().date()} a {df['FECHA'].max().date()}")

    print("\n[2/4] Rolling Windows 30d (target_30d + folds con corte retrasado)...")
    fe = DataFeaturing()
    df, metadata = rw30.procesar(df, target.columna_demanda, fe.clasificar_intermitencia)
    print(f"  {config.N_CORTES} folds: fold 1 test {metadata.iloc[0]['test_inicio'].date()} ... "
          f"fold {config.N_CORTES} test {metadata.iloc[-1]['test_fin'].date()}")

    print("\n[3/4] Baseline 30d (CPFR real, misma ventana)...")
    df = base30.cruzar_y_calcular_baseline_30d(df)
    resultados_baseline = base30.evaluar_todos_los_folds(df, metadata, target.columna_demanda)
    mae_baseline = resultados_baseline["mae"].mean()
    print(f"  MAE promedio baseline 30d: {mae_baseline:.2f}")

    print("\n[4/4] LightGBM 30d...")
    resultados_lgbm = lgbm30.evaluar_todos_los_folds(df, metadata)
    mae_lgbm = resultados_lgbm["mae"].mean()
    print(f"  MAE promedio LightGBM 30d: {mae_lgbm:.2f} "
          f"(mejora {100*(1-mae_lgbm/mae_baseline):.1f}% sobre el baseline)")

    resultados_baseline.to_csv(DIRECTORIO_SALIDA / f"baseline_30d_resultados_{target.nombre}.csv", index=False)
    resultados_lgbm.to_csv(DIRECTORIO_SALIDA / f"lightgbm_30d_resultados_{target.nombre}.csv", index=False)
    metadata.to_csv(DIRECTORIO_SALIDA / f"folds_metadata_30d_{target.nombre}.csv", index=False)

    return {
        "target": target.nombre, "mae_baseline": mae_baseline, "mae_lightgbm": mae_lgbm,
        "mejora_pct": 100 * (1 - mae_lgbm / mae_baseline),
    }


def main():
    resumen_final = [ejecutar_para_target(t) for t in config.TARGETS_30D]

    print(f"\n{'='*70}")
    print("RESUMEN FINAL (30 días) — todos los targets")
    print(f"{'='*70}")
    print(pd.DataFrame(resumen_final).round(2).to_string(index=False))
    print("\nPipeline de modelado a 30 días completo.")


if __name__ == "__main__":
    main()
