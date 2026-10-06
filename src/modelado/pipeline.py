"""
Orquestador del módulo de modelado. Corre los pasos 1 a 5 de punta a
punta, una vez por cada target definido en config.TARGETS (MEDIANA y SBA
por defecto), y deja los resultados guardados en disco + un resumen
final impreso en pantalla.

Requiere: src/featuring.py (para clasificar_intermitencia) y el CSV de
CPFR_FORECAST_RESULTADO ya exportado (ver config.RUTA_CPFR_REAL).

Uso (desde la raíz del proyecto, junto a main.py):
    python -m src.modelado.pipeline
"""

from pathlib import Path

import pandas as pd

from . import config
from . import preparacion
from . import rolling_windows
from . import baseline
from . import lightgbm_model
from . import desglose
from ..featuring import DataFeaturing
from src import rutas

DIRECTORIO_SALIDA = rutas.DIR_RESULTADOS_1D


def ejecutar_para_target(target: config.TargetConfig) -> dict:
    print(f"\n{'='*70}")
    print(f"TARGET: {target.nombre}")
    print(f"{'='*70}")

    print("\n[1/5] Preparación...")
    df = preparacion.cargar_dataset(target.ruta_csv)
    print(f"  {len(df):,} filas cargadas, rango {df['FECHA'].min().date()} a {df['FECHA'].max().date()}")

    print("\n[2/5] Rolling Windows (recálculo sin fuga por fold)...")
    fe = DataFeaturing()
    df, metadata = rolling_windows.procesar(df, target.columna_demanda, fe.clasificar_intermitencia)
    print(f"  {config.N_CORTES} folds generados: "
          f"fold 1 = test {metadata.iloc[0]['test_inicio'].date()} ... "
          f"fold {config.N_CORTES} = test {metadata.iloc[-1]['test_fin'].date()}")

    print("\n[3/5] Baseline Nivel 1 (CPFR real)...")
    df = baseline.cruzar_cpfr_real(df)
    resultados_baseline = baseline.evaluar_todos_los_folds(df, metadata, target.columna_demanda)
    mae_baseline = resultados_baseline["mae"].mean()
    print(f"  MAE promedio baseline: {mae_baseline:.2f}")

    print("\n[4/5] LightGBM Nivel 2...")
    resultados_lgbm = lightgbm_model.evaluar_todos_los_folds(df, metadata, target.columna_demanda)
    mae_lgbm = resultados_lgbm["mae"].mean()
    print(f"  MAE promedio LightGBM: {mae_lgbm:.2f} "
          f"(mejora {100*(1-mae_lgbm/mae_baseline):.1f}% sobre el baseline)")

    print("\n[5/5] Desglose por categoría de intermitencia...")
    resumen_categoria = desglose.calcular_desglose(df, metadata, target.columna_demanda)
    print(resumen_categoria.to_string())

    # ── Guardado ──────────────────────────────────────────────────────
    resultados_baseline.to_csv(DIRECTORIO_SALIDA / f"baseline_resultados_{target.nombre}.csv", index=False)
    resultados_lgbm.to_csv(DIRECTORIO_SALIDA / f"lightgbm_resultados_{target.nombre}.csv", index=False)
    resumen_categoria.to_csv(DIRECTORIO_SALIDA / f"desglose_categoria_{target.nombre}.csv")
    metadata.to_csv(DIRECTORIO_SALIDA / f"folds_metadata_{target.nombre}.csv", index=False)

    return {
        "target": target.nombre,
        "mae_baseline": mae_baseline,
        "mae_lightgbm": mae_lgbm,
        "mejora_pct": 100 * (1 - mae_lgbm / mae_baseline),
    }


def main():
    resumen_final = [ejecutar_para_target(t) for t in config.TARGETS]

    print(f"\n{'='*70}")
    print("RESUMEN FINAL — todos los targets")
    print(f"{'='*70}")
    print(pd.DataFrame(resumen_final).round(2).to_string(index=False))
    print("\nPipeline de modelado completo.")


if __name__ == "__main__":
    main()
