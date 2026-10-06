"""
Análisis de robustez por fold. Reutiliza los CSV de resultados ya
generados (baseline_resultados_*.csv, lightgbm_resultados_*.csv del
modelo de 1 día; residual_learning_30d_resultados_*.csv del de 30 días)
-- no reentrena nada, es análisis puro de estabilidad.

Preguntas que responde:
1. ¿Qué tan estable es la mejora de LightGBM entre folds (coeficiente
   de variación), o depende mucho de qué mes le toque evaluar?
2. ¿Hay algún fold sistemáticamente peor, y coincide entre el modelo
   de 1 día y el de 30 días (señal de algo real del período, no ruido
   de un solo modelo)?
3. ¿En cuántos de los 6 folds LightGBM efectivamente le gana al baseline?

Uso:
    python robustez_por_fold.py
"""
import sys
from pathlib import Path
RAIZ_PROYECTO = Path(__file__).resolve().parents[3]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import pandas as pd
from src import rutas

DIRECTORIO_BASE = rutas.DIR_RESULTADOS_1D

def analizar_1_dia(target: str):
    baseline = pd.read_csv(DIRECTORIO_BASE/f"baseline_resultados_{target}.csv")
    lightgbm = pd.read_csv(DIRECTORIO_BASE/f"lightgbm_resultados_{target}.csv")
    metadata = pd.read_csv(DIRECTORIO_BASE/f"folds_metadata_{target}.csv", parse_dates=["test_inicio", "test_fin"])

    tabla = lightgbm[["fold", "mae"]].rename(columns={"mae": "mae_lightgbm"})
    tabla = tabla.merge(baseline[["fold", "mae"]].rename(columns={"mae": "mae_baseline"}), on="fold")
    tabla = tabla.merge(metadata[["fold", "test_inicio", "test_fin"]], on="fold")
    tabla["mejora_pct"] = 100 * (1 - tabla["mae_lightgbm"] / tabla["mae_baseline"])

    return tabla


def resumen_robustez(tabla: pd.DataFrame, nombre: str):
    print(f"\n{'='*70}")
    print(f"ROBUSTEZ POR FOLD — {nombre}")
    print(f"{'='*70}")
    print(tabla.to_string(index=False))

    media = tabla["mejora_pct"].mean()
    std = tabla["mejora_pct"].std()
    cv = abs(std / media) if media != 0 else float("nan")
    n_folds_gana = (tabla["mejora_pct"] > 0).sum()

    print(f"\nMejora promedio: {media:.1f}% | Desvío estándar: {std:.1f} pp | "
          f"Coeficiente de variación: {cv:.2f}")
    print(f"Folds donde LightGBM supera al baseline: {n_folds_gana} de {len(tabla)}")

    peor_fold = tabla.loc[tabla["mejora_pct"].idxmin()]
    mejor_fold = tabla.loc[tabla["mejora_pct"].idxmax()]
    print(f"Peor fold: #{int(peor_fold['fold'])} ({peor_fold['mejora_pct']:.1f}%)")
    print(f"Mejor fold: #{int(mejor_fold['fold'])} ({mejor_fold['mejora_pct']:.1f}%)")


def main():
    print("MODELO DE 1 DÍA")
    for target in ["MEDIANA", "SBA"]:
        tabla = analizar_1_dia(target)
        resumen_robustez(tabla, f"1 día — {target}")
        tabla.to_csv(DIRECTORIO_BASE/f"robustez_1dia_{target}.csv", index=False)
    """
    print("\n\nMODELO DE 30 DÍAS (Residual Learning)")
    for target in ["MEDIANA", "SBA"]:
        tabla = pd.read_csv(f"residual_learning_30d_resultados_{target}.csv")
        tabla = tabla.rename(columns={
            "mae_residual_learning": "mae_lightgbm",
            "mae_baseline_en_este_fold": "mae_baseline",
        })
        resumen_robustez(tabla, f"30 días (Residual Learning) — {target}")
    """

if __name__ == "__main__":
    main()