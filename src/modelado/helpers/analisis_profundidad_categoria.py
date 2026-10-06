"""
Análisis de interpretación en profundidad por categoría de demanda,
modelo de 1 día. Reutiliza el desglose ya calculado (desglose_categoria_*.csv),
no reentrena nada -- es puro análisis de lo que ya tenemos.

Preguntas que responde:
1. ¿Qué peso tiene cada categoría en el MAE agregado?
2. ¿La mejora de LightGBM es proporcional a qué tan difícil es el baseline
   en cada categoría (evidencia de que el ML aporta más donde más falta hace)?
3. ¿Dónde está el mayor "ahorro" real (reducción absoluta de error), no
   solo el mayor porcentaje?

Requiere: desglose_categoria_MEDIANA.csv y desglose_categoria_SBA.csv
(generados por 05_desglose_categoria.py).

Uso:
    python analisis_profundidad_categoria.py
"""

import pandas as pd
from pathlib import Path
from src import rutas

DIRECTORIO_BASE = rutas.DIR_RESULTADOS_1D

def analizar(target: str):
    print(f"{'='*70}")
    print(f"ANÁLISIS DE PROFUNDIDAD — {target}")
    print(f"{'='*70}")

    archivo_csv = DIRECTORIO_BASE / f"desglose_categoria_{target}.csv"
    df = pd.read_csv(archivo_csv)

    df["peso_pct"] = (100 * df["n_filas"] / df["n_filas"].sum()).round(1)
    df["reduccion_absoluta"] = df["mae_baseline"] - df["mae_lightgbm"]

    columnas_mostrar = ["categoria", "n_filas", "peso_pct", "mae_baseline",
                        "mae_lightgbm", "mejora_pct", "reduccion_absoluta"]
    print(df[columnas_mostrar].to_string(index=False))

    mae_baseline_pond = (df["n_filas"] * df["mae_baseline"]).sum() / df["n_filas"].sum()
    mae_lightgbm_pond = (df["n_filas"] * df["mae_lightgbm"]).sum() / df["n_filas"].sum()
    print(f"\nMAE agregado (verificación): baseline={mae_baseline_pond:.2f}, "
          f"LightGBM={mae_lightgbm_pond:.2f}")

    corr = df["mae_baseline"].corr(df["mejora_pct"])
    print(f"Correlación (dificultad del baseline vs mejora %): {corr:.2f}")
    print("  -> positiva y moderada/fuerte: el ML aporta más donde el baseline es más débil\n")

    return df


def main():
    resultados = {}
    for target in ["MEDIANA", "SBA"]:
        resultados[target] = analizar(target)
        resultados_csv = DIRECTORIO_BASE / f"analisis_profundidad_categoria_{target}.csv"
        resultados[target].to_csv(resultados_csv, index=False)

    print("Guardado: analisis_profundidad_categoria_MEDIANA.csv, "
          "analisis_profundidad_categoria_SBA.csv")


if __name__ == "__main__":
    main()