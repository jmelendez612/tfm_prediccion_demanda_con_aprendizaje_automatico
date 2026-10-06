"""
Compara dos carpetas de resultados (CSV) y reporta diferencias numéricas.

Útil para comprobar que una nueva ejecución reproduce los resultados de una
referencia. No modifica nada.

Uso:
    python -m src.reportes.comparar_resultados <carpeta_referencia> <carpeta_nueva> [tolerancia_relativa]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Archivos con las métricas principales; se comparan solo si existen en ambas carpetas
PATRONES = ["baseline_resultados_*.csv", "lightgbm_resultados_*.csv",
            "baseline_30d_resultados_*.csv", "lightgbm_30d_resultados_*.csv",
            "residual_learning_30d_resultados_*.csv"]


def comparar(ref: Path, nueva: Path, rtol: float = 1e-6) -> bool:
    todo_ok, comparados = True, 0
    for patron in PATRONES:
        for archivo in sorted(ref.glob(patron)):
            otro = nueva / archivo.name
            if not otro.exists():
                print(f"  {archivo.name}: no existe en la carpeta nueva")
                todo_ok = False
                continue
            a, b = pd.read_csv(archivo), pd.read_csv(otro)
            if a.shape != b.shape:
                print(f"  {archivo.name}: forma distinta {a.shape} vs {b.shape}")
                todo_ok = False
                continue
            num = a.select_dtypes("number").columns.intersection(b.columns)
            dif = {c: float(np.max(np.abs(a[c] - b[c]) / (np.abs(a[c]) + 1e-12))) for c in num}
            peor = max(dif.items(), key=lambda kv: kv[1]) if dif else ("-", 0.0)
            ok = peor[1] <= rtol
            todo_ok &= ok
            comparados += 1
            print(f"  {'OK' if ok else 'DIF'} {archivo.name}: máxima diferencia relativa "
                  f"{peor[1]:.2e} (columna {peor[0]})")
    print(f"\n{comparados} archivos comparados. "
          f"{'Resultados equivalentes.' if todo_ok and comparados else 'Hay diferencias o no se comparó nada.'}")
    return bool(todo_ok and comparados)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
    sys.exit(0 if comparar(Path(sys.argv[1]), Path(sys.argv[2]), tol) else 2)
