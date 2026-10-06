"""
Genera las figuras del informe (PNG, 300 dpi) en data/graficos/.
Solo lee resultados ya guardados; no entrena ni modifica nada.

  1. MAE diario por fold (CPFR vs LightGBM).
  2. Mejora relativa por hospital (anonimizados H01-H20).
  3. SHAP beeswarm del modelo diario (fold 6).
  4. Horizonte de 30 días: LightGBM directo vs Residual Learning.
  5. Curva de aprendizaje del modelo diario.

La mejora agregada usa la media simple del MAE de los 6 folds.
Uso: opción 11 de main.py, o  python -m src.reportes.generar_figuras
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import rutas
DIR_DIARIO = rutas.DIR_RESULTADOS_1D
DIR_30D = rutas.DIR_RESULTADOS_30D
DIR_SALIDA = rutas.DIR_GRAFICOS

TARGETS = ["MEDIANA", "SBA"]
DPI = 300

COLOR_CPFR = "#E07B39"      # naranja
COLOR_LGBM = "#2E6DA4"      # azul
COLOR_RESIDUAL = "#3A9E6F"  # verde
COLOR_SBA = "#8E5BA8"       # violeta (para figura 2)
COLOR_NEUTRO = "#555555"

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def _fmt(valor: float, decimales: int = 1) -> str:
    """Número con coma decimal (como en el informe)."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def _etiqueta_mes(fecha) -> str:
    f = pd.Timestamp(fecha)
    return f"{MESES[f.month - 1]}-{f.year}"


def _estilo_ejes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def _guardar(fig, nombre: str):
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = DIR_SALIDA / nombre
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {ruta}")


def _leer(ruta: Path, **kwargs) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"Falta el archivo de resultados: {ruta}")
    return pd.read_csv(ruta, **kwargs)


# ----------------------------------------------------------------------------
# Figura 1 — MAE diario por fold
# ----------------------------------------------------------------------------
def figura_1_mae_diario():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, target in zip(axes, TARGETS):
        df = _leer(DIR_DIARIO / f"robustez_1dia_{target}.csv").sort_values("fold")
        x = np.arange(len(df))
        ancho = 0.38
        ax.bar(x - ancho / 2, df["mae_baseline"], ancho, color=COLOR_CPFR, label="CPFR")
        ax.bar(x + ancho / 2, df["mae_lightgbm"], ancho, color=COLOR_LGBM, label="LightGBM")
        for xi, (_, fila) in zip(x, df.iterrows()):
            ax.text(xi + ancho / 2, fila["mae_lightgbm"] + 1.5,
                    f"−{_fmt(fila['mejora_pct'])} %\nvs CPFR",
                    ha="center", va="bottom", fontsize=8, color=COLOR_LGBM)
        ax.set_xticks(x)
        ax.set_xticklabels([_etiqueta_mes(f) for f in df["test_inicio"]])
        ax.set_title(f"Target {target}")
        ax.set_xlabel("Fold temporal (mes de test)")
        _estilo_ejes(ax)
    axes[0].set_ylabel("MAE")
    axes[0].legend(frameon=False)
    fig.suptitle("MAE diario por fold temporal: CPFR frente a LightGBM", fontsize=12)
    fig.text(0.01, 0.005,
             "Altura de las barras: MAE. El porcentaje sobre cada barra azul indica la reducción "
             "del MAE de LightGBM respecto al CPFR en ese fold.",
             fontsize=8, color=COLOR_NEUTRO)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _guardar(fig, "figura_01_mae_diario_por_fold.png")


# ----------------------------------------------------------------------------
# Figura 2 — Mejora relativa por hospital
# ----------------------------------------------------------------------------
def figura_2_mejora_por_hospital():
    tablas = {t: _leer(DIR_DIARIO / f"analisis_hospital_{t}.csv", dtype={"hospital": str})
              for t in TARGETS}
    codigos = sorted(set(tablas["MEDIANA"]["hospital"]) | set(tablas["SBA"]["hospital"]))
    if len(codigos) != 20:
        raise ValueError(f"Se esperaban 20 hospitales y hay {len(codigos)}.")
    mapa = {cod: f"H{i + 1:02d}" for i, cod in enumerate(codigos)}   # mapeo fijo por código

    datos = pd.DataFrame({t: tablas[t].set_index("hospital")["mejora_pct"] for t in TARGETS})
    if datos.isna().any().any():
        raise ValueError("Algún hospital no aparece en ambos targets.")
    datos = datos.sort_values("MEDIANA")          # orden visual (peor a mejor en MEDIANA)
    etiquetas = [mapa[c] for c in datos.index]

    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(datos))
    alto = 0.38
    ax.barh(y - alto / 2, datos["MEDIANA"], alto, color=COLOR_LGBM, label="MEDIANA")
    ax.barh(y + alto / 2, datos["SBA"], alto, color=COLOR_SBA, label="SBA")
    ax.set_yticks(y)
    ax.set_yticklabels(etiquetas)
    ax.axvline(0, color=COLOR_NEUTRO, linewidth=1)
    ax.set_xlabel("Mejora relativa del MAE de LightGBM frente al CPFR (%)")
    ax.set_ylabel("Hospital (anonimizado)")
    ax.set_title("Mejora relativa de LightGBM por hospital (modelo diario)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _guardar(fig, "figura_02_mejora_por_hospital.png")


# ----------------------------------------------------------------------------
# Figura 3 — SHAP beeswarm (fold 6, modelo diario)
# ----------------------------------------------------------------------------
def _normalizar(nombre: str) -> str:
    return re.sub(r"_fold\d+$", "", nombre)


def figura_3_shap_beeswarm(top_n: int = 15):
    import shap  # import local: solo se necesita para esta figura

    for target in TARGETS:
        valores = np.load(DIR_DIARIO / f"shap_values_{target}.npy")
        muestra = _leer(DIR_DIARIO / f"shap_muestra_features_{target}.csv")

        # 1) Consistencia entre el .npy y la muestra
        if valores.shape != muestra.shape:
            raise ValueError(f"[{target}] shap_values {valores.shape} no coincide con la muestra "
                             f"{muestra.shape}: el .npy no corresponde a este CSV.")

        # 2) Reproduce la importancia SHAP oficial (Tabla 9)
        muestra.columns = [_normalizar(c) for c in muestra.columns]
        reproducida = pd.Series(np.abs(valores).mean(axis=0), index=muestra.columns)
        oficial = _leer(DIR_DIARIO / f"importancia_shap_{target}.csv", index_col=0)["importancia_shap_abs"]
        comparable = reproducida.loc[oficial.index]
        if not np.allclose(comparable.values, oficial.values, rtol=1e-4):
            raise ValueError(f"[{target}] El .npy no reproduce importancia_shap_{target}.csv. "
                             "Regenera los análisis de importancia antes de graficar.")
        print(f"  [{target}] SHAP verificado contra importancia_shap_{target}.csv "
              f"(primer lugar: {oficial.index[0]} = {_fmt(oficial.iloc[0], 2)})")

        # 3) Categóricas -> códigos numéricos solo para el color
        numerica = muestra.copy()
        for col in numerica.columns:
            if not pd.api.types.is_numeric_dtype(numerica[col]):
                codigos = numerica[col].astype("category").cat.codes.astype(float)
                numerica[col] = codigos.where(codigos >= 0)

        shap.summary_plot(valores, numerica, max_display=top_n, show=False, plot_size=(9, 6.5))
        ax = plt.gca()
        ax.set_xlabel("Valor SHAP (impacto en la predicción de demanda)")
        ax.set_title(f"SHAP beeswarm — modelo diario, fold 6 (target {target})")
        fig = plt.gcf()
        fig.tight_layout()
        _guardar(fig, f"figura_03_shap_beeswarm_{target}.png")


# ----------------------------------------------------------------------------
# Figura 4 — Estrategias a 30 días
# ----------------------------------------------------------------------------
def _datos_30d(target: str) -> pd.DataFrame:
    base = _leer(DIR_30D / f"baseline_30d_resultados_{target}.csv").set_index("fold")
    direc = _leer(DIR_30D / f"lightgbm_30d_resultados_{target}.csv").set_index("fold")
    resid = _leer(DIR_30D / f"residual_learning_30d_resultados_{target}.csv").set_index("fold")
    df = pd.DataFrame({
        "mae_cpfr": base["mae"],
        "mae_directo": direc["mae"],
        "mae_residual": resid["mae_residual_learning"],
    }).sort_index()
    if len(df) != 6 or df.isna().any().any():
        raise ValueError(f"[{target}] resultados de 30 días incompletos.")
    return df


def figura_4_estrategias_30d():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ancho = 0.34
    posiciones = np.arange(len(TARGETS))
    estrategias = [("mae_directo", "LightGBM directo", COLOR_LGBM, -ancho / 2),
                   ("mae_residual", "Residual Learning", COLOR_RESIDUAL, ancho / 2)]
    etiquetas_usadas = set()

    for i, target in enumerate(TARGETS):
        df = _datos_30d(target)
        media_cpfr = df["mae_cpfr"].mean()      # media simple de los 6 folds
        for col, nombre, color, desplazamiento in estrategias:
            agregada = 100 * (media_cpfr - df[col].mean()) / media_cpfr
            por_fold = 100 * (df["mae_cpfr"] - df[col]) / df["mae_cpfr"]
            x = posiciones[i] + desplazamiento
            ax.bar(x, agregada, ancho, color=color, alpha=0.85,
                   label=nombre if nombre not in etiquetas_usadas else None)
            etiquetas_usadas.add(nombre)
            rng = np.random.RandomState(42 + i)
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(por_fold)), por_fold,
                       color="black", s=14, zorder=3,
                       label="Mejora por fold" if "fold" not in etiquetas_usadas else None)
            etiquetas_usadas.add("fold")
            ax.text(x, agregada + (1.2 if agregada >= 0 else -1.2),
                    f"{_fmt(agregada, 2)} %", ha="center",
                    va="bottom" if agregada >= 0 else "top", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(posiciones)
    ax.set_xticklabels([f"Target {t}" for t in TARGETS])
    ax.set_ylabel("Mejora relativa del MAE frente al CPFR (%)")
    ax.set_title("Horizonte de 30 días: mejora frente al CPFR por estrategia")
    ax.legend(frameon=False, loc="best")
    _estilo_ejes(ax)
    fig.text(0.01, 0.005,
             "Barras: media simple de MAE de los 6 folds (dic-2025 a may-2026). "
             "Puntos: mejora de cada fold.",
             fontsize=8, color=COLOR_NEUTRO)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _guardar(fig, "figura_04_estrategias_30d.png")


# ----------------------------------------------------------------------------
# Figura 5 — Curva de aprendizaje del modelo diario (fold 6)
# ----------------------------------------------------------------------------
def figura_5_curva_aprendizaje():
    df = _leer(DIR_DIARIO / "curva_aprendizaje_completa.csv")
    colores = {"MEDIANA": COLOR_LGBM, "SBA": COLOR_SBA}

    fig, ax = plt.subplots(figsize=(8, 5))
    for target in TARGETS:
        d = df[df["target"] == target].sort_values("fraccion")
        ax.plot(d["fraccion"] * 100, d["mae"], marker="o", linewidth=2,
                color=colores[target], label=target)
        for _, fila in d.iterrows():
            ax.annotate(_fmt(fila["mae"], 1), (fila["fraccion"] * 100, fila["mae"]),
                        textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=8, color=colores[target])

    ref = df[df["target"] == "MEDIANA"].sort_values("fraccion")
    ax.set_xticks(ref["fraccion"] * 100)
    ax.set_xticklabels([f"{int(f * 100)} %\n({n} SKU×hosp.)"
                        for f, n in zip(ref["fraccion"], ref["n_skus"])])
    ax.set_xlabel("Fracción de combinaciones producto × hospital usadas en el entrenamiento")
    ax.set_ylabel("MAE en el bloque de test (fold 6)")
    ax.set_title("Curva de aprendizaje del modelo diario")
    ax.legend(frameon=False)
    _estilo_ejes(ax)
    fig.text(0.01, 0.005,
             "Una única selección aleatoria por fracción (semilla 42) y un solo fold: "
             "descriptiva, no demuestra un tamaño de muestra óptimo.",
             fontsize=8, color=COLOR_NEUTRO)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _guardar(fig, "figura_05_curva_aprendizaje.png")


# ----------------------------------------------------------------------------
def main():
    print("\nGenerando figuras (solo lectura de resultados ya guardados)...")
    print(f"Salida: {DIR_SALIDA}\n")
    print("Figura 1 — MAE diario por fold")
    figura_1_mae_diario()
    print("Figura 2 — Mejora por hospital")
    figura_2_mejora_por_hospital()
    print("Figura 3 — SHAP beeswarm")
    figura_3_shap_beeswarm()
    print("Figura 4 — Estrategias a 30 días")
    figura_4_estrategias_30d()
    print("Figura 5 — Curva de aprendizaje")
    figura_5_curva_aprendizaje()
    print("\nFiguras generadas.")


if __name__ == "__main__":
    main()
