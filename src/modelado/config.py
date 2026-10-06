"""
Configuración central del modelado diario (baseline CPFR + LightGBM).
Las rutas se definen en src/rutas.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from src import rutas

@dataclass
class TargetConfig:
    nombre: str                 # "MEDIANA" o "SBA" — usado en nombres de archivo de salida
    ruta_csv: str                # CSV exportado de DATASET_ENTRENAMIENTO_COMPLETO_*
    columna_demanda: str          # "DEMANDA_IMPUTADA" o "DEMANDA_IMPUTADA_SBA"


# Targets del proyecto
DIRECTORIO_BASE = rutas.DIR_PROCESADO
TARGETS = [
    TargetConfig(
        nombre="MEDIANA",
        ruta_csv=DIRECTORIO_BASE /"DATASET_ENTRENAMIENTO_COMPLETO_MEDIANA.csv",
        columna_demanda="DEMANDA_IMPUTADA",
    ),
    TargetConfig(
        nombre="SBA",
        ruta_csv=DIRECTORIO_BASE /"DATASET_ENTRENAMIENTO_COMPLETO_SBA.csv",
        columna_demanda="DEMANDA_IMPUTADA_SBA",
    ),
]

# Réplica real del CPFR (tabla CPFR_FORECAST_RESULTADO en SQLite, universo
# completo de 5,266 SKU x Hospital), usada como baseline Nivel 1 definitivo
# — reemplaza la aproximación mediana_180d de la primera versión del pipeline.
# El cruce se hace por SQL, filtrando directamente a la muestra de 358
# (tabla MUESTRA_SKUS_RED05), sin necesidad de exportar/filtrar CSV a mano.
DB_PATH = rutas.DB_PATH
TABLA_CPFR_REAL = "CPFR_FORECAST_RESULTADO"
TABLA_MUESTRA = "MUESTRA_SKUS_RED05"

# Si preferís seguir usando un CSV ya exportado/filtrado en vez del cruce
# directo por SQL, dejá esto en True y ajustá RUTA_CPFR_REAL más abajo.
CPFR_DESDE_CSV = False
RUTA_CPFR_REAL = "cpfr_real.csv"

# Rolling Windows — walk-forward, ventana expansiva
N_CORTES = 6

# Filtro estricto de "demanda genuinamente real" (sin días imputados),
# usado para recalcular adi/cv2/clasificacion_demanda/pct_censura_historica
# sin fuga de información en cada fold.
COLUMNAS_FILTRO_DIA_REAL = ["stock_flag", "STOCK_ES_REAL", "CONSUMO_ES_REAL"]

# ── Features de LightGBM ────────────────────────────────────────────────
CATEGORICAS_FIJAS = ["GRUPO_FARMACOLOGICO", "UNIDAD_MEDIDA", "CURVA_XYZ"]

NUMERICAS_FIJAS = [
    "dia_semana", "mes", "trimestre", "dia_del_año", "es_fin_semana", "es_festivo",
    "lag_7", "lag_14", "lag_30", "lag_90", "lag_180",
    "media_7d", "mediana_7d", "media_30d", "mediana_30d",
    "media_90d", "mediana_90d", "std_30d", "mediana_180d",
    "dias_sin_stock_7d", "dias_sin_stock_30d",
    "stock_ayer",
]

# Se recalculan por fold (sufijo _foldN agregado en tiempo de ejecución)
CATEGORICA_POR_FOLD = "clasificacion_demanda"
NUMERICAS_POR_FOLD = ["adi", "cv2", "pct_censura_historica"]

# Excluidas explícitamente por fuga de información (documentado en el
# contexto del proyecto, sección 11.3): DEMANDA_IMPUTADA/_SBA (target),
# CONSUMO_REGISTRADO, CANTIDAD_STOCK sin desplazar, stock_flag,
# STOCK_ES_REAL, CONSUMO_ES_REAL, CURVA_PQR, QTY_FORECAST (es el baseline,
# no una feature del modelo).

PARAMS_LGBM = dict(
    objective="mae",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=30,
    random_state=42,
    verbosity=-1,
)

EARLY_STOPPING_ROUNDS = 50
