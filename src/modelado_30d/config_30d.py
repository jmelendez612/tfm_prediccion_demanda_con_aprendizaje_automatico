"""
Configuración del modelo de 30 días. Reutiliza las variables y parámetros
de LightGBM del modelo diario (src/modelado/config.py).
"""

from dataclasses import dataclass
from src.modelado import config as config_1d
from pathlib import Path
from src import rutas


@dataclass
class TargetConfig30d:
    nombre: str
    ruta_csv: str                 # CSV extendido a jun-2026, con CURVA_XYZ
    columna_demanda: str           # "DEMANDA_IMPUTADA" o "DEMANDA_IMPUTADA_SBA"


# IMPORTANTE: estos CSV deben ser los extendidos a jun-2026 (ver registro
# de cambios #28) -- si usás los que llegan solo a abril, casi ninguna
# fila va a tener target_30d válido cerca del final de la serie.
DIRECTORIO_BASE = rutas.DIR_PROCESADO
TARGETS_30D = [
    TargetConfig30d(
        nombre="MEDIANA",
        ruta_csv=DIRECTORIO_BASE /"DATASET_ENTRENAMIENTO_COMPLETO_MEDIANA.csv",
        columna_demanda="DEMANDA_IMPUTADA",
    ),
    TargetConfig30d(
        nombre="SBA",
        ruta_csv=DIRECTORIO_BASE /"DATASET_ENTRENAMIENTO_COMPLETO_SBA.csv",
        columna_demanda="DEMANDA_IMPUTADA_SBA",
    ),
]

VENTANA_DIAS = 30

# Una ventana de 30 días futuros necesita al menos este % de días reales
# (stock_flag==1, STOCK_ES_REAL==1, CONSUMO_ES_REAL==1); si no, target_30d = NaN.
UMBRAL_COMPLETITUD = 0.70

N_CORTES = 6

# Reutilizados tal cual del modelado de 1 día -- mismas features, mismos
# hiperparámetros, misma réplica del CPFR real como baseline.
CATEGORICAS_FIJAS = config_1d.CATEGORICAS_FIJAS
NUMERICAS_FIJAS = config_1d.NUMERICAS_FIJAS
CATEGORICA_POR_FOLD = config_1d.CATEGORICA_POR_FOLD
NUMERICAS_POR_FOLD = config_1d.NUMERICAS_POR_FOLD
PARAMS_LGBM = dict(
    objective="mae",
    # n_estimators generoso a propósito: early_stopping corta apenas deja
    # de mejorar, así que un techo alto no cuesta tiempo. El fold 1 (el
    # usado para tunear) es el que menos historia de train tiene de los 6
    # folds walk-forward -- folds posteriores, con mucha más data, podrían
    # necesitar más iteraciones que las 371 observadas ahí.
    n_estimators=4000,
    # Elegido con el fold 1 (tuning_30d.py), el único cuya validación no se
    # solapa con ningún test reportado. Ver data/results/30d/tuning_30d_resultados.csv
    learning_rate=0.08,
    num_leaves=63,
    max_depth=6,
    min_child_samples=50,
    feature_fraction=0.6,
    bagging_fraction=0.6,
    bagging_freq=1,
    random_state=42,
    verbosity=-1,
)
EARLY_STOPPING_ROUNDS = config_1d.EARLY_STOPPING_ROUNDS

DB_PATH = config_1d.DB_PATH
TABLA_CPFR_REAL = config_1d.TABLA_CPFR_REAL
TABLA_MUESTRA = config_1d.TABLA_MUESTRA
CPFR_DESDE_CSV = config_1d.CPFR_DESDE_CSV
RUTA_CPFR_REAL = config_1d.RUTA_CPFR_REAL
