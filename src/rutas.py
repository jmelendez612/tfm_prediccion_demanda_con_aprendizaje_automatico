"""
Rutas centrales del proyecto. Todo es relativo a la raíz del repositorio.

Estructura de datos esperada (la carpeta data/ NO se versiona):

    data/raw/tfm_demanda.db        Base SQLite con los datos extraídos del ERP
    data/processed/                CSV del dataset de entrenamiento (MEDIANA / SBA)
    data/results/diario/           Resultados del horizonte diario
    data/results/30d/              Resultados del horizonte de 30 días
    data/graficos/                 Figuras del informe

Para usar los datos desde otra ubicación, define la variable de entorno
TFM_DATA_DIR con la ruta de la carpeta que contiene raw/, processed/ y results/.
"""
import os
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DIR_DATOS = Path(os.getenv("TFM_DATA_DIR", RAIZ_PROYECTO / "data"))

DB_PATH = DIR_DATOS / "raw" / "tfm_demanda.db"
DIR_PROCESADO = DIR_DATOS / "processed"
DIR_RESULTADOS_1D = DIR_DATOS / "results" / "diario"
DIR_RESULTADOS_30D = DIR_DATOS / "results" / "30d"
DIR_GRAFICOS = DIR_DATOS / "graficos"
