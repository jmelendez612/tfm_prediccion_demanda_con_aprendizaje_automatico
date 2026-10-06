# Carpeta de datos (no versionada)

Esta carpeta se mantiene vacía en el repositorio. Los datos del estudio provienen de un ERP
hospitalario y son confidenciales, por lo que **no se publican**. Estructura esperada:

```
data/
├── raw/
│   └── tfm_demanda.db        Base SQLite generada por las opciones 1-5 del menú
├── processed/
│   ├── DATASET_ENTRENAMIENTO_COMPLETO_MEDIANA.csv
│   └── DATASET_ENTRENAMIENTO_COMPLETO_SBA.csv
├── results/
│   ├── diario/               Salidas del modelado diario (opciones 6 y 10)
│   └── 30d/                  Salidas del modelado a 30 días (opción 7)
└── graficos/                 Figuras del informe (opción 11)
```

Para usar otra ubicación, define la variable de entorno `TFM_DATA_DIR`.
