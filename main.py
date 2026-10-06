"""
Punto de entrada del proyecto: menú interactivo.

Ejecutar desde la raíz del repositorio:   python main.py
Consulta el README para el orden recomendado y los datos requeridos.
"""
from src.database import DatabaseManager  # noqa: F401  (comprueba rutas al importar)


def mostrar_menu():
    print("\n" + "=" * 62)
    print("   PREDICCIÓN DE DEMANDA HOSPITALARIA (TFM)")
    print("=" * 62)
    print(" -- Preparación de datos (requieren ERP / base SQLite local) --")
    print("  1. [ETL] Descargar maestro de SKU (Oracle)")
    print("  2. [ETL] Descargar histórico de stock (Oracle)")
    print("  3. [ETL] Descargar histórico de consumos (Oracle)")
    print("  4. Muestreo + imputación (MEDIANA y SBA) + ingeniería de variables")
    print("  5. Réplica funcional del CPFR (QTY_FORECAST)")
    print(" -- Modelado --")
    print("  6. Modelado diario: baseline CPFR + LightGBM")
    print("  7. Modelado a 30 días: directo + desglose + Residual Learning")
    print("  8. Re-tuning de hiperparámetros de 30 días (opcional)")
    print("  9. Diagnóstico de early stopping de 30 días (auditoría)")
    print(" -- Análisis complementarios e informe --")
    print(" 10. Análisis complementarios del modelo diario (robustez, hospital,")
    print("     categoría, importancia/SHAP, curva de aprendizaje)")
    print(" 11. Generar figuras del informe (lee resultados guardados)")
    print("  0. Salir")
    print("=" * 62)
    return input("Seleccione una opción: ").strip()


def main():
    while True:
        opcion = mostrar_menu()

        if opcion in ("1", "2", "3"):
            from src.extractor import DataExtractor
            extractor = DataExtractor()
            if opcion == "1":
                extractor.extraer_maestro_skus()
            elif opcion == "2":
                extractor.extraer_historico_stock()
            else:
                extractor.extraer_historico_consumos()

        elif opcion == "4":
            print("\n--- MUESTREO + IMPUTACIÓN + FEATURES ---")
            from src.pipeline_muestra_red05 import ejecutar_pipeline_muestra
            ejecutar_pipeline_muestra()

        elif opcion == "5":
            print("\n--- RÉPLICA CPFR: QTY_FORECAST ---")
            try:
                from src.cpfr.pipeline import ejecutar_cpfr
                ejecutar_cpfr(
                    fecha_inicio="2022-12-01",
                    fecha_fin="2026-06-30",
                    hospital=None,
                    limite_por_curva=None,
                    guardar_detalle=False,
                )
            except Exception as e:
                print(f"Error durante el cálculo CPFR: {e}")

        elif opcion == "6":
            print("\n--- MODELADO DIARIO: CPFR + LIGHTGBM ---")
            print("Aviso: tarda ~13 minutos (6 folds x 2 targets).\n")
            from src.modelado.pipeline import main as ejecutar_modelado_completo
            ejecutar_modelado_completo()

        elif opcion == "7":
            print("\n--- MODELADO A 30 DÍAS ---")
            print("Aviso: tarda ~40-50 minutos en total (4 sub-pasos).\n")
            from src.modelado_30d.pipeline_30d import main as ejecutar_modelado_30d
            from src.modelado_30d.desglose_30d import main as ejecutar_desglose_30d
            from src.modelado_30d.residual_learning_30d import main as ejecutar_residual_30d
            from src.modelado_30d.desglose_residual_30d import main as ejecutar_desglose_residual_30d
            print(">>> [1/4] Pipeline directo...")
            ejecutar_modelado_30d()
            print("\n>>> [2/4] Desglose por categoría (directo)...")
            ejecutar_desglose_30d()
            print("\n>>> [3/4] Residual Learning...")
            ejecutar_residual_30d()
            print("\n>>> [4/4] Desglose por categoría (Residual Learning)...")
            ejecutar_desglose_residual_30d()
            print("\nModelado a 30 días completo.")

        elif opcion == "8":
            print("\n--- RE-TUNING DE HIPERPARÁMETROS (30 días) ---")
            print("No aplica el resultado automáticamente: revisa la combinación ganadora y,")
            print("si corresponde, actualiza PARAMS_LGBM en src/modelado_30d/config_30d.py.\n")
            from src.modelado_30d.tuning_30d import main as ejecutar_tuning_30d
            ejecutar_tuning_30d()

        elif opcion == "9":
            print("\n--- DIAGNÓSTICO: CURVAS DE EARLY STOPPING (30 días, directo) ---")
            print("Auditoría; no forma parte del pipeline normal. Tarda unos minutos.\n")
            from src.modelado_30d.diagnostico_early_stopping_30d import main as ejecutar_diagnostico
            ejecutar_diagnostico()

        elif opcion == "10":
            print("\n--- ANÁLISIS COMPLEMENTARIOS DEL MODELO DIARIO ---")
            print("Requiere haber ejecutado antes la opción 6.\n")
            from src.modelado.helpers import robustez_por_fold, analisis_por_hospital
            from src.modelado.helpers import analisis_profundidad_categoria, importancia_variables
            from src.modelado.helpers import curva_aprendizaje
            for nombre, modulo in [
                ("Robustez por fold", robustez_por_fold),
                ("Análisis por hospital", analisis_por_hospital),
                ("Profundidad por categoría", analisis_profundidad_categoria),
                ("Importancia de variables y SHAP", importancia_variables),
                ("Curva de aprendizaje", curva_aprendizaje),
            ]:
                print(f"\n>>> {nombre}...")
                modulo.main()

        elif opcion == "11":
            print("\n--- GENERAR FIGURAS DEL INFORME ---")
            print("Lee los resultados ya guardados y guarda 6 PNG a 300 dpi en data/graficos/.\n")
            from src.reportes.generar_figuras import main as ejecutar_generacion_figuras
            ejecutar_generacion_figuras()

        elif opcion == "0":
            print("Cerrando sesión de trabajo.")
            break

        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()
