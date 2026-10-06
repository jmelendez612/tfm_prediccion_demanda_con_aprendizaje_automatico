import time
import sqlite3
import pandas as pd

from src.muestreo_estratificado import MuestreoEstratificado
from src.imputacion_demanda import ImputadorDemanda
from src.featuring import DataFeaturing
from src import rutas


def ejecutar_pipeline_muestra(db_path=rutas.DB_PATH,
                               prefijo_red="05%",
                               n_total=None,
                               medir_tiempo_con=None):
    """
    Flujo completo, en orden:

      1. Muestreo estratificado por GRUPO_FARMACOLOGICO, acotado a la
         red hospitalaria indicada (filtros de negocio ya incluidos:
         medicina, activo, planificable).
      2. Preprocesamiento (imputación de demanda) — UN loop por hospital,
         porque procesar_demanda_imputada_pdc solo acepta un hospital_id
         a la vez. Se acumulan los resultados y se guardan UNA sola vez
         al final (evita el bug de sobreescritura con if_exists='replace').
      3. Feature engineering: se ejecuta dos veces, generando
         DATASET_ENTRENAMIENTO_COMPLETO_MEDIANA y _SBA por separado.

    medir_tiempo_con: si se especifica un número (ej. 30), procesa solo
    esa cantidad de SKUs primero (tomados de la propia muestra) y reporta
    el tiempo, sin seguir con el resto (sirve para estimar el rendimiento
    antes de correr la muestra completa).
    """

    # ── Paso 1 — Muestreo ────────────────────────────────────────────────
    print("=" * 60)
    print("PASO 1 — Muestreo estratificado")
    print("=" * 60)
    ms = MuestreoEstratificado(db_path=db_path)
    muestra = ms.generar_muestra(n_total=n_total, filtro_hospital_like=prefijo_red)
    ms.guardar_muestra(muestra, nombre_tabla="MUESTRA_SKUS_RED05")

    if medir_tiempo_con:
        print(f"\nModo medición de tiempo activado — usando solo "
              f"{medir_tiempo_con} SKUs de la muestra.")
        muestra = muestra.head(medir_tiempo_con)

    # ── Paso 2 — Preprocesamiento, un loop por hospital ─────────────────
    print("\n" + "=" * 60)
    print("PASO 2 — Preprocesamiento (imputación) por hospital")
    print("=" * 60)

    preprocesador = ImputadorDemanda(db_path=db_path)
    resultados_por_hospital = []
    inicio = time.time()

    hospitales = muestra.groupby('HOSPITAL_ID')
    for i, (hospital_id, grupo) in enumerate(hospitales, 1):
        skus_del_hospital = grupo['PRODUCTO_ID'].tolist()
        print(f"\n[{i}/{len(hospitales)}] Hospital {hospital_id} — "
              f"{len(skus_del_hospital)} SKUs")

        df_hospital = preprocesador.procesar_demanda_imputada_pdc(
            skus_del_hospital, hospital_id, guardar_en_db=False,
            generar_traza=bool(medir_tiempo_con)
        )
        if df_hospital is not None:
            resultados_por_hospital.append(df_hospital)

    tiempo_total = time.time() - inicio
    n_skus_procesados = len(muestra)
    print(f"\nTiempo total de preprocesamiento: {tiempo_total:.1f}s "
          f"para {n_skus_procesados} SKU x Hospital "
          f"({tiempo_total/max(n_skus_procesados,1):.2f}s por SKU en promedio)")

    if medir_tiempo_con:
        print(f"\nProyección para el universo completo de la red "
              f"(estimación lineal, ver nota de no-linealidad abajo):")
        return {"tiempo_total": tiempo_total, "n_procesados": n_skus_procesados}

    if not resultados_por_hospital:
        print("No se procesó ningún hospital. Deteniendo pipeline.")
        return None

    # ── Guardado único, ahora sí, de todos los hospitales juntos ───────
    df_pdc_final = pd.concat(resultados_por_hospital, ignore_index=True)
    conn = sqlite3.connect(db_path)
    df_pdc_final.to_sql("PDC_DATASET_ENTRENAMIENTO", conn, if_exists='replace', index=False)
    conn.close()
    print(f"\nPDC_DATASET_ENTRENAMIENTO guardado con {len(df_pdc_final):,} filas "
          f"({len(hospitales)} hospitales).")

    # ── Paso 3 — Feature engineering, ambos targets ─────────────────────
    print("\n" + "=" * 60)
    print("PASO 3 — Feature engineering (una tabla por target)")
    print("=" * 60)

    fe = DataFeaturing(db_path=db_path)
    df_mediana = fe.construir_dataset_completo(columna_demanda="DEMANDA_IMPUTADA")
    df_sba = fe.construir_dataset_completo(columna_demanda="DEMANDA_IMPUTADA_SBA")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETO — resumen")
    print("=" * 60)
    print(f"SKUs x Hospital procesados : {len(muestra):,}")
    print(f"Hospitales                : {len(hospitales)}")
    print(f"Tiempo de preprocesamiento : {tiempo_total:.1f}s")
    print(f"DATASET_ENTRENAMIENTO_COMPLETO_MEDIANA : {len(df_mediana):,} filas")
    print(f"DATASET_ENTRENAMIENTO_COMPLETO_SBA     : {len(df_sba):,} filas")

    return {"mediana": df_mediana, "sba": df_sba, "tiempo_preprocesamiento": tiempo_total}


if __name__ == "__main__":
    # ── Paso 6 (recomendado primero) — medir tiempo con un lote chico ──
    print(">>> Midiendo tiempo con un lote chico antes de correr todo...\n")
    medicion = ejecutar_pipeline_muestra(medir_tiempo_con=30)

    if medicion:
        tiempo_por_sku = medicion["tiempo_total"] / medicion["n_procesados"]
        print(f"\nTiempo estimado por SKU: {tiempo_por_sku:.2f}s")
        print("Si el tiempo total te parece razonable, comenta la línea de "
              "arriba y descomenta la siguiente para correr la muestra completa:\n")
        print("# resultado = ejecutar_pipeline_muestra()")

    # ── Corrida completa (descomentar cuando ya mediste el tiempo) ─────
    # resultado = ejecutar_pipeline_muestra()
