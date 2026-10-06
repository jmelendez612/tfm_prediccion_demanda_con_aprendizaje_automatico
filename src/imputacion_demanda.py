import pandas as pd
import numpy as np
import sqlite3
from src import rutas

class ImputadorDemanda:
    def __init__(self, db_path=rutas.DB_PATH):
        self.db_path = db_path

    def croston_sba(self, serie_fechas, serie_consumo, alpha=0.1):
        """
        Estima la demanda esperada con Croston + corrección SBA
        (Syntetos & Boylan, 2005), a partir de una serie de días
        NO censurados (ya filtrados por stock_flag==1, STOCK_ES_REAL==1,
        CONSUMO_ES_REAL==1), ordenada cronológicamente.

        Devuelve un único valor: la demanda esperada para el próximo período,
        que se usa como imputación puntual del día censurado.
        """
        consumo = serie_consumo.reset_index(drop=True)

        if len(consumo) == 0:
            return 0.0

        posiciones_no_cero = consumo[consumo > 0].index.to_numpy()

        if len(posiciones_no_cero) == 0:
            # Nunca hubo demanda registrada en el historial disponible
            return 0.0

        demandas = consumo.loc[posiciones_no_cero].to_numpy(dtype=float)

        if len(posiciones_no_cero) == 1:
            # Solo una observación de demanda: no hay intervalo que estimar,
            # se usa el propio valor como mejor estimador disponible
            return float(demandas[0])

        intervalos = np.diff(posiciones_no_cero).astype(float)

        z_hat = demandas[0]
        p_hat = intervalos[0]

        for i in range(1, len(demandas)):
            z_hat = alpha * demandas[i] + (1 - alpha) * z_hat
            if i - 1 < len(intervalos):
                p_hat = alpha * intervalos[i - 1] + (1 - alpha) * p_hat

        if p_hat <= 0:
            return float(z_hat)

        forecast_croston = z_hat / p_hat
        forecast_sba = (1 - alpha / 2) * forecast_croston

        return max(float(forecast_sba), 0.0)

    def procesar_demanda_imputada_pdc(self, lista_skus, hospital_id, guardar_en_db=True,
                                       generar_traza=True):
        fecha_inicio = '2022-12-01'
        fecha_fin    = '2026-06-30'

        print(f"\n Iniciando Prueba de Concepto para {len(lista_skus)} SKUs...")
        conn = sqlite3.connect(self.db_path)

        # ─────────────────────────────────────────────────────────────
        # 0. Calendarios de actividad del sistema
        # ─────────────────────────────────────────────────────────────
        query_actividad_consumo = """
            SELECT DISTINCT FECHA, 1 AS SISTEMA_CONSUMOS_ACTIVO
            FROM CONSUMOS_HISTORICO
        """
        query_actividad_stock = """
            SELECT DISTINCT FECHA, 1 AS SISTEMA_STOCK_ACTIVO
            FROM STOCK_HISTORICO
        """

        df_actividad_consumo = pd.read_sql_query(query_actividad_consumo, conn)
        df_actividad_stock   = pd.read_sql_query(query_actividad_stock, conn)

        df_actividad_consumo['FECHA'] = pd.to_datetime(df_actividad_consumo['FECHA'])
        df_actividad_stock['FECHA']   = pd.to_datetime(df_actividad_stock['FECHA'])

        datasets_procesados = []

        for sku in lista_skus:
            print(f"Procesando SKU: {sku} | Hospital: {hospital_id}...")

            hospital_stock   = hospital_id
            hospital_consumo = f"05H0.{hospital_id}"

            # ─────────────────────────────────────────────────────────────
            # 1. Extracción con flags independientes por sistema
            # ─────────────────────────────────────────────────────────────
            query = f"""
            SELECT
                S.FECHA,
                S.PRODUCTO_ID,
                S.HOSPITAL_ID,
                S.CANTIDAD_STOCK,
                C.CONSUMO_REGISTRADO                                        AS CONSUMO_RAW,
                CASE WHEN S.CANTIDAD_STOCK      IS NULL THEN 0 ELSE 1 END   AS STOCK_ES_REAL,
                CASE WHEN C.CONSUMO_REGISTRADO  IS NULL THEN 0 ELSE 1 END   AS CONSUMO_ES_REAL
            FROM STOCK_HISTORICO S
            LEFT JOIN CONSUMOS_HISTORICO C
                ON  S.FECHA        = C.FECHA
                AND S.PRODUCTO_ID  = C.PRODUCTO_ID
                AND C.HOSPITAL_ID  = '{hospital_consumo}'
            WHERE S.PRODUCTO_ID = '{sku}'
            AND   S.HOSPITAL_ID = '{hospital_stock}'
            ORDER BY S.FECHA ASC
            """

            df_sku = pd.read_sql_query(query, conn)

            if df_sku.empty:
                print(f"  No se encontraron datos para el SKU {sku}. Saltando...")
                continue

            # --- ESTRUCTURA INICIAL DE LA TRAZA ---
            df_traza = pd.DataFrame(index=df_sku['FECHA'])
            df_traza['1_Stock_Original'] = df_sku.set_index('FECHA')['CANTIDAD_STOCK']
            df_traza['1_Consumo_Raw'] = df_sku.set_index('FECHA')['CONSUMO_RAW']

            # ─────────────────────────────────────────────────────────────
            # 2. Inversión Dinámica de Signo
            # ─────────────────────────────────────────────────────────────
            df_sku['CONSUMO_REGISTRADO'] = np.where(
                df_sku['CONSUMO_RAW'] < 0,
                df_sku['CONSUMO_RAW'] * -1,
                0
            )

            # ─────────────────────────────────────────────────────────────
            # 3. Reindexación por Calendario Completo
            # ─────────────────────────────────────────────────────────────
            df_sku['FECHA'] = pd.to_datetime(df_sku['FECHA'])
            df_sku = df_sku.set_index('FECHA')

            calendario = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
            df_sku = df_sku.reindex(calendario)
            df_sku.index.name = 'FECHA'
            df_sku = df_sku.reset_index()

            # ─────────────────────────────────────────────────────────────
            # 4. Flags provisionales post-reindex
            # ─────────────────────────────────────────────────────────────
            df_sku['STOCK_ES_REAL']   = df_sku['STOCK_ES_REAL'].fillna(0).astype(int)
            df_sku['CONSUMO_ES_REAL'] = df_sku['CONSUMO_ES_REAL'].fillna(0).astype(int)

            # ─────────────────────────────────────────────────────────────
            # 5. Cruzar con calendarios de actividad del sistema
            # ─────────────────────────────────────────────────────────────
            df_sku = df_sku.merge(df_actividad_consumo, on='FECHA', how='left')
            df_sku = df_sku.merge(df_actividad_stock,   on='FECHA', how='left')

            df_sku['SISTEMA_CONSUMOS_ACTIVO'] = df_sku['SISTEMA_CONSUMOS_ACTIVO'].fillna(0).astype(int)
            df_sku['SISTEMA_STOCK_ACTIVO']    = df_sku['SISTEMA_STOCK_ACTIVO'].fillna(0).astype(int)

            df_sku['CONSUMO_ES_REAL'] = np.where(
                (df_sku['CONSUMO_ES_REAL'] == 1) | (df_sku['SISTEMA_CONSUMOS_ACTIVO'] == 1),
                1, # 0 real (sistema activo)
                0  # caída de interfaz
            )
            df_sku['STOCK_ES_REAL'] = np.where(
                (df_sku['STOCK_ES_REAL'] == 1) | (df_sku['SISTEMA_STOCK_ACTIVO'] == 1),
                1, # stock 0 real (sistema activo)
                0  # caída de interfaz
            )

            # Guardamos los estados de los sistemas globales en nuestra traza visual
            df_sku_temp = df_sku.set_index('FECHA')
            df_traza.index = pd.to_datetime(df_traza.index)
            df_traza = df_traza.reindex(calendario)

            df_traza['2_Sistema_Consumo_Activo_Global'] = np.where(df_sku_temp['SISTEMA_CONSUMOS_ACTIVO'] == 1, 'SÍ', 'NO (Caída Transaccional)')
            df_traza['2_Sistema_Stock_Activo_Global'] = np.where(df_sku_temp['SISTEMA_STOCK_ACTIVO'] == 1, 'SÍ', 'NO (Caída Transaccional)')
            df_traza['3_Diagnóstico_Consumo'] = np.where(df_sku_temp['CONSUMO_ES_REAL'] == 1, 'Dato u Opc. Real (0)', 'Caída Interfaz')
            df_traza['3_Diagnóstico_Stock'] = np.where(df_sku_temp['STOCK_ES_REAL'] == 1, 'Dato u Opc. Real (0)', 'Caída Interfaz')

            # ─────────────────────────────────────────────────────────────
            # 6. Rellenos de valores
            # ─────────────────────────────────────────────────────────────
            # ═══════════════════════════════════════════════════════════════════
            # CORRECCIÓN 1 — Cero real de stock ANTES del ffill
            # ─────────────────────────────────────────────────────────────────────
            # Antes: se aplicaba ffill a TODOS los vacíos por igual, sin distinguir
            # entre "el sistema de stock funcionó y este SKU simplemente no tenía
            # stock" (cero real, inferido por STOCK_ES_REAL==1) y "el sistema de
            # stock completo se cayó ese día" (caída real de interfaz).
            #
            # Ahora:
            #   Paso A → si STOCK_ES_REAL==1 y no hay valor de CANTIDAD_STOCK,
            #            es un cero real → se fija explícitamente en 0.
            #   Paso B → ffill se aplica SOLO sobre lo que sigue en NaN después
            #            del paso A, es decir, únicamente las caídas reales de
            #            interfaz (donde no sabemos qué pasó y la mejor estimación
            #            disponible es heredar el último stock conocido con certeza).
            # Esto evita que un cero inferido "contamine" por ffill los días
            # posteriores de caída real, y evita que una caída real "tape" con
            # un valor viejo positivo lo que en realidad era un cero real.
            # ═══════════════════════════════════════════════════════════════════

            mascara_cero_real_stock = df_sku['CANTIDAD_STOCK'].isna() & (df_sku['STOCK_ES_REAL'] == 1)
            df_sku.loc[mascara_cero_real_stock, 'CANTIDAD_STOCK'] = 0

            df_sku['CANTIDAD_STOCK']     = df_sku['CANTIDAD_STOCK'].ffill().fillna(0)
            df_sku['CONSUMO_REGISTRADO'] = df_sku['CONSUMO_REGISTRADO'].fillna(0)

            df_sku['PRODUCTO_ID'] = sku
            df_sku['HOSPITAL_ID'] = hospital_id

            # ─────────────────────────────────────────────────────────────
            # 7 y 8. Inicialización y Flags de Stock
            # ─────────────────────────────────────────────────────────────
            # ═══════════════════════════════════════════════════════════════════
            # CORRECCIÓN 2 — stock_flag respeta el Tipo B (desfase de registro)
            # ─────────────────────────────────────────────────────────────────────
            # Antes: stock_flag = 1 solo si CANTIDAD_STOCK > 0, lo que trataba
            # como "censurado" (stock_flag=0) cualquier día con stock=0, incluso
            # cuando hubo consumo real registrado ese mismo día (Tipo B, definido
            # en el Word como "desfase de registro Oracle, NO requiere imputación").
            # Esto sesgaba las features de Bloque 7 (dias_sin_stock_7d/30d), que
            # contaban esos días como censura real sin serlo.
            #
            # Ahora: stock_flag = 1 si (stock>0) OR (stock=0 AND consumo>0),
            # (desfase de registro, sin necesidad de imputación).
            # ═══════════════════════════════════════════════════════════════════

            df_sku['stock_flag'] = np.where(
                (df_sku['CANTIDAD_STOCK'] > 0) | (df_sku['CONSUMO_REGISTRADO'] > 0),
                1,
                0
            )
            df_sku['DEMANDA_IMPUTADA'] = df_sku['CONSUMO_REGISTRADO'].astype(float)
            # Columna paralela — mismo valor real en días no censurados; se
            # completa con Croston/SBA (en vez de mediana) solo en los días
            # censurados dentro de la cascada de imputación, más abajo.
            df_sku['DEMANDA_IMPUTADA_SBA'] = df_sku['CONSUMO_REGISTRADO'].astype(float)

            df_sku_temp = df_sku.set_index('FECHA')
            df_traza['4_Stock_Final_Post_FFill'] = df_sku_temp['CANTIDAD_STOCK']
            df_traza['4_Consumo_Final_Post_FillNa'] = df_sku_temp['CONSUMO_REGISTRADO']

            # ─────────────────────────────────────────────────────────────
            # 9. Máscara de Imputación (Nueva Lógica)
            # ─────────────────────────────────────────────────────────────
            mascara_imputar = (
                (df_sku['CONSUMO_ES_REAL'] == 0)
            ) | (
                (df_sku['CONSUMO_ES_REAL'] == 1) &
                (df_sku['CONSUMO_REGISTRADO'] == 0) &
                (df_sku['stock_flag'] == 0) &
                (df_sku['STOCK_ES_REAL'] == 1)
            )

            mascara_imputar = mascara_imputar & ~(
                (df_sku['CONSUMO_ES_REAL'] == 1) & (df_sku['CONSUMO_REGISTRADO'] > 0)
            )

            indices_imputar = df_sku[mascara_imputar].index

            df_sku_temp = df_sku.copy()
            df_sku_temp['¿Sufre Imputación?'] = np.where(mascara_imputar, 'SÍ ', 'NO (Consumo u Opc. Válida)')
            df_sku_temp = df_sku_temp.set_index('FECHA')
            df_traza['5_Evaluación_Algoritmo'] = df_sku_temp['¿Sufre Imputación?']

            # ─────────────────────────────────────────────────────────────
            # 10. Cascada de Imputación
            # ─────────────────────────────────────────────────────────────
            for idx in indices_imputar:
                fecha_actual = df_sku.loc[idx, 'FECHA']

                # ═══════════════════════════════════════════════════════════════
                # DECISIÓN CONFIRMADA: NO filtrar por CONSUMO_REGISTRADO > 0 aquí.
                # ─────────────────────────────────────────────────────────────────
                # stock_flag==1 ya garantiza que el día NO está censurado (Tipo A o B),
                # así que un consumo=0 dentro de este filtro es demanda real igual a
                # cero (día sin rotación), no un dato faltante. Excluirlo forzaría a
                # que la demanda imputada nunca pueda ser 0, lo cual sesga sistemáti-
                # camente hacia arriba a los productos de curva R (baja rotación,
                # intermitentes), contradiciendo la naturaleza de su demanda.
                # Coherente con el enfoque estándar de demanda intermitente
                # (Croston/SBA), que modela explícitamente la probabilidad de
                # demanda cero en vez de descartar esos períodos.
                # ═══════════════════════════════════════════════════════════════

                historico_activo = df_sku[
                    (df_sku['FECHA'] < fecha_actual) &
                    (df_sku['stock_flag'] == 1) &
                    (df_sku['STOCK_ES_REAL'] == 1) &
                    (df_sku['CONSUMO_ES_REAL'] == 1)
                ]

                if len(historico_activo) >= 1:
                    ventana_30 = historico_activo.tail(30)
                    df_sku.loc[idx, 'DEMANDA_IMPUTADA'] = ventana_30['CONSUMO_REGISTRADO'].median()

                    # Croston/SBA usa TODO el historial activo disponible (no solo
                    # los últimos 30), porque necesita suficientes observaciones de
                    # intervalo entre demandas no-cero para que el suavizado
                    # exponencial sea estable.
                    df_sku.loc[idx, 'DEMANDA_IMPUTADA_SBA'] = self.croston_sba(
                        historico_activo['FECHA'],
                        historico_activo['CONSUMO_REGISTRADO']
                    )
                else:
                    # Nivel 2: se añade stock_flag==1 (faltaba en el original) para
                    # excluir explícitamente los días Tipo C (censurados) del cálculo
                    # de la mediana global de respaldo — esos ceros censurados no
                    # deben mezclarse con los ceros reales de demanda.
                    historico_global = df_sku[
                        (df_sku['FECHA'] < fecha_actual) &
                        (df_sku['stock_flag'] == 1) &
                        (df_sku['STOCK_ES_REAL'] == 1) &
                        (df_sku['CONSUMO_ES_REAL'] == 1)
                    ]
                    mediana_global = historico_global['CONSUMO_REGISTRADO'].median()

                    if not np.isnan(mediana_global):
                        df_sku.loc[idx, 'DEMANDA_IMPUTADA'] = mediana_global
                        df_sku.loc[idx, 'DEMANDA_IMPUTADA_SBA'] = self.croston_sba(
                            historico_global['FECHA'],
                            historico_global['CONSUMO_REGISTRADO']
                        )
                    else:
                        # Nivel 3: sin ningún historial activo — limitación
                        # documentada, ambas columnas quedan en 0.0
                        df_sku.loc[idx, 'DEMANDA_IMPUTADA'] = 0.0
                        df_sku.loc[idx, 'DEMANDA_IMPUTADA_SBA'] = 0.0

            df_sku_temp = df_sku.set_index('FECHA')
            df_traza['6_Demanda_Final_Mediana'] = df_sku_temp['DEMANDA_IMPUTADA']
            df_traza['6_Demanda_Final_Croston_SBA'] = df_sku_temp['DEMANDA_IMPUTADA_SBA']

            # ─────────────────────────────────────────────────────────────
            # EXPORTACIÓN DEL INFORME HTML (Traza de Auditoría)
            # Solo si generar_traza=True; desactivar en corridas grandes.
            # ─────────────────────────────────────────────────────────────
            if generar_traza:
                df_traza = df_traza.reset_index().rename(columns={'index': 'FECHA'})
                df_traza['FECHA'] = df_traza['FECHA'].dt.strftime('%Y-%m-%d')

                archivo_html = f"traza_sku_{sku}.html"
                df_traza.to_html(archivo_html, index=False, justify='center')
                print(f"[TRAZA GENERADA] Archivo '{archivo_html}' creado con éxito.")

            datasets_procesados.append(df_sku)

        # ─────────────────────────────────────────────────────────────────
        # 11. Consolidar y guardar
        # ─────────────────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────
        # 11. Consolidar y, opcionalmente, guardar
        # ─────────────────────────────────────────────────────────────────
        # guardar_en_db=False permite que un orquestador externo (ej. loop
        # por hospital sobre una muestra estratificada) acumule los
        # resultados de varias llamadas y los guarde UNA sola vez al final,
        # evitando que cada llamada sobreescriba (if_exists='replace') los
        # datos del hospital anterior.
        if datasets_procesados:
            df_pdc_final = pd.concat(datasets_procesados, ignore_index=True)
            df_pdc_final['FECHA'] = df_pdc_final['FECHA'].dt.strftime('%Y-%m-%d')

            if guardar_en_db:
                df_pdc_final.to_sql("PDC_DATASET_ENTRENAMIENTO", conn, if_exists='replace', index=False)
                print(f"\n¡Prueba de Concepto terminada! Tabla 'PDC_DATASET_ENTRENAMIENTO' creada con {len(df_pdc_final)} filas.")
            else:
                print(f"\nProcesado (sin guardar aún): {len(df_pdc_final)} filas para hospital {hospital_id}.")

            conn.close()
            return df_pdc_final
        else:
            print("No se pudo procesar ningún SKU.")
            conn.close()
            return None
