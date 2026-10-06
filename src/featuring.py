import pandas as pd
import numpy as np
import sqlite3
from datetime import date
from src import rutas

class DataFeaturing:
    def __init__(self, db_path=rutas.DB_PATH):
        self.db_path = db_path

    def clasificar_intermitencia(self, serie_demanda):
        """
        Clasifica un SKU×Hospital según el criterio ADI / CV² de
        Syntetos, Boylan & Croston (2005) — extensión directa de
        Syntetos & Boylan (2001), ya citado en la bibliografía [3].

        Reemplaza a CURVA_PQR como proxy de rotación/intermitencia:
        se comprobó sobre datos reales que CURVA_PQR no correlaciona
        con la frecuencia de demanda (SKUs "P" con 32%-94.6% de ceros).

        ADI (Average Demand Interval): cada cuántos días, en promedio,
        hay demanda no-cero. Valores altos = demanda esporádica.

        CV² (coef. de variación al cuadrado): qué tan variable es el
        TAMAÑO de la demanda cuando ocurre. Valores altos = tamaño errático.

        Umbrales estándar de la literatura: ADI=1.32, CV²=0.49.

        NOTA DE DISEÑO — leakage temporal menor y consciente:
        se calcula sobre TODO el historial disponible del SKU (no solo
        hasta un corte de validación), igual que se trataría una variable
        maestra como CURVA_PQR. Para un Rolling Window estrictamente sin
        fuga, esto debería recalcularse por corte — se deja como
        simplificación documentada, no como limitación oculta.

        Retorna: (adi, cv2, clasificacion)
        """
        demanda = serie_demanda.to_numpy(dtype=float)
        n_total = len(demanda)

        if n_total == 0:
            return np.nan, np.nan, "sin_datos"

        no_cero = demanda[demanda > 0]
        n_no_cero = len(no_cero)

        if n_no_cero == 0:
            # Nunca hubo demanda registrada en todo el historial
            return np.inf, np.nan, "sin_demanda"

        adi = n_total / n_no_cero

        media_no_cero = no_cero.mean()
        std_no_cero = no_cero.std()

        if media_no_cero == 0 or n_no_cero < 2:
            cv2 = 0.0
        else:
            cv2 = (std_no_cero / media_no_cero) ** 2

        if adi < 1.32 and cv2 < 0.49:
            clasificacion = "smooth"
        elif adi < 1.32 and cv2 >= 0.49:
            clasificacion = "erratic"
        elif adi >= 1.32 and cv2 < 0.49:
            clasificacion = "intermittent"
        else:
            clasificacion = "lumpy"

        return round(adi, 4), round(cv2, 4), clasificacion

    def es_festivo_peru(self, fecha: date) -> int:
        # ─────────────────────────────────────────────────────────────────────
        # FESTIVOS PERÚ — Ley N° 27386 y modificatorias
        # Se incluyen los festivos fijos + los móviles más comunes
        # ─────────────────────────────────────────────────────────────────────
        FESTIVOS_PERU = {
            # Fijos (mes, dia)
            (1,  1),   # Año Nuevo
            (5,  1),   # Día del Trabajo
            (6,  7),   # Batalla de Arica (solo Tacna, pero se incluye por completitud)
            (7,  28),  # Fiestas Patrias - Independencia
            (7,  29),  # Fiestas Patrias - Fuerzas Armadas
            (8,  6),   # Batalla de Junín
            (8,  30),  # Santa Rosa de Lima
            (10, 8),   # Combate de Angamos
            (11, 1),   # Todos los Santos
            (12, 8),   # Inmaculada Concepción
            (12, 9),   # Batalla de Ayacucho
            (12, 25),  # Navidad
        }

        # Festivos móviles por año (Jueves/Viernes Santo varían cada año)
        FESTIVOS_MOVILES = {
            date(2022, 4, 14), date(2022, 4, 15),
            date(2023, 4,  6), date(2023, 4,  7),
            date(2024, 3, 28), date(2024, 3, 29),
            date(2025, 4, 17), date(2025, 4, 18),
            date(2026, 4,  2), date(2026, 4,  3),
        }

        """Retorna 1 si la fecha es festivo en Perú, 0 si no."""
        if fecha in FESTIVOS_MOVILES:
            return 1
        if (fecha.month, fecha.day) in FESTIVOS_PERU:
            return 1
        return 0

    def construir_dataset_completo(self, columna_demanda="DEMANDA_IMPUTADA",
                                    nombre_tabla_salida=None):
        """
        Construye el dataset de entrenamiento completo con los 7 bloques de features
        a partir de la tabla PDC_DATASET_ENTRENAMIENTO ya generada por ImputadorDemanda.

        columna_demanda define la demanda base para lags y ventanas móviles
        (bloques 5 y 6), lo que permite generar una tabla por cada imputación
        (mediana o Croston/SBA).

        Uso:
            construir_dataset_completo(columna_demanda="DEMANDA_IMPUTADA")
                → genera DATASET_ENTRENAMIENTO_COMPLETO_MEDIANA

            construir_dataset_completo(columna_demanda="DEMANDA_IMPUTADA_SBA")
                → genera DATASET_ENTRENAMIENTO_COMPLETO_SBA

        Bloques:
            1 — Identificadores
            2 — Variable objetivo Y (incluye AMBAS columnas de demanda imputada,
                para trazabilidad, aunque los lags/rolling solo se calculen
                sobre `columna_demanda`)
            3 — Features del producto
            4 — Features temporales (calendario + festivos Perú)
            5 — Features de lag (sobre `columna_demanda`)
            6 — Features de ventana móvil (sobre `columna_demanda`)
            7 — Features de stock (demanda censurada)
        """
        columnas_demanda_validas = {"DEMANDA_IMPUTADA", "DEMANDA_IMPUTADA_SBA"}
        if columna_demanda not in columnas_demanda_validas:
            raise ValueError(
                f"columna_demanda debe ser una de {columnas_demanda_validas}, "
                f"recibido: '{columna_demanda}'"
            )

        if nombre_tabla_salida is None:
            sufijo = "MEDIANA" if columna_demanda == "DEMANDA_IMPUTADA" else "SBA"
            nombre_tabla_salida = f"DATASET_ENTRENAMIENTO_COMPLETO_{sufijo}"

        print(f"\nIniciando construcción del dataset completo (base: {columna_demanda})...")

        conn = sqlite3.connect(self.db_path)

        # ─────────────────────────────────────────────────────────────────────
        # BLOQUE 1 / 2 / 3 — Carga base desde SQL
        # ─────────────────────────────────────────────────────────────────────
        print("Cargando bloques 1, 2 y 3 desde base de datos...")

        query_base = """
        SELECT
            PDE.FECHA,
            PDE.PRODUCTO_ID,
            PDE.HOSPITAL_ID,
            PDE.CONSUMO_REGISTRADO,
            PDE.CANTIDAD_STOCK,
            PDE.DEMANDA_IMPUTADA,
            PDE.DEMANDA_IMPUTADA_SBA,
            PDE.stock_flag,
            PDE.STOCK_ES_REAL,
            PDE.CONSUMO_ES_REAL,
            SSC.CURVA_PQR,
            SSC.CURVA_XYZ,
            SSC.GRUPO_FARMACOLOGICO,
            SSC.UNIDAD_MEDIDA
        FROM PDC_DATASET_ENTRENAMIENTO PDE
        INNER JOIN SCM_SKU_CLEAN SSC
            ON  SSC.PRODUCTO_ID = PDE.PRODUCTO_ID
            AND SSC.HOSPITAL_ID = PDE.HOSPITAL_ID
        ORDER BY PDE.PRODUCTO_ID, PDE.HOSPITAL_ID, PDE.FECHA ASC
        """

        df = pd.read_sql_query(query_base, conn)
        conn.close()

        if df.empty:
            print("No se encontraron datos en PDC_DATASET_ENTRENAMIENTO.")
            return

        df['FECHA'] = pd.to_datetime(df['FECHA'])
        print(f"  {len(df):,} filas cargadas.")

        # ─────────────────────────────────────────────────────────────────────
        # BLOQUE 4 — Features temporales
        # ─────────────────────────────────────────────────────────────────────
        print("Calculando bloque 4: features temporales...")

        df['dia_semana']    = df['FECHA'].dt.dayofweek
        df['mes']           = df['FECHA'].dt.month
        df['trimestre']     = df['FECHA'].dt.quarter
        df['dia_del_año']   = df['FECHA'].dt.dayofyear
        df['es_fin_semana'] = (df['FECHA'].dt.dayofweek >= 5).astype(int)
        df['es_festivo']    = df['FECHA'].dt.date.apply(self.es_festivo_peru)

        # ─────────────────────────────────────────────────────────────────────
        # BLOQUES 5, 6 y 7 — Features de lag, ventana móvil y stock
        # Todas sobre `columna_demanda` (parametrizada — ver docstring)
        # ─────────────────────────────────────────────────────────────────────
        print(f"Calculando bloques 5, 6 y 7 por SKU-Hospital (base: {columna_demanda})...")

        grupos = df.groupby(['PRODUCTO_ID', 'HOSPITAL_ID'])
        resultados = []

        for (sku, hospital), grupo in grupos:
            g = grupo.sort_values('FECHA').copy()

            # ── BLOQUE 3 (nuevo) — Clasificación de intermitencia ADI/CV² ──────
            # Reemplaza a CURVA_PQR como feature de rotación (ver nota en el
            # docstring de clasificar_intermitencia). Se calcula UNA vez por
            # SKU×Hospital, y se repite en todas sus filas.
            #
            # CORREGIDO tras validación con 359 SKU×Hospital reales:
            # se detectaron 5 de 6 SKUs clasificados "smooth" con 53%-92% de
            # sus días censurados — la cascada de imputación repite el mismo
            # valor de mediana durante los stockouts, lo que fabrica
            # artificialmente varianza cero (falso "smooth"). Por eso la
            # clasificación ahora se calcula SOLO sobre días con demanda
            # genuinamente observada (no imputada).
            #
            # CORREGIDO (2da vuelta) — el filtro stock_flag==1 por sí solo
            # no basta: imputacion_demanda.py imputa un día (con mediana o
            # con Croston/SBA, según la columna) cuando CONSUMO_ES_REAL==0,
            # SIN exigir stock_flag==0 en ese caso. Es decir, existen días
            # con stock_flag==1 (había stock registrado) que igual fueron
            # imputados porque el sistema de consumos falló ese día. Usar
            # solo stock_flag==1 dejaba colar esos valores sintéticos como
            # si fueran consumo crudo, y como la mediana y el Croston/SBA
            # difieren en esos días imputados, la clasificación terminaba
            # dependiendo de qué tabla (MEDIANA vs SBA) se estuviera
            # construyendo — una propiedad del SKU no debería depender del
            # método de imputación. Se usa ahora el mismo filtro estricto
            # de "demanda genuinamente real" que ya usa la cascada de
            # imputación para historico_activo (imputacion_demanda.py,
            # líneas 285-290): stock_flag==1 AND STOCK_ES_REAL==1 AND
            # CONSUMO_ES_REAL==1. Con este filtro, la clasificación coincide
            # entre MEDIANA y SBA (correcto: depende solo del SKU).
            dias_reales = g[
                (g['stock_flag'] == 1) &
                (g['STOCK_ES_REAL'] == 1) &
                (g['CONSUMO_ES_REAL'] == 1)
            ]
            adi, cv2, clasificacion = self.clasificar_intermitencia(dias_reales[columna_demanda])

            # Diagnóstico de confianza — % de la historia que es censura,
            # para que el modelador sepa qué tan fiable es la etiqueta,
            # incluso dentro de una misma categoría.
            pct_censura = (g['stock_flag'] == 0).mean() * 100

            g['adi'] = adi
            g['cv2'] = cv2
            g['clasificacion_demanda'] = clasificacion
            g['pct_censura_historica'] = round(pct_censura, 2)

            # ── BLOQUE 5: Lags ───────────────────────────────────────────────
            demanda = g[columna_demanda]
            g['lag_7']   = demanda.shift(7)
            g['lag_14']  = demanda.shift(14)
            g['lag_30']  = demanda.shift(30)
            g['lag_90']  = demanda.shift(90)
            g['lag_180'] = demanda.shift(180)

            # ── BLOQUE 6: Ventanas móviles ───────────────────────────────────
            g['media_7d']     = demanda.shift(1).rolling(7,   min_periods=1).mean()
            g['mediana_7d']   = demanda.shift(1).rolling(7,   min_periods=1).median()
            g['media_30d']    = demanda.shift(1).rolling(30,  min_periods=1).mean()
            g['mediana_30d']  = demanda.shift(1).rolling(30,  min_periods=1).median()
            g['media_90d']    = demanda.shift(1).rolling(90,  min_periods=1).mean()
            g['mediana_90d']  = demanda.shift(1).rolling(90,  min_periods=1).median()
            g['std_30d']      = demanda.shift(1).rolling(30,  min_periods=2).std().fillna(0)
            g['mediana_180d'] = demanda.shift(1).rolling(180, min_periods=1).median()

            # ── BLOQUE 7: Features de stock ──────────────────────────────────
            sin_stock = (g['stock_flag'] == 0).astype(int)
            g['dias_sin_stock_7d']  = sin_stock.shift(1).rolling(7,  min_periods=1).sum().astype('Int64')
            g['dias_sin_stock_30d'] = sin_stock.shift(1).rolling(30, min_periods=1).sum().astype('Int64')

            resultados.append(g)

        df_final = pd.concat(resultados, ignore_index=True)

        # ─────────────────────────────────────────────────────────────────────
        # Ordenar columnas por bloques para legibilidad
        # ─────────────────────────────────────────────────────────────────────
        columnas_ordenadas = [
            'FECHA', 'PRODUCTO_ID', 'HOSPITAL_ID',
            'CONSUMO_REGISTRADO', 'CANTIDAD_STOCK',
            'DEMANDA_IMPUTADA', 'DEMANDA_IMPUTADA_SBA',
            'CURVA_PQR', 'CURVA_XYZ', 'adi', 'cv2', 'clasificacion_demanda', 'pct_censura_historica',
            'GRUPO_FARMACOLOGICO', 'UNIDAD_MEDIDA',
            'dia_semana', 'mes', 'trimestre', 'dia_del_año', 'es_fin_semana', 'es_festivo',
            'lag_7', 'lag_14', 'lag_30', 'lag_90', 'lag_180',
            'media_7d', 'mediana_7d', 'media_30d', 'mediana_30d',
            'media_90d', 'mediana_90d', 'std_30d', 'mediana_180d',
            'stock_flag', 'STOCK_ES_REAL', 'CONSUMO_ES_REAL',
            'dias_sin_stock_7d', 'dias_sin_stock_30d',
        ]

        df_final = df_final[columnas_ordenadas].sort_values(
            ['PRODUCTO_ID', 'HOSPITAL_ID', 'FECHA']
        ).reset_index(drop=True)

        # ─────────────────────────────────────────────────────────────────────
        # Guardar en SQLite
        # ─────────────────────────────────────────────────────────────────────
        print(f"Guardando dataset final en tabla '{nombre_tabla_salida}'...")

        df_final['FECHA'] = df_final['FECHA'].dt.strftime('%Y-%m-%d')

        conn = sqlite3.connect(self.db_path)
        df_final.to_sql(nombre_tabla_salida, conn, if_exists='replace', index=False)
        conn.close()

        print(f"\n¡Dataset completo generado!")
        print(f"   Base de demanda : {columna_demanda}")
        print(f"   Filas    : {len(df_final):,}")
        print(f"   Columnas : {len(df_final.columns)}")
        print(f"   Tabla    : {nombre_tabla_salida}")

        return df_final


if __name__ == "__main__":
    fe = DataFeaturing(db_path=rutas.DB_PATH)

    # Genera las dos tablas comparables — una por cascada de imputación
    fe.construir_dataset_completo(columna_demanda="DEMANDA_IMPUTADA")
    fe.construir_dataset_completo(columna_demanda="DEMANDA_IMPUTADA_SBA")