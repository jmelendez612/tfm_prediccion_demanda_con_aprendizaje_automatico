"""
Extracción de datos desde el ERP institucional (Oracle) hacia SQLite local.

PLANTILLA ILUSTRATIVA: por confidencialidad, los nombres de esquema, tablas y
columnas del ERP han sido sustituidos por marcadores genéricos (ESQUEMA_ERP,
TABLA_*, COL_*). Para ejecutarla contra un ERP real hay que reemplazarlos por
los nombres reales. Lo que cada consulta DEBE devolver (alias) sí es exacto,
porque es lo que el resto del proyecto espera encontrar en las tablas locales:

    SCM_SKU_CLEAN        HOSPITAL_ID, PRODUCTO_ID, GRUPO_FARMACOLOGICO,
                         DESC_GRUPO_FARMACOLOGICO, PLANIFICABILIDAD,
                         DESC_PLANIFICABILIDAD, CURVA_PQR, CURVA_XYZ,
                         ESTADO_SKU, ID_SKU, UNIDAD_MEDIDA
    STOCK_HISTORICO      HOSPITAL_ID, PRODUCTO_ID, CANTIDAD_STOCK, FECHA
    CONSUMOS_HISTORICO   HOSPITAL_ID, PRODUCTO_ID, CONSUMO_REGISTRADO, FECHA

HOSPITAL_ID es el código de 4 dígitos del hospital y PRODUCTO_ID el código de
9 dígitos del producto (ambos como texto, conservando los ceros a la izquierda).

Esta etapa no es necesaria para reproducir los resultados si ya se dispone de la
base local (ver README). Sin credenciales, las opciones 1-3 solo muestran un aviso.
"""
import pandas as pd

from src.connections import ConnectionManager
from src.database import DatabaseManager


class DataExtractor:
    def __init__(self):
        self.cm = ConnectionManager()
        self.db = DatabaseManager()

    def extraer_maestro_skus(self):
        print("\n>>> Conectando a Oracle para extraer Maestro SKUs...")
        conn = self.cm.get_oracle_connection()
        if not conn:
            return False

        # Reemplazar los marcadores por los nombres reales del ERP.
        query = """
        SELECT
            COL_ESTABLECIMIENTO         AS HOSPITAL_ID,   -- reducir al código de 4 dígitos
            COL_PRODUCTO                AS PRODUCTO_ID,
            COL_GRUPO_FARMACOLOGICO     AS GRUPO_FARMACOLOGICO,
            G3.COL_DESCRIPCION          AS DESC_GRUPO_FARMACOLOGICO,
            COL_PLANIFICABILIDAD        AS PLANIFICABILIDAD,
            G5.COL_DESCRIPCION          AS DESC_PLANIFICABILIDAD,
            COL_CURVA_PQR               AS CURVA_PQR,
            COL_CURVA_XYZ               AS CURVA_XYZ,
            COL_ESTADO                  AS ESTADO_SKU,
            COL_ID_SKU                  AS ID_SKU,
            COL_UNIDAD_MEDIDA           AS UNIDAD_MEDIDA
        FROM ESQUEMA_ERP.TABLA_MAESTRO_SKU S
        INNER JOIN ESQUEMA_ERP.TABLA_GRUPOS G3 ON G3.COL_TIPO_GRUPO = 3 AND G3.COL_CODIGO = S.COL_GRUPO_FARMACOLOGICO
        INNER JOIN ESQUEMA_ERP.TABLA_GRUPOS G5 ON G5.COL_TIPO_GRUPO = 5 AND G5.COL_CODIGO = S.COL_PLANIFICABILIDAD
        """
        try:
            df = pd.read_sql(query, conn)
            self.db.save_to_db(df, "SCM_SKU_CLEAN")
            print(f"Éxito: {len(df)} registros guardados en tabla SCM_SKU_CLEAN.")
            return True
        except Exception as e:
            print(f"Error en la consulta: {e}")
            return False
        finally:
            conn.close()

    # stock
    def extraer_historico_stock(self):
        """
        Descarga controlada por rango de fechas.
        Edita las fechas antes de ejecutar. Los datos se ACUMULAN (append):
        no repetir un mismo rango para evitar duplicados.
        """
        # --- CONFIGURACIÓN MANUAL (EDITAR AQUÍ) ---
        fecha_inicio = "2026-05-01"
        fecha_fin = "2026-06-30"
        # ------------------------------------------

        print(f"\n>>> Iniciando extracción de Stock: {fecha_inicio} al {fecha_fin}")

        conn = self.cm.get_oracle_connection()
        if not conn:
            return

        query = f"""
        SELECT
            COL_ALMACEN     AS HOSPITAL_ID,
            COL_PRODUCTO    AS PRODUCTO_ID,
            COL_STOCK       AS CANTIDAD_STOCK,
            COL_FECHA       AS FECHA
        FROM ESQUEMA_ERP.TABLA_STOCK_DIARIO
        WHERE COL_FECHA >= TO_DATE('{fecha_inicio}', 'YYYY-MM-DD')
        AND COL_FECHA <= TO_DATE('{fecha_fin}', 'YYYY-MM-DD')
        """

        try:
            chunk_size = 200000
            print(f"Descargando en bloques de {chunk_size} filas...")

            for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
                self.db.save_to_db(chunk, "STOCK_HISTORICO", mode="append")
                print(f" [+] {len(chunk)} filas añadidas a STOCK_HISTORICO...")

            print(f"\nCarga finalizada para el periodo {fecha_inicio} a {fecha_fin}")

        except Exception as e:
            print(f"Error durante la extracción: {e}")
        finally:
            conn.close()

    # consumos
    def extraer_historico_consumos(self):
        """
        Extracción de consumos diarios por rango de fechas.
        Edita las fechas antes de ejecutar. Los datos se ACUMULAN (append).
        """
        # --- CONFIGURACIÓN MANUAL (EDITAR AQUÍ) ---
        fecha_inicio = "2026-05-01"
        fecha_fin = "2026-06-30"
        # ------------------------------------------

        print(f"\n>>> Iniciando extracción de Consumos: {fecha_inicio} al {fecha_fin}")

        conn = self.cm.get_oracle_connection()
        if not conn:
            return

        query = f"""
        SELECT
            COL_ESTABLECIMIENTO AS HOSPITAL_ID,
            COL_PRODUCTO        AS PRODUCTO_ID,
            COL_CANTIDAD        AS CONSUMO_REGISTRADO,
            COL_FECHA           AS FECHA
        FROM ESQUEMA_ERP.TABLA_CONSUMOS_DIARIOS
        WHERE COL_FECHA >= TO_DATE('{fecha_inicio}', 'YYYY-MM-DD')
        AND COL_FECHA <= TO_DATE('{fecha_fin}', 'YYYY-MM-DD')
        """

        try:
            chunk_size = 200000
            print(f"Descargando en paquetes de {chunk_size}...")

            for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
                self.db.save_to_db(chunk, "CONSUMOS_HISTORICO", mode="append")
                print(f" [+] {len(chunk)} filas añadidas a CONSUMOS_HISTORICO...")

            print(f"\nCarga exitosa: {fecha_inicio} a {fecha_fin}")

        except Exception as e:
            print(f"Error en extracción de consumos: {e}")
        finally:
            conn.close()
