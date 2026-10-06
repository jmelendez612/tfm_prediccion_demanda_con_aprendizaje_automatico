from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, timedelta

import pandas as pd

from src.database import DatabaseManager


class CPFRExtractor:
    """Lee únicamente las columnas necesarias para el cálculo CPFR.

    Las condiciones del WHERE evitan aplicar DATE/CAST/SUBSTR sobre columnas
    indexadas. Las normalizaciones se realizan únicamente en el SELECT.
    """

    TAMANO_LOTE_PRODUCTOS = 400

    def __init__(self, db: DatabaseManager | None = None) -> None:
        self.db = db or DatabaseManager()

    @staticmethod
    def _placeholders(values: Sequence[str]) -> str:
        if not values:
            raise ValueError("La lista de valores no puede estar vacía.")
        return ",".join("?" for _ in values)

    @staticmethod
    def _lotes(values: Sequence[str], size: int) -> Iterable[list[str]]:
        for inicio in range(0, len(values), size):
            yield list(values[inicio : inicio + size])

    @staticmethod
    def _limites_fecha(fecha_inicio: date, fecha_fin: date) -> tuple[str, str]:
        """Devuelve un intervalo semiabierto [inicio, día posterior a fin)."""
        return fecha_inicio.isoformat(), (fecha_fin + timedelta(days=1)).isoformat()

    def leer_skus(
        self,
        hospital: str | None = None,
        producto: str | None = None,
    ) -> pd.DataFrame:
        condiciones: list[str] = [
            "UPPER(TRIM(CURVA_PQR)) IN ('P', 'Q', 'R')",
            "SUBSTR(PRODUCTO_ID, 1, 2) = '01'",
            "SUBSTR(HOSPITAL_ID, 1, 2) = '05'",
            "ESTADO_SKU = 1",
            "PLANIFICABILIDAD = 1",
        ]
        parametros: list[str] = []

        if hospital:
            condiciones.append("HOSPITAL_ID = ?")
            parametros.append(str(hospital).zfill(4))
        if producto:
            condiciones.append("PRODUCTO_ID = ?")
            parametros.append(str(producto))

        query = f"""
            SELECT
                PRODUCTO_ID,
                HOSPITAL_ID,
                UPPER(TRIM(CURVA_PQR)) AS CURVA_PQR
            FROM SCM_SKU_CLEAN
            WHERE {' AND '.join(condiciones)}
            ORDER BY HOSPITAL_ID, PRODUCTO_ID
        """

        with self.db.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=parametros, dtype=str)

        if df.empty:
            return df

        df["PRODUCTO_ID"] = df["PRODUCTO_ID"].str.strip()
        df["HOSPITAL_ID"] = df["HOSPITAL_ID"].str.strip().str[-4:].str.zfill(4)
        df["CURVA_PQR"] = df["CURVA_PQR"].str.strip().str.upper()
        return df.drop_duplicates(["HOSPITAL_ID", "PRODUCTO_ID"], keep="first")

    def leer_consumos(
        self,
        fecha_inicio: date,
        fecha_fin: date,
        skus: pd.DataFrame,
    ) -> pd.DataFrame:
        if skus.empty:
            return pd.DataFrame(
                columns=["PRODUCTO_ID", "HOSPITAL_ID", "FECHA", "CONSUMO"]
            )

        fecha_desde, fecha_hasta_exclusiva = self._limites_fecha(fecha_inicio, fecha_fin)
        partes: list[pd.DataFrame] = []

        # El pipeline trabaja por hospital. Mantener esta iteración también hace
        # segura una llamada accidental con más de uno, sin cargar toda la base.
        for hospital, grupo in skus.groupby("HOSPITAL_ID", sort=False):
            productos = sorted(grupo["PRODUCTO_ID"].astype(str).unique().tolist())
            patron_hospital = f"%{str(hospital).zfill(4)}"

            for lote in self._lotes(productos, self.TAMANO_LOTE_PRODUCTOS):
                query = f"""
                    SELECT
                        PRODUCTO_ID,
                        SUBSTR(HOSPITAL_ID, -4) AS HOSPITAL_ID,
                        DATE(FECHA) AS FECHA,
                        -1.0 * CONSUMO_REGISTRADO AS CONSUMO
                    FROM CONSUMOS_HISTORICO
                    WHERE PRODUCTO_ID IN ({self._placeholders(lote)})
                      AND HOSPITAL_ID LIKE ?
                      AND FECHA >= ?
                      AND FECHA < ?
                """
                parametros = [*lote, patron_hospital, fecha_desde, fecha_hasta_exclusiva]
                with self.db.get_connection() as conn:
                    parte = pd.read_sql_query(query, conn, params=parametros)
                if not parte.empty:
                    partes.append(parte)

        if not partes:
            return pd.DataFrame(
                columns=["PRODUCTO_ID", "HOSPITAL_ID", "FECHA", "CONSUMO"]
            )
        return pd.concat(partes, ignore_index=True)

    def leer_stock(
        self,
        fecha_inicio: date,
        fecha_fin: date,
        skus: pd.DataFrame,
    ) -> pd.DataFrame:
        if skus.empty:
            return pd.DataFrame(
                columns=["PRODUCTO_ID", "HOSPITAL_ID", "FECHA", "STOCK"]
            )

        fecha_desde, fecha_hasta_exclusiva = self._limites_fecha(fecha_inicio, fecha_fin)
        partes: list[pd.DataFrame] = []

        for hospital, grupo in skus.groupby("HOSPITAL_ID", sort=False):
            hospital = str(hospital).zfill(4)
            productos = sorted(grupo["PRODUCTO_ID"].astype(str).unique().tolist())

            for lote in self._lotes(productos, self.TAMANO_LOTE_PRODUCTOS):
                query = f"""
                    SELECT
                        PRODUCTO_ID,
                        HOSPITAL_ID,
                        DATE(FECHA) AS FECHA,
                        1.0 * CANTIDAD_STOCK AS STOCK
                    FROM STOCK_HISTORICO
                    WHERE PRODUCTO_ID IN ({self._placeholders(lote)})
                      AND HOSPITAL_ID = ?
                      AND FECHA >= ?
                      AND FECHA < ?
                """
                parametros = [*lote, hospital, fecha_desde, fecha_hasta_exclusiva]
                with self.db.get_connection() as conn:
                    parte = pd.read_sql_query(query, conn, params=parametros)
                if not parte.empty:
                    partes.append(parte)

        if not partes:
            return pd.DataFrame(
                columns=["PRODUCTO_ID", "HOSPITAL_ID", "FECHA", "STOCK"]
            )
        return pd.concat(partes, ignore_index=True)