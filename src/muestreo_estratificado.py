import pandas as pd
import numpy as np
import sqlite3
import math
from src import rutas


class MuestreoEstratificado:
    """
    Selecciona una muestra representativa de SKUs del universo completo,
    estratificada por GRUPO_FARMACOLOGICO, usando el tamaño de muestra
    mínimo estadísticamente defendible (fórmula de Cochran para
    poblaciones finitas).

    Fuente del universo: tabla SCM_SKU_CLEAN (la misma que usa
    DataFeaturing para el Bloque 3), que se asume contiene el catálogo
    completo de SKU x Hospital con su GRUPO_FARMACOLOGICO.
    """

    def __init__(self, db_path=rutas.DB_PATH):
        self.db_path = db_path

    def calcular_tamano_muestra(self, N, confianza=0.95, margen_error=0.05, p=0.5):
        """
        Fórmula de Cochran para poblaciones finitas.

        N            → tamaño del universo total
        confianza    → nivel de confianza (0.95 = 95%)
        margen_error → error máximo tolerado (0.05 = 5%)
        p            → proporción esperada (0.5 = máxima varianza, más conservador)

        Retorna el tamaño de muestra mínimo recomendado (redondeado hacia arriba).
        """
        z_valores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_valores.get(confianza, 1.96)

        n0 = (z ** 2 * p * (1 - p)) / (margen_error ** 2)
        n = n0 / (1 + (n0 - 1) / N)

        return math.ceil(n)

    def _construir_where(self, filtro_hospital_like=None):
        """
        Filtros de negocio confirmados con el usuario:
        - SUBSTR(PRODUCTO_ID,1,2)='01'  → solo medicina
        - ESTADO_SKU = 1                → solo SKUs activos
        - PLANIFICABILIDAD = 1          → solo SKUs planificables

        filtro_hospital_like: ej. "05%" para acotar además a una red
        hospitalaria específica (usa el operador SQL LIKE).
        """
        condiciones = [
            "SUBSTR(PRODUCTO_ID,1,2)='01'",
            "ESTADO_SKU = 1",
            "PLANIFICABILIDAD = 1",
        ]
        params = []
        if filtro_hospital_like:
            condiciones.append("HOSPITAL_ID LIKE ?")
            params.append(filtro_hospital_like)
        return " AND ".join(condiciones), params

    def contar_por_grupo(self, filtro_hospital_like=None):
        """
        Cuenta cuántos SKU x Hospital hay por GRUPO_FARMACOLOGICO en el
        universo (completo o acotado), aplicando los filtros de negocio
        confirmados. Paso 1 — solo metadata.

        filtro_hospital_like: ej. "05%" para acotar a una red hospitalaria
        específica (usa el operador SQL LIKE). None = universo completo
        (dentro de los filtros de negocio).
        """
        conn = sqlite3.connect(self.db_path)
        where, params = self._construir_where(filtro_hospital_like)
        query = f"""
        SELECT GRUPO_FARMACOLOGICO, COUNT(*) AS n_skus
        FROM SCM_SKU_CLEAN
        WHERE {where}
        GROUP BY GRUPO_FARMACOLOGICO
        ORDER BY n_skus DESC
        """
        conteo = pd.read_sql_query(query, conn, params=params)
        conn.close()

        conteo['proporcion'] = (conteo['n_skus'] / conteo['n_skus'].sum()).round(4)
        return conteo

    def generar_muestra(self, n_total=None, confianza=0.95, margen_error=0.05,
                         random_state=42, min_por_grupo=1, filtro_hospital_like=None):
        """
        Genera la muestra estratificada proporcional.

        filtro_hospital_like: ej. "05%" para acotar el universo a una red
        hospitalaria específica antes de calcular Cochran y estratificar.
        Recomendado documentar en el TFM la razón de negocio del recorte
        (ver discusión: reduce la heterogeneidad de hospital como fuente
        de variación adicional, no solo de SKU).
        """
        conn = sqlite3.connect(self.db_path)
        where, params = self._construir_where(filtro_hospital_like)
        query = f"""
        SELECT PRODUCTO_ID, HOSPITAL_ID, GRUPO_FARMACOLOGICO,
               CURVA_PQR, UNIDAD_MEDIDA
        FROM SCM_SKU_CLEAN
        WHERE {where}
        """
        universo = pd.read_sql_query(query, conn, params=params)
        conn.close()

        N = len(universo)

        if n_total is None:
            n_total = self.calcular_tamano_muestra(N, confianza, margen_error)
            print(f"Tamaño de muestra calculado (Cochran, N={N:,}, "
                  f"confianza={confianza:.0%}, margen={margen_error:.0%}): {n_total}")

        conteo = universo.groupby('GRUPO_FARMACOLOGICO').size().reset_index(name='n_universo')
        conteo['proporcion'] = conteo['n_universo'] / N
        conteo['n_muestra'] = (conteo['proporcion'] * n_total).round().astype(int)
        conteo['n_muestra'] = conteo['n_muestra'].clip(lower=min_por_grupo)
        # Ningún grupo puede aportar más SKUs de los que realmente tiene
        conteo['n_muestra'] = conteo[['n_muestra', 'n_universo']].min(axis=1)

        diferencia = n_total - conteo['n_muestra'].sum()
        if diferencia != 0:
            # Ajuste fino: reparte la diferencia en el/los grupo(s) más grandes
            idx_ajuste = conteo.sort_values('n_universo', ascending=False).index[0]
            conteo.loc[idx_ajuste, 'n_muestra'] += diferencia

        print("\nAsignación por grupo farmacológico:")
        print(conteo[['GRUPO_FARMACOLOGICO', 'n_universo', 'proporcion', 'n_muestra']]
              .to_string(index=False))

        muestras = []
        rng = np.random.RandomState(random_state)
        for _, fila in conteo.iterrows():
            grupo = fila['GRUPO_FARMACOLOGICO']
            n_grupo = int(fila['n_muestra'])
            candidatos = universo[universo['GRUPO_FARMACOLOGICO'] == grupo]
            seleccionados = candidatos.sample(n=min(n_grupo, len(candidatos)),
                                               random_state=rng)
            muestras.append(seleccionados)

        muestra_final = pd.concat(muestras, ignore_index=True)

        print(f"\nMuestra final: {len(muestra_final)} SKU x Hospital "
              f"({len(muestra_final)/N:.2%} del universo de {N:,})")

        return muestra_final

    def guardar_muestra(self, muestra_df, nombre_tabla="MUESTRA_SKUS_ENTRENAMIENTO"):
        """Guarda la muestra seleccionada en SQLite para reutilizarla en corridas posteriores."""
        conn = sqlite3.connect(self.db_path)
        muestra_df.to_sql(nombre_tabla, conn, if_exists='replace', index=False)
        conn.close()
        print(f"Muestra guardada en tabla '{nombre_tabla}'.")


if __name__ == "__main__":
    ms = MuestreoEstratificado(db_path=rutas.DB_PATH)

    RED_HOSPITALARIA = "05%"  # medicina + activo + planificable + red 05

    # Paso 1 — ver la distribución real del universo acotado, antes de decidir nada
    print("=== Distribución del universo (medicina, activo, planificable, red 05) ===")
    print(ms.contar_por_grupo(filtro_hospital_like=RED_HOSPITALARIA).to_string(index=False))

    # Paso 2 — generar la muestra (n_total=None → usa Cochran automáticamente sobre N real)
    muestra = ms.generar_muestra(n_total=None, confianza=0.95, margen_error=0.05,
                                  filtro_hospital_like=RED_HOSPITALARIA)

    # Paso 3 — guardar para no tener que re-samplear cada vez
    ms.guardar_muestra(muestra, nombre_tabla="MUESTRA_SKUS_RED05")

    # Lista de PRODUCTO_ID lista para pasar a ImputadorDemanda
    print("\nEjemplo de uso con el preprocesador:")
    print("  skus = muestra['PRODUCTO_ID'].tolist()")
    print("  hospitales = muestra['HOSPITAL_ID'].tolist()")