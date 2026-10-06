from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.cpfr.config import BLOQUES_POR_CURVA, DIAS_HISTORICO


@dataclass(frozen=True)
class ResultadoBloques:
    promedios_validos: list[float]
    detalle: list[dict]


def construir_bloques(
    serie_diaria: pd.DataFrame,
    fecha_forecast: pd.Timestamp,
    curva: str,
) -> ResultadoBloques:
    configuracion = BLOQUES_POR_CURVA.get(curva)
    if configuracion is None:
        raise ValueError(f"Curva no soportada: {curva!r}")

    dias_bloque = configuracion["dias_bloque"]
    n_bloques = configuracion["n_bloques"]

    fecha_fin_hist = fecha_forecast - pd.Timedelta(days=1)
    fecha_inicio_hist = fecha_forecast - pd.Timedelta(days=DIAS_HISTORICO)

    ventana = serie_diaria.loc[fecha_inicio_hist:fecha_fin_hist]
    if len(ventana) != DIAS_HISTORICO:
        raise ValueError(
            f"La ventana debe tener {DIAS_HISTORICO} días; recibió {len(ventana)}."
        )

    consumos = ventana["CONSUMO"].to_numpy(dtype=float, copy=False)
    stocks = ventana["STOCK"].to_numpy(dtype=float, copy=False)

    promedios_validos: list[float] = []
    detalle: list[dict] = []

    for indice in range(n_bloques):
        inicio = indice * dias_bloque
        fin = inicio + dias_bloque

        suma_consumo = float(np.sum(consumos[inicio:fin]))
        suma_stock = float(np.sum(stocks[inicio:fin]))
        promedio_diario = suma_consumo / dias_bloque
        valido = not np.isclose(suma_consumo + suma_stock, 0.0)

        fecha_inicio_bloque = fecha_inicio_hist + pd.Timedelta(days=inicio)
        fecha_fin_bloque = fecha_inicio_hist + pd.Timedelta(days=fin - 1)

        if valido:
            promedios_validos.append(promedio_diario)

        detalle.append(
            {
                "NUMERO_BLOQUE": indice + 1,
                "FECHA_INICIO_BLOQUE": fecha_inicio_bloque,
                "FECHA_FIN_BLOQUE": fecha_fin_bloque,
                "DIAS_BLOQUE": dias_bloque,
                "SUMA_CONSUMO": suma_consumo,
                "SUMA_STOCK": suma_stock,
                "PROMEDIO_DIARIO": promedio_diario,
                "BLOQUE_VALIDO": int(valido),
            }
        )

    return ResultadoBloques(
        promedios_validos=promedios_validos,
        detalle=detalle,
    )
