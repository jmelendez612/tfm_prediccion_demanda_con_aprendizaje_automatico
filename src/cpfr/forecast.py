from __future__ import annotations

import numpy as np


def calcular_qty_forecast(promedios_validos: list[float]) -> tuple[float, str]:
    if not promedios_validos:
        return np.nan, "SIN_BLOQUES_VALIDOS"

    mediana = float(np.median(np.asarray(promedios_validos, dtype=float)))
    if mediana <= 0:
        return 0.0, "MEDIANA_NO_POSITIVA"

    return mediana, "OK"
