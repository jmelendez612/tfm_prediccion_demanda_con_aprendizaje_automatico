from __future__ import annotations

from dataclasses import dataclass
from typing import Final


BLOQUES_POR_CURVA: Final[dict[str, dict[str, int]]] = {
    "P": {"dias_bloque": 15, "n_bloques": 12},
    "Q": {"dias_bloque": 15, "n_bloques": 12},
    "R": {"dias_bloque": 30, "n_bloques": 6},
}

DIAS_HISTORICO: Final[int] = 180


@dataclass(frozen=True)
class CPFRConfig:
    fecha_inicio: str
    fecha_fin: str
    hospital: str | None = None
    producto: str | None = None
    limite_por_curva: int | None = 10
    guardar_detalle: bool = True
    tabla_resultado: str = "CPFR_FORECAST_RESULTADO"
    tabla_detalle: str = "CPFR_FORECAST_BLOQUES"
    modo_guardado: str = "replace"

    def validar(self) -> None:
        if self.limite_por_curva is not None and self.limite_por_curva <= 0:
            raise ValueError("limite_por_curva debe ser mayor que cero o None.")
        if self.modo_guardado not in {"replace", "append", "fail"}:
            raise ValueError("modo_guardado debe ser replace, append o fail.")
