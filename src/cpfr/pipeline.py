from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import pandas as pd

from src.cpfr.bloques import construir_bloques
from src.cpfr.config import CPFRConfig, DIAS_HISTORICO
from src.cpfr.extractor import CPFRExtractor
from src.cpfr.forecast import calcular_qty_forecast
from src.cpfr.preparacion import preparar_historicos, seleccionar_muestra_balanceada
from src.database import DatabaseManager
from src import rutas


def _normalizar_fecha(valor: str, nombre: str) -> pd.Timestamp:
    fecha = pd.to_datetime(valor, errors="raise").normalize()
    if pd.isna(fecha):
        raise ValueError(f"{nombre} no es una fecha válida: {valor!r}")
    return fecha


def ejecutar_cpfr(
    fecha_inicio: str,
    fecha_fin: str,
    hospital: str | None = None,
    producto: str | None = None,
    limite_por_curva: int | None = 10,
    guardar_detalle: bool = True,
    tabla_resultado: str = "CPFR_FORECAST_RESULTADO",
    tabla_detalle: str = "CPFR_FORECAST_BLOQUES",
    modo_guardado: str = "replace",
    db_path: str = rutas.DB_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = CPFRConfig(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        hospital=hospital,
        producto=producto,
        limite_por_curva=limite_por_curva,
        guardar_detalle=guardar_detalle,
        tabla_resultado=tabla_resultado,
        tabla_detalle=tabla_detalle,
        modo_guardado=modo_guardado,
    )
    config.validar()

    fecha_inicio_ts = _normalizar_fecha(fecha_inicio, "fecha_inicio")
    fecha_fin_ts = _normalizar_fecha(fecha_fin, "fecha_fin")
    if fecha_fin_ts < fecha_inicio_ts:
        raise ValueError("fecha_fin no puede ser anterior a fecha_inicio.")

    fecha_inicio_lectura = fecha_inicio_ts - pd.Timedelta(days=DIAS_HISTORICO)
    fecha_fin_lectura = fecha_fin_ts - pd.Timedelta(days=1)

    db = DatabaseManager(db_path=db_path)
    extractor = CPFRExtractor(db)

    inicio_total = perf_counter()
    print("\n[1/5] Seleccionando SKU...")
    skus = extractor.leer_skus(hospital=hospital, producto=producto)
    skus = seleccionar_muestra_balanceada(skus, limite_por_curva)
    if skus.empty:
        raise RuntimeError("No se encontraron SKU P, Q o R para los filtros indicados.")

    print(f"      SKU seleccionados: {len(skus):,}")
    print(skus.groupby("CURVA_PQR").size().rename("CANTIDAD").to_string())
    print(f"      Lectura histórica: {fecha_inicio_lectura.date()} a {fecha_fin_lectura.date()}")

    t0 = perf_counter()
    print("[2/5] Leyendo consumos...")
    consumos = extractor.leer_consumos(
        fecha_inicio=fecha_inicio_lectura.date(),
        fecha_fin=fecha_fin_lectura.date(),
        skus=skus,
    )
    tiempo_consumos = perf_counter() - t0
    print(f"      Filas: {len(consumos):,} | Tiempo: {tiempo_consumos:.2f} s")

    t0 = perf_counter()
    print("[3/5] Leyendo stock...")
    stock = extractor.leer_stock(
        fecha_inicio=fecha_inicio_lectura.date(),
        fecha_fin=fecha_fin_lectura.date(),
        skus=skus,
    )
    tiempo_stock = perf_counter() - t0
    print(f"      Filas: {len(stock):,} | Tiempo: {tiempo_stock:.2f} s")

    t0 = perf_counter()
    print("[4/5] Preparando históricos...")
    historico = preparar_historicos(consumos, stock, skus)
    tiempo_preparacion = perf_counter() - t0
    print(f"      Filas combinadas: {len(historico):,} | Tiempo: {tiempo_preparacion:.2f} s")

    t0 = perf_counter()
    print("[5/5] Calculando qty_forecast...")

    fechas_forecast = pd.date_range(fecha_inicio_ts, fecha_fin_ts, freq="D")
    indice_completo = pd.date_range(fecha_inicio_lectura, fecha_fin_lectura, freq="D")

    resultados: list[dict] = []
    detalles: list[dict] = []

    historicos_por_sku = {
        (hospital_id, producto_id): grupo.set_index("FECHA")[["CONSUMO", "STOCK"]]
        for (hospital_id, producto_id), grupo in historico.groupby(
            ["HOSPITAL_ID", "PRODUCTO_ID"], sort=False
        )
    }

    total_skus = len(skus)
    for posicion, sku in enumerate(skus.itertuples(index=False), start=1):
        clave = (sku.HOSPITAL_ID, sku.PRODUCTO_ID)
        serie = historicos_por_sku.get(clave)

        if serie is None:
            serie = pd.DataFrame(index=indice_completo, data={"CONSUMO": 0.0, "STOCK": 0.0})
        else:
            serie = serie.reindex(indice_completo, fill_value=0.0)

        for fecha_forecast in fechas_forecast:
            bloques = construir_bloques(serie, fecha_forecast, sku.CURVA_PQR)
            qty_forecast, estado = calcular_qty_forecast(bloques.promedios_validos)

            resultados.append(
                {
                    "FECHA_FORECAST": fecha_forecast,
                    "HOSPITAL_ID": sku.HOSPITAL_ID,
                    "PRODUCTO_ID": sku.PRODUCTO_ID,
                    "CURVA_PQR": sku.CURVA_PQR,
                    "QTY_FORECAST": qty_forecast,
                    "BLOQUES_VALIDOS": len(bloques.promedios_validos),
                    "ESTADO_CALCULO": estado,
                }
            )

            if guardar_detalle:
                for bloque in bloques.detalle:
                    detalles.append(
                        {
                            "FECHA_FORECAST": fecha_forecast,
                            "HOSPITAL_ID": sku.HOSPITAL_ID,
                            "PRODUCTO_ID": sku.PRODUCTO_ID,
                            "CURVA_PQR": sku.CURVA_PQR,
                            **bloque,
                        }
                    )

        if posicion == total_skus or posicion % 10 == 0:
            print(f"      Procesados {posicion:,}/{total_skus:,} SKU")

    df_resultados = pd.DataFrame(resultados)
    df_detalles = pd.DataFrame(detalles)
    tiempo_calculo = perf_counter() - t0

    db.save_to_db(df_resultados, tabla_resultado, mode=modo_guardado)
    if guardar_detalle:
        db.save_to_db(df_detalles, tabla_detalle, mode=modo_guardado)

    tiempo_total = perf_counter() - inicio_total
    print("\nCálculo CPFR completado")
    print(f"   Resultados: {len(df_resultados):,}")
    print(f"   Detalles:   {len(df_detalles):,}")
    print(f"   Consumos:   {tiempo_consumos:.2f} s")
    print(f"   Stock:      {tiempo_stock:.2f} s")
    print(f"   Preparación:{tiempo_preparacion:.2f} s")
    print(f"   Cálculo:    {tiempo_calculo:.2f} s")
    print(f"   Total:      {tiempo_total:.2f} s")
    print(f"   Tabla:      {tabla_resultado}")
    if guardar_detalle:
        print(f"   Auditoría:  {tabla_detalle}")

    resumen_estados = df_resultados["ESTADO_CALCULO"].value_counts(dropna=False)
    print("\n   Estados del cálculo:")
    print(resumen_estados.to_string())
    sin_validos = (
        df_resultados.groupby("CURVA_PQR")["BLOQUES_VALIDOS"]
        .agg(["min", "max", "mean"])
        .round(2)
    )
    print("\n   Bloques válidos por curva:")
    print(sin_validos.to_string())

    df_resultados.attrs["metricas"] = {
        "config": asdict(config),
        "segundos_consumos": tiempo_consumos,
        "segundos_stock": tiempo_stock,
        "segundos_preparacion": tiempo_preparacion,
        "segundos_calculo": tiempo_calculo,
        "segundos_total": tiempo_total,
    }

    return df_resultados, df_detalles
