from __future__ import annotations

import pandas as pd


COLUMNAS_CLAVE = ["HOSPITAL_ID", "PRODUCTO_ID", "FECHA"]


def seleccionar_muestra_balanceada(
    skus: pd.DataFrame,
    limite_por_curva: int | None,
) -> pd.DataFrame:
    """Toma hasta N SKU de cada curva. Si N es None, conserva todos."""
    if skus.empty or limite_por_curva is None:
        return skus.copy()

    partes = []
    for curva in ("P", "Q", "R"):
        parte = skus.loc[skus["CURVA_PQR"] == curva].head(limite_por_curva)
        if not parte.empty:
            partes.append(parte)

    if not partes:
        return skus.iloc[0:0].copy()

    return pd.concat(partes, ignore_index=True)


def preparar_historicos(
    consumos: pd.DataFrame,
    stock: pd.DataFrame,
    skus: pd.DataFrame,
) -> pd.DataFrame:
    """
    Devuelve únicamente días con algún registro histórico.

    Los días completamente ausentes se completan a cero posteriormente al
    reindexar cada SKU; no se realiza forward fill de stock.
    """
    consumos = consumos.copy()
    stock = stock.copy()
    skus = skus.copy()

    for df in (consumos, stock):
        if not df.empty:
            df["PRODUCTO_ID"] = df["PRODUCTO_ID"].astype(str).str.strip()
            df["HOSPITAL_ID"] = df["HOSPITAL_ID"].astype(str).str.strip().str[-4:]
            df["FECHA"] = pd.to_datetime(df["FECHA"], errors="raise").dt.normalize()

    if not consumos.empty:
        consumos["CONSUMO"] = pd.to_numeric(consumos["CONSUMO"], errors="coerce").fillna(0.0)
        consumos = (
            consumos.groupby(COLUMNAS_CLAVE, as_index=False, sort=False)["CONSUMO"]
            .sum()
        )

    if not stock.empty:
        stock["STOCK"] = pd.to_numeric(stock["STOCK"], errors="coerce").fillna(0.0)
        stock = stock.groupby(COLUMNAS_CLAVE, as_index=False, sort=False)["STOCK"].sum()

    historico = pd.merge(
        consumos,
        stock,
        on=COLUMNAS_CLAVE,
        how="outer",
        validate="one_to_one",
    )

    historico["CONSUMO"] = historico["CONSUMO"].fillna(0.0)
    historico["STOCK"] = historico["STOCK"].fillna(0.0)

    historico = historico.merge(
        skus[["HOSPITAL_ID", "PRODUCTO_ID", "CURVA_PQR"]],
        on=["HOSPITAL_ID", "PRODUCTO_ID"],
        how="inner",
        validate="many_to_one",
    )

    return historico.sort_values(COLUMNAS_CLAVE).reset_index(drop=True)
