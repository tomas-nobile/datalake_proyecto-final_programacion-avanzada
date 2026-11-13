"""Diagnóstico de calidad de datos para el EDA.

Estas funciones solo describen el dataset; no lo modifican. Las decisiones de
limpieza se aplican en ``src.limpieza``.
"""

import pandas as pd


def auditar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Resume por columna: tipo, nulos, porcentaje de nulos, ceros y valores únicos."""
    numericas = df.select_dtypes("number").columns
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "pct_nulos": (df.isna().mean() * 100).round(2),
        "ceros": (df[numericas] == 0).sum().reindex(df.columns),
        "valores_unicos": df.nunique(),
    })


def resumen_outliers_iqr(
    df: pd.DataFrame, columnas: list[str], factor: float = 1.5
) -> pd.DataFrame:
    """Cuenta outliers por el criterio IQR (fuera de Q1 - factor·IQR y Q3 + factor·IQR).

    Se usa como diagnóstico: en este proyecto la limpieza aplica límites de
    dominio fijos, porque en precios la cola derecha (segmento de lujo) es real.
    """
    filas = []
    for columna in columnas:
        q1, q3 = df[columna].quantile([0.25, 0.75])
        iqr = q3 - q1
        inferior, superior = q1 - factor * iqr, q3 + factor * iqr
        outliers = (~df[columna].between(inferior, superior)) & df[columna].notna()
        filas.append({
            "columna": columna,
            "lim_inferior": inferior,
            "lim_superior": superior,
            "outliers": int(outliers.sum()),
            "pct_outliers": round(100 * outliers.mean(), 2),
        })
    return pd.DataFrame(filas).set_index("columna")
