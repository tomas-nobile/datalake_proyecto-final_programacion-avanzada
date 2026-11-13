"""Partición train/test y construcción de la variable objetivo (segmento de valor).

Orden obligatorio para evitar fuga de información (M3-2):
1. Se separa train/test.
2. Los cortes de los terciles de USD/m² se calculan **solo con train**.
3. Esos mismos cortes se aplican a train y a test.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CLASES, RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)

COLUMNA_SPLIT: str = "SPLIT"
COLUMNA_OBJETIVO: str = "SEGMENTO"
COLUMNA_PRECIO_M2: str = "USD_M2"
# Bins de USD/m² usados solo para estratificar el split (no son la variable objetivo).
BINS_ESTRATIFICACION: int = 10


def separar_train_test(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Agrega la columna SPLIT ('train' / 'test') con una partición estratificada y reproducible.

    Se estratifica por deciles de USD/m² para que train y test tengan la misma
    distribución de precios; los deciles no se usan como etiqueta ni como feature.
    """
    estratos = pd.qcut(df[COLUMNA_PRECIO_M2], q=BINS_ESTRATIFICACION, labels=False, duplicates="drop")
    _, indices_test = train_test_split(
        df.index, test_size=test_size, stratify=estratos, random_state=random_state
    )
    split = pd.Series("train", index=df.index).mask(df.index.isin(indices_test), "test")
    logger.info("Split: %d train / %d test (test_size=%.2f, random_state=%d).",
                (split == "train").sum(), (split == "test").sum(), test_size, random_state)
    return df.assign(**{COLUMNA_SPLIT: split})


def calcular_cortes(precio_m2_train: pd.Series) -> tuple[float, float]:
    """Devuelve los cortes de los terciles (percentiles 33,3 y 66,7) del precio por m² de train."""
    inferior, superior = precio_m2_train.quantile([1 / 3, 2 / 3])
    return float(inferior), float(superior)


def asignar_segmento(
    precio_m2: pd.Series,
    cortes: tuple[float, float],
    clases: tuple[str, ...] = CLASES,
) -> pd.Series:
    """Asigna Económico / Medio / Premium según los cortes (límite derecho incluido)."""
    bordes = [-np.inf, cortes[0], cortes[1], np.inf]
    return pd.cut(precio_m2, bins=bordes, labels=list(clases), ordered=True).astype(str)


def construir_dataset_modelado(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Separa train/test y agrega el SEGMENTO usando cortes calculados solo con train.

    Returns:
        El DataFrame con las columnas SPLIT y SEGMENTO, y los cortes usados.
    """
    con_split = separar_train_test(df, test_size, random_state)
    es_train = con_split[COLUMNA_SPLIT] == "train"
    cortes = calcular_cortes(con_split.loc[es_train, COLUMNA_PRECIO_M2])
    logger.info("Target: cortes de terciles (solo train) = %.0f y %.0f USD/m².", *cortes)

    resultado = con_split.assign(
        **{COLUMNA_OBJETIVO: asignar_segmento(con_split[COLUMNA_PRECIO_M2], cortes)}
    )
    distribucion = pd.crosstab(resultado[COLUMNA_OBJETIVO], resultado[COLUMNA_SPLIT], normalize="columns")
    logger.debug("Target: distribución por split =\n%s", distribucion.round(3))
    return resultado, cortes
