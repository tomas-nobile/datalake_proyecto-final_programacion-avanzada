"""Carga del dataset original (DBF de BA Data) en un DataFrame.

Este módulo solo lee: no limpia ni transforma. Las decisiones de limpieza
viven en ``src.limpieza``.
"""

import logging
from pathlib import Path

import pandas as pd
from dbfread import DBF

from src.config import ENCODING_DBF

logger = logging.getLogger(__name__)


def cargar_dbf(ruta: Path, encoding: str = ENCODING_DBF) -> pd.DataFrame:
    """Lee un archivo DBF y lo devuelve como DataFrame, sin modificar los datos.

    Args:
        ruta: ruta al archivo ``.dbf``.
        encoding: codificación de los campos de texto.

    Returns:
        DataFrame con una fila por registro y las columnas originales del DBF.

    Raises:
        FileNotFoundError: si el archivo no existe (se registra antes de relanzar).
    """
    logger.info("Carga: inicio. Archivo=%s", ruta)
    if not ruta.exists():
        logger.error("Carga: no se encontró el archivo %s. Ver README (sección Datos).", ruta)
        raise FileNotFoundError(f"No existe el archivo de datos: {ruta}")

    tabla = DBF(ruta, encoding=encoding, ignore_missing_memofile=True)
    df = pd.DataFrame(iter(tabla))

    logger.info("Carga: %d filas y %d columnas leídas.", len(df), df.shape[1])
    logger.debug("Carga: columnas=%s", list(df.columns))
    return df
