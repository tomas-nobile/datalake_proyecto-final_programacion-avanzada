"""Configuración de logging del proyecto.

Sigue el esquema visto en clase (M4-1): consola en nivel INFO para seguir la
ejecución y archivo en nivel DEBUG como evidencia completa de cada corrida.
Los módulos de ``src`` obtienen su logger con ``logging.getLogger(__name__)``
(por ejemplo ``src.carga``) y heredan los handlers del logger raíz ``src``.
"""

import logging
from datetime import datetime
from pathlib import Path

FORMATO_LOG: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
NOMBRE_LOGGER_RAIZ: str = "src"


def generar_run_id(prefijo: str = "run") -> str:
    """Devuelve un identificador único de ejecución basado en fecha y hora."""
    return f"{prefijo}_{datetime.now():%Y%m%d_%H%M%S}"


def crear_logger(
    archivo: Path,
    nombre: str = NOMBRE_LOGGER_RAIZ,
    nivel_archivo: int = logging.DEBUG,
) -> logging.Logger:
    """Crea un logger reejecutable (no duplica handlers) con salida a consola y archivo.

    Args:
        archivo: ruta del archivo de log; se crean las carpetas si no existen.
        nombre: nombre del logger. Por defecto el raíz del paquete, para que los
            loggers de cada módulo (``src.*``) escriban en los mismos destinos.
        nivel_archivo: nivel mínimo que se registra en el archivo.

    Returns:
        El logger configurado.
    """
    logger = logging.getLogger(nombre)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Al reejecutar (por ejemplo en Jupyter) se eliminan los handlers previos.
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    archivo.parent.mkdir(parents=True, exist_ok=True)
    formato = logging.Formatter(FORMATO_LOG)

    consola = logging.StreamHandler()
    consola.setLevel(logging.INFO)
    consola.setFormatter(formato)

    fichero = logging.FileHandler(archivo, mode="w", encoding="utf-8")
    fichero.setLevel(nivel_archivo)
    fichero.setFormatter(formato)

    logger.addHandler(consola)
    logger.addHandler(fichero)
    return logger
