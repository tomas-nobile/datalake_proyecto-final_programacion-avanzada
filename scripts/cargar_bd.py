"""Script de carga a la base de datos.

Uso (desde la raíz del proyecto):
    python -m scripts.cargar_bd              # SQLite + verificación e índices de MongoDB Atlas
    python -m scripts.cargar_bd --sin-mongo  # solo SQLite (sin conexión a Atlas)

Flujo: DBF -> limpieza -> partición train/test + segmento (cortes de train) ->
SQLite, tabla `propiedades_limpias` (Tabla 1 de la consigna). En MongoDB se verifica
la conexión y se crean los índices de las colecciones de resultados.
"""

import argparse

from src.carga import cargar_dbf
from src.config import OUTPUTS_DIR, RUTA_DBF, RUTA_SQLITE
from src.limpieza import limpiar_datos
from src.logger import crear_logger, generar_run_id
from src.persistencia import MongoRepositorio, SQLiteRepositorio
from src.target import construir_dataset_modelado


def parsear_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carga los datos limpios en la base de datos.")
    parser.add_argument("--sin-mongo", action="store_true",
                        help="no se conecta a MongoDB Atlas (solo carga SQLite)")
    return parser.parse_args(argv)


def main(sin_mongo: bool = False) -> None:
    run_id = generar_run_id("carga")
    logger = crear_logger(OUTPUTS_DIR / run_id / "ejecucion.log")
    logger.info("Ejecución %s: inicio.", run_id)

    df_limpio, _ = limpiar_datos(cargar_dbf(RUTA_DBF))
    df, cortes = construir_dataset_modelado(df_limpio)

    repositorio = SQLiteRepositorio(RUTA_SQLITE)
    guardadas = repositorio.guardar_propiedades(df)

    # Comprobación: lo que se lee de la base coincide con lo que se cargó.
    leidas = repositorio.leer_propiedades()
    if len(leidas) != guardadas or leidas["ID_PROPIEDAD"].nunique() != guardadas:
        logger.error("SQLite: se guardaron %d filas pero se leyeron %d.", guardadas, len(leidas))
        raise RuntimeError("La carga en SQLite no coincide con los datos limpios.")
    logger.info("SQLite: resumen por split y segmento (cortes %.0f y %.0f USD/m²):\n%s",
                *cortes, repositorio.resumen_por_split().to_string(index=False))

    if not sin_mongo:
        with MongoRepositorio.desde_entorno() as mongo:
            mongo.verificar_conexion()
            mongo.crear_indices()

    logger.info("Ejecución %s: fin.", run_id)


if __name__ == "__main__":
    main(parsear_argumentos().sin_mongo)
