"""Persistencia del proyecto: SQLite para los datos limpios y MongoDB Atlas para los resultados.

Decisión de modelado (M5-2, "primero las preguntas, después la tecnología"):

- **SQLite – Tabla 1 `propiedades_limpias`**: datos tabulares de esquema fijo. Se leen
  completos para entrenar y se filtran por columnas (split, segmento, barrio). Las
  restricciones CHECK dejan que la base rechace valores imposibles.
- **MongoDB – `modelos_config`** (Tabla 3): un documento por modelo y ejecución. Cada
  algoritmo tiene hiperparámetros distintos: el esquema flexible evita columnas vacías.
- **MongoDB – `metricas`** (Tabla 2): la matriz de confusión y las métricas por clase se
  **embeben**, porque se leen y se escriben juntas, como una unidad.
- **MongoDB – `predicciones`** (Tabla 2): un documento por propiedad de test y modelo.
  Cada uno **referencia** su configuración (`modelo_id`), porque miles de predicciones
  comparten el mismo modelo. Cada documento lleva su ubicación en GeoJSON con índice
  `2dsphere`, para hacer consultas geográficas ("aciertos cerca de un punto").
"""

import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import ASCENDING, GEOSPHERE, MongoClient

from src.config import CLASES, RAIZ
from src.limpieza import COLUMNA_ID, LIMITES, LIMITES_USD_M2
from src.target import COLUMNA_OBJETIVO, COLUMNA_SPLIT

logger = logging.getLogger(__name__)

# --- SQLite -----------------------------------------------------------------
TABLA_PROPIEDADES: str = "propiedades_limpias"

_CLASES_SQL = ", ".join(f"'{c}'" for c in CLASES)
ESQUEMA_PROPIEDADES: str = f"""
CREATE TABLE {TABLA_PROPIEDADES} (
    {COLUMNA_ID}   INTEGER PRIMARY KEY,
    ANTIG          REAL CHECK (ANTIG IS NULL OR ANTIG BETWEEN {LIMITES['ANTIG'][0]} AND {LIMITES['ANTIG'][1]}),
    M2TOTAL        REAL,
    M2CUB          REAL NOT NULL CHECK (M2CUB BETWEEN {LIMITES['M2CUB'][0]} AND {LIMITES['M2CUB'][1]}),
    AMBIENTES      REAL,
    BANOS          REAL,
    PRECIOUSD      REAL NOT NULL,
    BARRIOS_1      TEXT NOT NULL,
    COMUNAS        INTEGER NOT NULL CHECK (COMUNAS BETWEEN 1 AND 15),
    MES            TEXT NOT NULL,
    LATITUD        REAL,
    LONGITUD       REAL,
    USD_M2         REAL NOT NULL CHECK (USD_M2 BETWEEN {LIMITES_USD_M2[0]} AND {LIMITES_USD_M2[1]}),
    TRAMO_ANTIG    TEXT,
    {COLUMNA_SPLIT}          TEXT NOT NULL CHECK ({COLUMNA_SPLIT} IN ('train', 'test')),
    {COLUMNA_OBJETIVO}       TEXT NOT NULL CHECK ({COLUMNA_OBJETIVO} IN ({_CLASES_SQL}))
)
"""
COLUMNAS_PROPIEDADES: list[str] = [
    COLUMNA_ID, "ANTIG", "M2TOTAL", "M2CUB", "AMBIENTES", "BANOS", "PRECIOUSD", "BARRIOS_1",
    "COMUNAS", "MES", "LATITUD", "LONGITUD", "USD_M2", "TRAMO_ANTIG", COLUMNA_SPLIT, COLUMNA_OBJETIVO,
]
INDICES_SQLITE: tuple[str, ...] = (COLUMNA_SPLIT, COLUMNA_OBJETIVO, "BARRIOS_1")


class SQLiteRepositorio:
    """Lectura y escritura de la Tabla 1 (datos de entrada limpios) en un archivo SQLite."""

    def __init__(self, ruta: Path):
        self.ruta = ruta

    def _conectar(self) -> closing:
        return closing(sqlite3.connect(self.ruta))

    def guardar_propiedades(self, df: pd.DataFrame) -> int:
        """Reemplaza la tabla con el DataFrame recibido (carga completa e idempotente)."""
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        datos = df[COLUMNAS_PROPIEDADES].assign(TRAMO_ANTIG=df["TRAMO_ANTIG"].astype(object))
        with self._conectar() as conexion, conexion:  # el segundo 'with' confirma la transacción
            conexion.execute(f"DROP TABLE IF EXISTS {TABLA_PROPIEDADES}")
            conexion.execute(ESQUEMA_PROPIEDADES)
            for columna in INDICES_SQLITE:
                conexion.execute(
                    f"CREATE INDEX idx_{columna.lower()} ON {TABLA_PROPIEDADES} ({columna})"
                )
            datos.to_sql(TABLA_PROPIEDADES, conexion, if_exists="append", index=False)
        logger.info("SQLite: %d filas guardadas en %s.", len(datos), TABLA_PROPIEDADES)
        return len(datos)

    def leer_propiedades(self, split: str | None = None) -> pd.DataFrame:
        """Lee la tabla completa o solo un split ('train' / 'test')."""
        if not self.ruta.exists():
            raise FileNotFoundError(
                f"No existe la base {self.ruta.name}. Ejecutar antes: python -m scripts.cargar_bd"
            )
        consulta = f"SELECT * FROM {TABLA_PROPIEDADES}"
        parametros: tuple = ()
        if split is not None:
            consulta += f" WHERE {COLUMNA_SPLIT} = ?"
            parametros = (split,)
        with self._conectar() as conexion:
            df = pd.read_sql_query(consulta, conexion, params=parametros)
        logger.info("SQLite: %d filas leídas de %s.", len(df), TABLA_PROPIEDADES)
        return df

    def resumen_por_split(self) -> pd.DataFrame:
        """Cantidad de propiedades por split y segmento (consulta SQL de agregación)."""
        consulta = f"""
            SELECT {COLUMNA_SPLIT}, {COLUMNA_OBJETIVO}, COUNT(*) AS propiedades,
                   ROUND(AVG(USD_M2), 0) AS usd_m2_promedio
            FROM {TABLA_PROPIEDADES}
            GROUP BY {COLUMNA_SPLIT}, {COLUMNA_OBJETIVO}
            ORDER BY {COLUMNA_SPLIT}, usd_m2_promedio
        """
        with self._conectar() as conexion:
            return pd.read_sql_query(consulta, conexion)


# --- MongoDB: construcción de documentos (funciones puras) ------------------
COLECCION_CONFIG: str = "modelos_config"
COLECCION_METRICAS: str = "metricas"
COLECCION_PREDICCIONES: str = "predicciones"
PREFIJO_PARAMETROS: str = "modelo__"


def documento_config(
    run_id: str,
    fecha: datetime,
    modelo: str,
    mejores_parametros: dict,
    cv: dict,
    configuracion: dict,
    es_elegido: bool,
    importancias: list[dict] | None = None,
) -> dict:
    """Documento de `modelos_config`: algoritmo, hiperparámetros óptimos y resultados de CV."""
    parametros = {k.removeprefix(PREFIJO_PARAMETROS): v for k, v in mejores_parametros.items()}
    documento = {
        "run_id": run_id,
        "fecha": fecha,
        "modelo": modelo,
        "es_elegido": es_elegido,
        "hiperparametros": parametros,
        "cv": cv,
        "configuracion": configuracion,
    }
    if importancias is not None:
        documento["importancias"] = importancias
    return documento


def documento_metricas(run_id: str, fecha: datetime, modelo: str, modelo_id: ObjectId,
                       metricas: dict) -> dict:
    """Documento de `metricas`: métricas de test con la matriz de confusión embebida."""
    return {"run_id": run_id, "fecha": fecha, "modelo": modelo, "modelo_id": modelo_id, **metricas}


def documentos_predicciones(
    run_id: str, modelo: str, modelo_id: ObjectId, test: pd.DataFrame, y_pred
) -> list[dict]:
    """Un documento por propiedad de test, con ubicación GeoJSON si tiene coordenadas."""
    documentos = []
    filas = test[[COLUMNA_ID, "BARRIOS_1", "USD_M2", "LATITUD", "LONGITUD", COLUMNA_OBJETIVO]]
    for fila, predicho in zip(filas.itertuples(index=False), y_pred):
        id_propiedad, barrio, usd_m2, latitud, longitud, real = fila
        documento = {
            "run_id": run_id,
            "modelo": modelo,
            "modelo_id": modelo_id,
            "id_propiedad": int(id_propiedad),
            "barrio": barrio,
            "usd_m2": round(float(usd_m2), 2),
            "segmento_real": real,
            "segmento_predicho": str(predicho),
            "acierto": real == predicho,
        }
        if pd.notna(latitud) and pd.notna(longitud):
            # GeoJSON exige el orden [longitud, latitud].
            documento["ubicacion"] = {"type": "Point", "coordinates": [float(longitud), float(latitud)]}
        documentos.append(documento)
    return documentos


# --- MongoDB: repositorio ----------------------------------------------------
def uri_desde_entorno(solo_lectura: bool = False) -> tuple[str, str]:
    """Arma la URI de Atlas con las credenciales del archivo .env (ver .env.example)."""
    load_dotenv(RAIZ / ".env")
    prefijo = "MONGO_RO_" if solo_lectura else "MONGO_"
    try:
        usuario = os.environ[f"{prefijo}USER"]
        clave = os.environ[f"{prefijo}PASSWORD"]
        host = os.environ["MONGO_HOST"]
        base = os.environ["MONGO_DB"]
    except KeyError as error:
        raise RuntimeError(f"Falta la variable {error} en .env (ver .env.example).") from error
    uri = (f"mongodb+srv://{quote_plus(usuario)}:{quote_plus(clave)}@{host}/"
           "?retryWrites=true&w=majority&appName=tp-final")
    return uri, base


class MongoRepositorio:
    """Escritura y consulta de resultados del modelo en MongoDB Atlas."""

    def __init__(self, uri: str, nombre_base: str, timeout_ms: int = 15_000):
        self.cliente = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms, tz_aware=True)
        self.base = self.cliente[nombre_base]

    @classmethod
    def desde_entorno(cls, solo_lectura: bool = False) -> "MongoRepositorio":
        return cls(*uri_desde_entorno(solo_lectura))

    def __enter__(self) -> "MongoRepositorio":
        return self

    def __exit__(self, *_) -> None:
        self.cliente.close()

    def verificar_conexion(self) -> None:
        self.cliente.admin.command("ping")
        logger.info("Mongo: conexión verificada con la base '%s'.", self.base.name)

    def crear_indices(self) -> None:
        """Índices según las consultas frecuentes (por ejecución/modelo y geográficas)."""
        predicciones = self.base[COLECCION_PREDICCIONES]
        predicciones.create_index([("ubicacion", GEOSPHERE)])
        predicciones.create_index([("run_id", ASCENDING), ("modelo", ASCENDING)])
        predicciones.create_index([("id_propiedad", ASCENDING)])
        for nombre in (COLECCION_METRICAS, COLECCION_CONFIG):
            self.base[nombre].create_index(
                [("run_id", ASCENDING), ("modelo", ASCENDING)], unique=True
            )
        logger.info("Mongo: índices creados o verificados.")

    # Escritura
    def guardar_config(self, documento: dict) -> ObjectId:
        return self.base[COLECCION_CONFIG].insert_one(documento).inserted_id

    def guardar_metricas(self, documento: dict) -> ObjectId:
        return self.base[COLECCION_METRICAS].insert_one(documento).inserted_id

    def guardar_predicciones(self, documentos: list[dict]) -> int:
        if not documentos:
            return 0
        resultado = self.base[COLECCION_PREDICCIONES].insert_many(documentos, ordered=False)
        return len(resultado.inserted_ids)

    # Lectura
    def borrar_ejecucion(self, run_id: str) -> dict[str, int]:
        """Elimina todos los documentos de una ejecución (por ejemplo, una corrida de prueba)."""
        borrados = {
            nombre: self.base[nombre].delete_many({"run_id": run_id}).deleted_count
            for nombre in (COLECCION_PREDICCIONES, COLECCION_METRICAS, COLECCION_CONFIG)
        }
        logger.info("Mongo: ejecución %s eliminada %s.", run_id, borrados)
        return borrados

    def ultima_ejecucion(self) -> str | None:
        """run_id de la ejecución completa (no de prueba con --muestra) más reciente."""
        documento = self.base[COLECCION_CONFIG].find_one(
            {"es_elegido": True, "configuracion.muestra": None},
            sort=[("fecha", -1)], projection={"run_id": 1},
        )
        return documento["run_id"] if documento else None

    def configuraciones(self, run_id: str) -> list[dict]:
        return list(self.base[COLECCION_CONFIG].find({"run_id": run_id}, {"_id": 0}))

    def metricas(self, run_id: str) -> list[dict]:
        return list(self.base[COLECCION_METRICAS].find({"run_id": run_id}, {"_id": 0, "modelo_id": 0}))

    def predicciones(self, run_id: str, modelo: str) -> pd.DataFrame:
        cursor = self.base[COLECCION_PREDICCIONES].find(
            {"run_id": run_id, "modelo": modelo}, {"_id": 0, "modelo_id": 0}
        )
        df = pd.DataFrame(list(cursor))
        if "ubicacion" in df:
            coordenadas = df["ubicacion"].map(lambda u: u["coordinates"] if isinstance(u, dict) else [None, None])
            df["longitud"] = coordenadas.str[0]
            df["latitud"] = coordenadas.str[1]
            df = df.drop(columns="ubicacion")
        return df

    def predicciones_cercanas(self, run_id: str, modelo: str, longitud: float, latitud: float,
                              metros: int) -> pd.DataFrame:
        """Consulta geográfica: predicciones a menos de `metros` del punto, de la más cercana a la más lejana."""
        filtro = {
            "run_id": run_id,
            "modelo": modelo,
            "ubicacion": {"$near": {
                "$geometry": {"type": "Point", "coordinates": [longitud, latitud]},
                "$maxDistance": metros,
            }},
        }
        return pd.DataFrame(list(self.base[COLECCION_PREDICCIONES].find(filtro, {"_id": 0, "modelo_id": 0})))

    def aciertos_por_barrio(self, run_id: str, modelo: str) -> pd.DataFrame:
        """Pipeline de agregación: cantidad de predicciones y % de aciertos por barrio."""
        pipeline = [
            {"$match": {"run_id": run_id, "modelo": modelo}},
            {"$group": {"_id": "$barrio", "predicciones": {"$sum": 1},
                        "aciertos": {"$avg": {"$cond": ["$acierto", 1, 0]}}}},
            {"$project": {"_id": 0, "barrio": "$_id", "predicciones": 1,
                          "pct_aciertos": {"$round": [{"$multiply": ["$aciertos", 100]}, 1]}}},
            {"$sort": {"pct_aciertos": 1}},
        ]
        return pd.DataFrame(list(self.base[COLECCION_PREDICCIONES].aggregate(pipeline)))

    def validar_ejecucion(self, run_id: str, filas_test: int, modelos_con_predicciones: list[str]) -> dict[str, bool]:
        """Tres comprobaciones (M5-2): columnas esperadas, valores imposibles y reconciliación con la fuente."""
        predicciones = self.base[COLECCION_PREDICCIONES]
        campos = {"run_id", "modelo", "modelo_id", "id_propiedad", "barrio", "usd_m2",
                  "segmento_real", "segmento_predicho", "acierto"}
        muestra = predicciones.find_one({"run_id": run_id}) or {}
        imposibles = predicciones.count_documents({
            "run_id": run_id,
            "$or": [{"segmento_real": {"$nin": list(CLASES)}},
                    {"segmento_predicho": {"$nin": list(CLASES)}}],
        })
        reconciliado = True
        for modelo in modelos_con_predicciones:
            cantidad = predicciones.count_documents({"run_id": run_id, "modelo": modelo})
            agregado = list(predicciones.aggregate([
                {"$match": {"run_id": run_id, "modelo": modelo}},
                {"$group": {"_id": None, "accuracy": {"$avg": {"$cond": ["$acierto", 1, 0]}}}},
            ]))
            accuracy_guardada = self.base[COLECCION_METRICAS].find_one(
                {"run_id": run_id, "modelo": modelo})["accuracy"]
            reconciliado &= cantidad == filas_test and bool(agregado) and abs(
                agregado[0]["accuracy"] - accuracy_guardada) < 1e-9
        comprobaciones = {
            "columnas_esperadas": campos.issubset(muestra),
            "sin_valores_imposibles": imposibles == 0,
            "totales_reconciliados": reconciliado,
        }
        logger.info("Mongo: comprobaciones de la ejecución %s -> %s", run_id, comprobaciones)
        return comprobaciones
