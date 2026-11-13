"""Entrena, compara y evalúa los modelos de segmentación de valor.

Uso (desde la raíz del proyecto, después de `python -m scripts.cargar_bd`):
    python -m scripts.entrenar                  # corrida completa (~3-4 min) y guardado en Mongo
    python -m scripts.entrenar --muestra 5000   # prueba rápida con 5.000 avisos al azar
    python -m scripts.entrenar --sin-mongo      # solo artefactos locales

Flujo: leer Tabla 1 (SQLite) -> GridSearchCV por modelo (CV estratificada sobre train)
-> elegir por F1 macro de CV -> evaluar en test -> artefactos en outputs/<run_id>/
-> configuración, métricas y predicciones en MongoDB Atlas (Tablas 2 y 3).
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.config import CV_FOLDS, OUTPUTS_DIR, RANDOM_STATE, RUTA_SQLITE, TEST_SIZE
from src.evaluacion import (calcular_metricas, graficar_matriz_confusion,
                            importancia_por_permutacion, tabla_comparativa)
from src.limpieza import COLUMNA_ID
from src.logger import crear_logger, generar_run_id
from src.modelado import (FEATURES_CATEGORICAS, FEATURES_NUMERICAS, METRICA_SELECCION,
                          ResultadoBusqueda, buscar_hiperparametros, elegir_mejor,
                          especificaciones_modelos, preparar_X)
from src.persistencia import (MongoRepositorio, SQLiteRepositorio, documento_config,
                              documento_metricas, documentos_predicciones)
from src.target import COLUMNA_OBJETIVO, COLUMNA_PRECIO_M2, COLUMNA_SPLIT, calcular_cortes

NOMBRE_BASELINE = "Dummy"


@dataclass
class Ejecucion:
    """Todo lo que produce una corrida de entrenamiento."""

    run_id: str
    fecha: datetime
    configuracion: dict
    test: pd.DataFrame
    busquedas: list[ResultadoBusqueda]
    mejor: ResultadoBusqueda
    resultados: list[dict]
    predicciones: dict[str, list]
    importancias: pd.DataFrame


def parsear_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena, compara y evalúa los modelos de segmentación de valor."
    )
    parser.add_argument("--muestra", type=int, default=None, metavar="N",
                        help="usa solo N avisos (al azar y reproducible) para una prueba rápida")
    parser.add_argument("--sin-mongo", action="store_true",
                        help="no guarda los resultados en MongoDB Atlas")
    argumentos = parser.parse_args(argv)
    if argumentos.muestra is not None and argumentos.muestra <= 0:
        parser.error("--muestra debe ser un entero positivo")
    return argumentos


def cargar_datos(muestra: int | None, logger) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Lee la Tabla 1 de SQLite; los cortes se recalculan con el train completo (como en la carga)."""
    df = SQLiteRepositorio(RUTA_SQLITE).leer_propiedades()
    cortes = calcular_cortes(df.loc[df[COLUMNA_SPLIT] == "train", COLUMNA_PRECIO_M2])
    if muestra:
        total = len(df)
        df = df.sample(min(muestra, total), random_state=RANDOM_STATE)
        logger.warning("Modo prueba: se usan %d de %d avisos. Los resultados no son los finales.",
                       len(df), total)
    return df, cortes


def entrenar_y_evaluar(df: pd.DataFrame, run_id: str, configuracion: dict, dir_run: Path,
                       logger) -> Ejecucion:
    train = df[df[COLUMNA_SPLIT] == "train"]
    test = df[df[COLUMNA_SPLIT] == "test"]
    X_train, y_train = preparar_X(train), train[COLUMNA_OBJETIVO]
    X_test, y_test = preparar_X(test), test[COLUMNA_OBJETIVO]

    # Búsqueda de hiperparámetros y selección: solo con train.
    busquedas = [buscar_hiperparametros(e, X_train, y_train) for e in especificaciones_modelos()]
    mejor = elegir_mejor([b for b in busquedas if b.nombre != NOMBRE_BASELINE])
    logger.info("Selección: %s (mejor %s en CV).", mejor.nombre, METRICA_SELECCION)

    # Evaluación final en test.
    resultados, predicciones = [], {}
    for b in busquedas:
        y_pred = b.pipeline.predict(X_test)
        metricas = calcular_metricas(y_test, y_pred)
        predicciones[b.nombre] = list(y_pred)
        graficar_matriz_confusion(metricas["matriz_confusion"], metricas["clases"],
                                  f"{b.nombre} — test", dir_run / f"matriz_{b.nombre}.png")
        resultados.append({"modelo": b.nombre, "mejores_parametros": b.mejores_parametros,
                           "cv": b.cv_resumen, "test": metricas})
        logger.info("Test: %s -> accuracy %.3f | F1 macro %.3f",
                    b.nombre, metricas["accuracy"], metricas["f1_macro"])

    comparativa = tabla_comparativa(resultados)
    logger.info("Comparativa de modelos:\n%s", comparativa.to_string())
    if comparativa.loc[mejor.nombre, "f1_test"] <= comparativa.loc[NOMBRE_BASELINE, "f1_test"]:
        logger.warning("El modelo elegido no supera a la línea base.")

    importancias = importancia_por_permutacion(mejor.pipeline, X_test, y_test)
    logger.info("Importancia por permutación (%s):\n%s", mejor.nombre,
                importancias.round(4).to_string())

    return Ejecucion(run_id=run_id, fecha=datetime.now(timezone.utc), configuracion=configuracion,
                     test=test, busquedas=busquedas, mejor=mejor, resultados=resultados,
                     predicciones=predicciones, importancias=importancias)


def guardar_artefactos(ejecucion: Ejecucion, dir_run: Path) -> None:
    """JSON (configuración y resultados), CSV (predicciones y comparativa) y modelo (M3-3)."""
    resumen = {
        "run_id": ejecucion.run_id,
        "fecha": ejecucion.fecha.isoformat(timespec="seconds"),
        "configuracion": ejecucion.configuracion,
        "modelo_elegido": ejecucion.mejor.nombre,
        "resultados": ejecucion.resultados,
        "importancias": ejecucion.importancias.to_dict(orient="records"),
    }
    (dir_run / "resultados.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    columnas = [COLUMNA_ID, "BARRIOS_1", "LATITUD", "LONGITUD", COLUMNA_PRECIO_M2, COLUMNA_OBJETIVO]
    ejecucion.test[columnas].assign(
        **{f"PRED_{nombre}": pred for nombre, pred in ejecucion.predicciones.items()}
    ).to_csv(dir_run / "predicciones.csv", index=False, encoding="utf-8")
    tabla_comparativa(ejecucion.resultados).to_csv(dir_run / "comparativa.csv", encoding="utf-8")
    # compress=3: el bosque sin comprimir pesa cientos de MB.
    joblib.dump(ejecucion.mejor.pipeline, dir_run / "modelo.joblib", compress=3)


def guardar_en_mongo(ejecucion: Ejecucion, logger) -> None:
    """Configuración (Tabla 3), métricas y predicciones (Tabla 2) en MongoDB Atlas."""
    modelos_con_predicciones = []
    with MongoRepositorio.desde_entorno() as mongo:
        mongo.crear_indices()
        for b, resultado in zip(ejecucion.busquedas, ejecucion.resultados):
            es_elegido = b.nombre == ejecucion.mejor.nombre
            modelo_id = mongo.guardar_config(documento_config(
                ejecucion.run_id, ejecucion.fecha, b.nombre, b.mejores_parametros, b.cv_resumen,
                ejecucion.configuracion, es_elegido,
                ejecucion.importancias.to_dict(orient="records") if es_elegido else None,
            ))
            mongo.guardar_metricas(documento_metricas(
                ejecucion.run_id, ejecucion.fecha, b.nombre, modelo_id, resultado["test"]))
            if b.nombre != NOMBRE_BASELINE:  # las predicciones constantes del Dummy no aportan
                guardadas = mongo.guardar_predicciones(documentos_predicciones(
                    ejecucion.run_id, b.nombre, modelo_id, ejecucion.test, ejecucion.predicciones[b.nombre]))
                modelos_con_predicciones.append(b.nombre)
                logger.info("Mongo: %d predicciones de %s guardadas.", guardadas, b.nombre)

        comprobaciones = mongo.validar_ejecucion(ejecucion.run_id, len(ejecucion.test),
                                                 modelos_con_predicciones)
    if not all(comprobaciones.values()):
        logger.error("Mongo: fallaron comprobaciones %s", comprobaciones)
        raise RuntimeError("Los datos guardados en MongoDB no pasan las comprobaciones.")


def main(muestra: int | None = None, sin_mongo: bool = False) -> None:
    run_id = generar_run_id("prueba" if muestra else "entrenamiento")
    dir_run = OUTPUTS_DIR / run_id
    logger = crear_logger(dir_run / "ejecucion.log")
    logger.info("Ejecución %s: inicio.", run_id)

    df, cortes = cargar_datos(muestra, logger)
    configuracion = {
        "muestra": muestra, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE, "metrica_seleccion": METRICA_SELECCION,
        "features_numericas": FEATURES_NUMERICAS, "features_categoricas": FEATURES_CATEGORICAS,
        "cortes_usd_m2": {"economico_medio": round(cortes[0], 2), "medio_premium": round(cortes[1], 2)},
        "filas_train": int((df[COLUMNA_SPLIT] == "train").sum()),
        "filas_test": int((df[COLUMNA_SPLIT] == "test").sum()),
    }

    ejecucion = entrenar_y_evaluar(df, run_id, configuracion, dir_run, logger)
    guardar_artefactos(ejecucion, dir_run)
    if not sin_mongo:
        guardar_en_mongo(ejecucion, logger)

    logger.info("Ejecución %s: fin. Artefactos en %s.", run_id, dir_run.relative_to(OUTPUTS_DIR.parent))


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    argumentos = parsear_argumentos()
    main(argumentos.muestra, argumentos.sin_mongo)
