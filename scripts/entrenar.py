"""Entrena, compara y evalúa los modelos de segmentación de valor.

Uso (desde la raíz del proyecto):
    python -m scripts.entrenar

Flujo: cargar -> limpiar -> separar train/test -> construir target (cortes de train)
-> GridSearchCV por modelo (CV estratificada sobre train) -> elegir por F1 macro de CV
-> evaluar en test -> guardar artefactos en outputs/<run_id>/.
"""

import json
from datetime import datetime

import joblib
import pandas as pd

from src.carga import cargar_dbf
from src.config import CV_FOLDS, OUTPUTS_DIR, RANDOM_STATE, RUTA_DBF, TEST_SIZE
from src.evaluacion import (calcular_metricas, graficar_matriz_confusion,
                            importancia_por_permutacion, tabla_comparativa)
from src.limpieza import COLUMNA_ID, limpiar_datos
from src.logger import crear_logger, generar_run_id
from src.modelado import (FEATURES_CATEGORICAS, FEATURES_NUMERICAS, METRICA_SELECCION,
                          buscar_hiperparametros, elegir_mejor, especificaciones_modelos,
                          preparar_X)
from src.target import COLUMNA_OBJETIVO, COLUMNA_SPLIT, construir_dataset_modelado

NOMBRE_BASELINE = "Dummy"


def main() -> None:
    run_id = generar_run_id("entrenamiento")
    dir_run = OUTPUTS_DIR / run_id
    logger = crear_logger(dir_run / "ejecucion.log")
    logger.info("Ejecución %s: inicio.", run_id)

    # 1. Datos
    df_limpio, registro_limpieza = limpiar_datos(cargar_dbf(RUTA_DBF))
    df, cortes = construir_dataset_modelado(df_limpio)
    train = df[df[COLUMNA_SPLIT] == "train"]
    test = df[df[COLUMNA_SPLIT] == "test"]
    X_train, y_train = preparar_X(train), train[COLUMNA_OBJETIVO]
    X_test, y_test = preparar_X(test), test[COLUMNA_OBJETIVO]

    # 2. Búsqueda de hiperparámetros y selección (solo con train)
    busquedas = [buscar_hiperparametros(e, X_train, y_train) for e in especificaciones_modelos()]
    candidatos = [b for b in busquedas if b.nombre != NOMBRE_BASELINE]
    mejor = elegir_mejor(candidatos)
    logger.info("Selección: %s (mejor %s en CV).", mejor.nombre, METRICA_SELECCION)

    # 3. Evaluación en test
    resultados, predicciones = [], test[[COLUMNA_ID, "BARRIOS_1", "LATITUD", "LONGITUD",
                                         "USD_M2", COLUMNA_OBJETIVO]].copy()
    for b in busquedas:
        y_pred = b.pipeline.predict(X_test)
        metricas = calcular_metricas(y_test, y_pred)
        predicciones[f"PRED_{b.nombre}"] = y_pred
        graficar_matriz_confusion(metricas["matriz_confusion"], metricas["clases"],
                                  f"{b.nombre} — test", dir_run / f"matriz_{b.nombre}.png")
        resultados.append({
            "modelo": b.nombre,
            "mejores_parametros": b.mejores_parametros,
            "cv": b.cv_resumen,
            "test": metricas,
        })
        logger.info("Test: %s -> accuracy %.3f | F1 macro %.3f",
                    b.nombre, metricas["accuracy"], metricas["f1_macro"])

    comparativa = tabla_comparativa(resultados)
    logger.info("Comparativa de modelos:\n%s", comparativa.to_string())

    f1_baseline = comparativa.loc[NOMBRE_BASELINE, "f1_test"]
    if comparativa.loc[mejor.nombre, "f1_test"] <= f1_baseline:
        logger.warning("El modelo elegido no supera a la línea base (F1 test %.3f).", f1_baseline)

    importancias = importancia_por_permutacion(mejor.pipeline, X_test, y_test)
    logger.info("Importancia por permutación (%s):\n%s", mejor.nombre, importancias.round(4).to_string())

    # 4. Artefactos de la ejecución
    resumen = {
        "run_id": run_id,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "configuracion": {
            "test_size": TEST_SIZE, "cv_folds": CV_FOLDS, "random_state": RANDOM_STATE,
            "metrica_seleccion": METRICA_SELECCION,
            "features_numericas": FEATURES_NUMERICAS,
            "features_categoricas": FEATURES_CATEGORICAS,
            "cortes_usd_m2": {"economico_medio": cortes[0], "medio_premium": cortes[1]},
        },
        "datos": {
            "registro_limpieza": registro_limpieza,
            "filas_train": len(train), "filas_test": len(test),
        },
        "modelo_elegido": mejor.nombre,
        "resultados": resultados,
        "importancias": importancias.to_dict(orient="records"),
    }
    (dir_run / "resultados.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    predicciones.to_csv(dir_run / "predicciones.csv", index=False, encoding="utf-8")
    comparativa.to_csv(dir_run / "comparativa.csv", encoding="utf-8")
    joblib.dump(mejor.pipeline, dir_run / "modelo.joblib", compress=3)  # el bosque sin comprimir pesa cientos de MB

    logger.info("Ejecución %s: fin. Artefactos en %s.", run_id, dir_run.relative_to(OUTPUTS_DIR.parent))


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    main()
