"""Métricas de evaluación, matriz de confusión e importancia de variables."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: los gráficos se guardan como archivo

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.config import CLASES, RANDOM_STATE

logger = logging.getLogger(__name__)


def calcular_metricas(
    y_real: pd.Series, y_pred: pd.Series, clases: tuple[str, ...] = CLASES
) -> dict:
    """Accuracy, precision, recall y F1 (macro y por clase) y matriz de confusión.

    La matriz usa filas = clase real y columnas = clase predicha, en el orden de ``clases``.
    """
    etiquetas = list(clases)
    reporte = classification_report(
        y_real, y_pred, labels=etiquetas, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "precision_macro": float(reporte["macro avg"]["precision"]),
        "recall_macro": float(reporte["macro avg"]["recall"]),
        "f1_macro": float(reporte["macro avg"]["f1-score"]),
        "por_clase": {
            clase: {
                "precision": float(reporte[clase]["precision"]),
                "recall": float(reporte[clase]["recall"]),
                "f1": float(reporte[clase]["f1-score"]),
                "soporte": int(reporte[clase]["support"]),
            }
            for clase in etiquetas
        },
        "clases": etiquetas,
        "matriz_confusion": confusion_matrix(y_real, y_pred, labels=etiquetas).tolist(),
    }


def graficar_matriz_confusion(
    matriz: list[list[int]], clases: list[str], titulo: str, ruta: Path
) -> Path:
    """Guarda la matriz de confusión (conteos y % por fila real) como imagen PNG."""
    conteos = pd.DataFrame(matriz, index=clases, columns=clases)
    porcentajes = conteos.div(conteos.sum(axis=1), axis=0) * 100
    anotaciones = conteos.astype(str) + "\n(" + porcentajes.round(1).astype(str) + " %)"

    fig, eje = plt.subplots(figsize=(6, 5))
    sns.heatmap(porcentajes, annot=anotaciones, fmt="", cmap="Blues", vmin=0, vmax=100,
                cbar_kws={"label": "% de la clase real"}, ax=eje)
    eje.set_xlabel("Clase predicha")
    eje.set_ylabel("Clase real")
    eje.set_title(titulo)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def importancia_por_permutacion(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    muestra: int = 5000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Caída del F1 macro al desordenar cada variable original (en una muestra del test).

    Mide cuánto depende el modelo de cada variable; no implica causalidad.
    """
    X_muestra = X.sample(min(muestra, len(X)), random_state=random_state)
    y_muestra = y.loc[X_muestra.index]
    resultado = permutation_importance(
        pipeline, X_muestra, y_muestra, scoring="f1_macro",
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "variable": X.columns,
            "caida_f1_media": resultado.importances_mean,
            "caida_f1_desvio": resultado.importances_std,
        })
        .sort_values("caida_f1_media", ascending=False)
        .reset_index(drop=True)
    )


def tabla_comparativa(resultados: list[dict]) -> pd.DataFrame:
    """Tabla con métricas de CV (media ± desvío) y de test para cada modelo."""
    filas = []
    for r in resultados:
        filas.append({
            "modelo": r["modelo"],
            "f1_cv": r["cv"]["f1_macro"]["media"],
            "f1_cv_desvio": r["cv"]["f1_macro"]["desvio"],
            "recall_cv": r["cv"]["recall_macro"]["media"],
            "accuracy_test": r["test"]["accuracy"],
            "precision_test": r["test"]["precision_macro"],
            "recall_test": r["test"]["recall_macro"],
            "f1_test": r["test"]["f1_macro"],
        })
    return pd.DataFrame(filas).set_index("modelo").round(3)
