"""Construcción, búsqueda de hiperparámetros y comparación de modelos de clasificación.

Todo el preprocesamiento vive dentro del Pipeline, de modo que en cada fold de la
validación cruzada (y en el ajuste final) se aprende solo con datos de train (M3-2).
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CV_FOLDS, RANDOM_STATE

logger = logging.getLogger(__name__)

# --- Variables de entrada --------------------------------------------------
FEATURES_NUMERICAS: list[str] = [
    "M2CUB", "M2TOTAL", "AMBIENTES", "BANOS", "ANTIG", "LATITUD", "LONGITUD",
]
FEATURES_CATEGORICAS: list[str] = ["BARRIOS_1", "COMUNAS", "TRAMO_ANTIG"]
# MES queda afuera: en el EDA no mostró variación del precio por m² y su relación con
# los faltantes (coordenadas de SEPTIEMBRE, antigüedad) es un artefacto de carga.
CATEGORIA_FALTANTE: str = "Desconocido"

METRICA_SELECCION: str = "f1_macro"
METRICAS_CV: dict[str, object] = {
    "accuracy": "accuracy",
    # zero_division=0: el Dummy nunca predice dos de las clases y su precisión ahí es 0.
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}


def preparar_X(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona las features y deja las categóricas como texto (con NaN donde falta el dato)."""
    categoricas = {
        columna: df[columna].astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))
        for columna in FEATURES_CATEGORICAS
    }
    return df[FEATURES_NUMERICAS].assign(**categoricas)


class ImputadorCoordenadasPorBarrio(BaseEstimator, TransformerMixin):
    """Imputa LATITUD/LONGITUD faltantes con el centroide (mediana) de su barrio.

    El centroide se aprende en ``fit`` (solo con train). Si un barrio no tiene
    coordenadas en train, se usa la mediana global.
    """

    def __init__(self, columna_barrio: str = "BARRIOS_1",
                 columnas_coordenadas: tuple[str, str] = ("LATITUD", "LONGITUD")):
        self.columna_barrio = columna_barrio
        self.columnas_coordenadas = columnas_coordenadas

    def fit(self, X: pd.DataFrame, y=None) -> "ImputadorCoordenadasPorBarrio":
        coordenadas = list(self.columnas_coordenadas)
        self.centroides_ = X.groupby(self.columna_barrio)[coordenadas].median()
        self.mediana_global_ = X[coordenadas].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        resultado = X.copy()
        for columna in self.columnas_coordenadas:
            por_barrio = resultado[self.columna_barrio].map(self.centroides_[columna])
            resultado[columna] = (
                resultado[columna].fillna(por_barrio).fillna(self.mediana_global_[columna])
            )
        return resultado

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def crear_preprocesador() -> Pipeline:
    """Imputación de coordenadas por barrio + imputación/escalado numérico + one-hot categórico."""
    numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("escalar", StandardScaler()),
    ])
    categorico = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value=CATEGORIA_FALTANTE)),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ])
    columnas = ColumnTransformer([
        ("num", numerico, FEATURES_NUMERICAS),
        ("cat", categorico, FEATURES_CATEGORICAS),
    ])
    return Pipeline([
        ("coordenadas", ImputadorCoordenadasPorBarrio()),
        ("columnas", columnas),
    ])


@dataclass(frozen=True)
class EspecificacionModelo:
    """Un algoritmo candidato y la grilla de hiperparámetros a explorar."""

    nombre: str
    estimador: BaseEstimator
    grilla: dict[str, list] = field(default_factory=dict)


def especificaciones_modelos(random_state: int = RANDOM_STATE) -> list[EspecificacionModelo]:
    """Línea base + dos algoritmos vistos en la materia, con grillas acotadas.

    Las claves de las grillas usan el prefijo 'modelo__' (nombre del paso en el Pipeline).
    """
    return [
        EspecificacionModelo("Dummy", DummyClassifier(strategy="most_frequent")),
        EspecificacionModelo(
            "LogisticRegression",
            LogisticRegression(max_iter=3000),
            {"modelo__C": [0.1, 1.0, 10.0]},
        ),
        EspecificacionModelo(
            "RandomForest",
            RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=1),
            {"modelo__max_depth": [None, 25], "modelo__min_samples_leaf": [1, 3]},
        ),
    ]


def crear_pipeline(estimador: BaseEstimator) -> Pipeline:
    """Pipeline completo: preprocesamiento + modelo."""
    return Pipeline([("preprocesamiento", crear_preprocesador()), ("modelo", estimador)])


@dataclass
class ResultadoBusqueda:
    """Resumen de la búsqueda de hiperparámetros de un modelo."""

    nombre: str
    pipeline: Pipeline
    mejores_parametros: dict
    cv_resumen: dict[str, dict[str, float]]


def buscar_hiperparametros(
    especificacion: EspecificacionModelo,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> ResultadoBusqueda:
    """Ejecuta GridSearchCV con validación cruzada estratificada solo sobre train.

    El conjunto de test no participa: la selección se hace con la métrica de CV.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    busqueda = GridSearchCV(
        crear_pipeline(especificacion.estimador),
        param_grid=especificacion.grilla or {},
        scoring=METRICAS_CV,
        refit=METRICA_SELECCION,
        cv=cv,
        n_jobs=n_jobs,
    )
    logger.info("Modelado: %s -> GridSearchCV con %d combinaciones x %d folds.",
                especificacion.nombre, contar_combinaciones(especificacion.grilla), cv_folds)
    busqueda.fit(X_train, y_train)

    mejor = busqueda.best_index_
    cv_resumen = {
        metrica: {
            "media": float(busqueda.cv_results_[f"mean_test_{metrica}"][mejor]),
            "desvio": float(busqueda.cv_results_[f"std_test_{metrica}"][mejor]),
        }
        for metrica in METRICAS_CV
    }
    logger.info("Modelado: %s -> mejores parámetros %s | F1 macro CV = %.3f ± %.3f",
                especificacion.nombre, busqueda.best_params_,
                cv_resumen["f1_macro"]["media"], cv_resumen["f1_macro"]["desvio"])
    return ResultadoBusqueda(
        nombre=especificacion.nombre,
        pipeline=busqueda.best_estimator_,
        mejores_parametros=busqueda.best_params_,
        cv_resumen=cv_resumen,
    )


def contar_combinaciones(grilla: dict[str, list]) -> int:
    """Cantidad de combinaciones de hiperparámetros de una grilla (1 si está vacía)."""
    return math.prod(len(valores) for valores in grilla.values())


def elegir_mejor(resultados: list[ResultadoBusqueda], metrica: str = METRICA_SELECCION) -> ResultadoBusqueda:
    """Elige el modelo con mejor métrica media de CV (sin mirar el test)."""
    return max(resultados, key=lambda r: r.cv_resumen[metrica]["media"])
