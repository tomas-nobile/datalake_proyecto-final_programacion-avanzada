"""Tests de src.target y de las piezas propias de src.modelado (datos sintéticos)."""

import numpy as np
import pandas as pd
import pytest

from src.modelado import (FEATURES_CATEGORICAS, FEATURES_NUMERICAS,
                          ImputadorCoordenadasPorBarrio, contar_combinaciones, preparar_X)
from src.target import (COLUMNA_OBJETIVO, COLUMNA_SPLIT, asignar_segmento, calcular_cortes,
                        construir_dataset_modelado, separar_train_test)


@pytest.fixture
def df_limpio() -> pd.DataFrame:
    generador = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "ID_PROPIEDAD": np.arange(n),
        "USD_M2": generador.uniform(1000, 6000, n),
        "M2CUB": generador.uniform(30, 120, n),
        "M2TOTAL": generador.uniform(30, 150, n),
        "AMBIENTES": generador.integers(1, 5, n).astype(float),
        "BANOS": generador.integers(1, 3, n).astype(float),
        "ANTIG": generador.uniform(0, 80, n),
        "LATITUD": generador.uniform(-34.65, -34.55, n),
        "LONGITUD": generador.uniform(-58.50, -58.38, n),
        "BARRIOS_1": generador.choice(["PALERMO", "LUGANO", "CABALLITO"], n),
        "COMUNAS": generador.integers(1, 16, n),
        "TRAMO_ANTIG": pd.Categorical(generador.choice(["1 a 10", "31 a 60"], n)),
    })


def test_split_es_reproducible_y_respeta_la_proporcion(df_limpio):
    a = separar_train_test(df_limpio, test_size=0.2, random_state=42)
    b = separar_train_test(df_limpio, test_size=0.2, random_state=42)
    assert a[COLUMNA_SPLIT].equals(b[COLUMNA_SPLIT])
    assert (a[COLUMNA_SPLIT] == "test").mean() == pytest.approx(0.2, abs=0.01)


def test_cortes_se_calculan_solo_con_train(df_limpio):
    df, cortes = construir_dataset_modelado(df_limpio)
    train = df[df[COLUMNA_SPLIT] == "train"]
    assert cortes == calcular_cortes(train["USD_M2"])
    # En train, cada segmento tiene ~1/3 de las filas.
    proporciones = train[COLUMNA_OBJETIVO].value_counts(normalize=True)
    assert proporciones.between(0.30, 0.37).all()


def test_asignar_segmento_respeta_los_bordes():
    segmento = asignar_segmento(pd.Series([1000.0, 2000.0, 2000.01, 3000.0, 9000.0]), (2000.0, 3000.0))
    assert segmento.tolist() == ["Económico", "Económico", "Medio", "Medio", "Premium"]


def test_no_modifica_el_input(df_limpio):
    copia = df_limpio.copy()
    construir_dataset_modelado(df_limpio)
    pd.testing.assert_frame_equal(df_limpio, copia)


def test_preparar_X_no_incluye_precio_ni_identificador(df_limpio):
    X = preparar_X(df_limpio)
    assert list(X.columns) == FEATURES_NUMERICAS + FEATURES_CATEGORICAS
    assert not {"USD_M2", "PRECIOUSD", "ID_PROPIEDAD"} & set(X.columns)


def test_imputador_de_coordenadas_usa_el_centroide_del_barrio_de_train():
    train = pd.DataFrame({
        "BARRIOS_1": ["PALERMO", "PALERMO", "LUGANO"],
        "LATITUD": [-34.58, -34.60, -34.68],
        "LONGITUD": [-58.42, -58.44, -58.47],
    })
    nuevo = pd.DataFrame({
        "BARRIOS_1": ["PALERMO", "BARRIO_NUEVO"],
        "LATITUD": [np.nan, np.nan],
        "LONGITUD": [np.nan, np.nan],
    })
    imputado = ImputadorCoordenadasPorBarrio().fit(train).transform(nuevo)
    assert imputado.loc[0, "LATITUD"] == pytest.approx(-34.59)       # centroide de Palermo
    assert imputado.loc[1, "LATITUD"] == pytest.approx(-34.60)       # mediana global (barrio desconocido)
    assert nuevo["LATITUD"].isna().all()                             # no modifica el input


def test_contar_combinaciones():
    assert contar_combinaciones({}) == 1
    assert contar_combinaciones({"a": [1, 2], "b": [1, 2, 3]}) == 6
