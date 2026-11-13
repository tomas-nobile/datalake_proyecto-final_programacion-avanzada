"""Tests de src.persistencia: SQLite real en carpeta temporal y documentos de Mongo (sin conexión)."""

import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from bson import ObjectId

from src.persistencia import (COLUMNAS_PROPIEDADES, SQLiteRepositorio, documento_config,
                              documento_metricas, documentos_predicciones)


@pytest.fixture
def propiedades() -> pd.DataFrame:
    return pd.DataFrame({
        "ID_PROPIEDAD": [10, 11, 12],
        "ANTIG": [5.0, np.nan, 40.0],
        "M2TOTAL": [60.0, np.nan, 80.0],
        "M2CUB": [50.0, 45.0, 70.0],
        "AMBIENTES": [2.0, 2.0, 3.0],
        "BANOS": [1.0, 1.0, 2.0],
        "PRECIOUSD": [150_000.0, 90_000.0, 300_000.0],
        "BARRIOS_1": ["PALERMO", "LUGANO", "NUÑEZ"],
        "COMUNAS": [14, 8, 13],
        "MES": ["JUNIO", "SEPTIEMBRE", "MARZO"],
        "LATITUD": [-34.58, np.nan, -34.54],
        "LONGITUD": [-58.42, np.nan, -58.46],
        "USD_M2": [3000.0, 2000.0, 4285.71],
        "TRAMO_ANTIG": pd.Categorical(["1 a 10", np.nan, "31 a 60"]),
        "SPLIT": ["train", "train", "test"],
        "SEGMENTO": ["Medio", "Económico", "Premium"],
    })


def test_sqlite_guarda_y_lee_sin_perder_datos(tmp_path, propiedades):
    repositorio = SQLiteRepositorio(tmp_path / "prueba.sqlite")
    assert repositorio.guardar_propiedades(propiedades) == 3
    leidas = repositorio.leer_propiedades()
    assert list(leidas.columns) == COLUMNAS_PROPIEDADES
    assert leidas["ID_PROPIEDAD"].tolist() == [10, 11, 12]
    assert pd.isna(leidas.loc[1, "LATITUD"]) and pd.isna(leidas.loc[1, "TRAMO_ANTIG"])
    assert leidas.loc[2, "BARRIOS_1"] == "NUÑEZ"


def test_sqlite_carga_es_idempotente_y_filtra_por_split(tmp_path, propiedades):
    repositorio = SQLiteRepositorio(tmp_path / "prueba.sqlite")
    repositorio.guardar_propiedades(propiedades)
    repositorio.guardar_propiedades(propiedades)  # reemplaza, no duplica
    assert len(repositorio.leer_propiedades()) == 3
    assert repositorio.leer_propiedades(split="test")["ID_PROPIEDAD"].tolist() == [12]


def test_sqlite_rechaza_valores_imposibles(tmp_path, propiedades):
    repositorio = SQLiteRepositorio(tmp_path / "prueba.sqlite")
    with pytest.raises(pd.errors.DatabaseError) as error:  # pandas envuelve el error de SQLite
        repositorio.guardar_propiedades(propiedades.assign(SEGMENTO=["Medio", "Lujo", "Premium"]))
    assert isinstance(error.value.__cause__, sqlite3.IntegrityError)  # lo rechazó el CHECK


def test_sqlite_inexistente_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="cargar_bd"):
        SQLiteRepositorio(tmp_path / "no_existe.sqlite").leer_propiedades()


def test_documento_config_quita_el_prefijo_del_pipeline():
    documento = documento_config("run_1", datetime.now(timezone.utc), "RandomForest",
                                 {"modelo__max_depth": 25}, {}, {}, es_elegido=True)
    assert documento["hiperparametros"] == {"max_depth": 25}
    assert "importancias" not in documento


def test_documento_metricas_embebe_la_matriz():
    metricas = {"accuracy": 0.7, "matriz_confusion": [[1, 0], [0, 1]]}
    documento = documento_metricas("run_1", datetime.now(timezone.utc), "RF", ObjectId(), metricas)
    assert documento["matriz_confusion"] == [[1, 0], [0, 1]]
    assert isinstance(documento["modelo_id"], ObjectId)


def test_documentos_prediccion_con_geojson_y_referencia(propiedades):
    modelo_id = ObjectId()
    documentos = documentos_predicciones("run_1", "RF", modelo_id, propiedades,
                                         ["Medio", "Medio", "Premium"])
    assert [d["acierto"] for d in documentos] == [True, False, True]
    assert documentos[0]["ubicacion"] == {"type": "Point", "coordinates": [-58.42, -34.58]}  # [lon, lat]
    assert "ubicacion" not in documentos[1]  # sin coordenadas, sin GeoJSON
    assert all(d["modelo_id"] == modelo_id for d in documentos)
    assert isinstance(documentos[0]["id_propiedad"], int)
