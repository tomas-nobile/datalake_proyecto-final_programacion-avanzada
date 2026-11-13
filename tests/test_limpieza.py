"""Tests de src.limpieza con datos sintéticos (no requieren el DBF)."""

import numpy as np
import pandas as pd
import pytest

from src.limpieza import (
    COLUMNA_ID,
    COLUMNAS_DESCARTADAS,
    LIMITES_USD_M2,
    _corregir_codificacion,
    anular_antiguedad_sin_dato,
    anular_coordenadas_invalidas,
    anular_fuera_de_dominio,
    anular_m2total_inconsistente,
    completar_comunas,
    limpiar_datos,
    normalizar_barrios,
)


def fila(**cambios) -> dict:
    """Devuelve un aviso válido; los argumentos reemplazan campos puntuales."""
    base = {
        "OPERACION": "VTA", "TIPO": "DTO", "ANTIG": 10.0, "AGE": "", "M2TOTAL": 60.0,
        "M2CUB": 50.0, "AMBIENTES": 2.0, "BANOS": 1.0, "DIRECCION": "Calle 123",
        "CAMBIO": "43.74", "PRECIOUSD": 150_000.0, "PRECIOARS": 0.0, "PRECIOUSDM": 3000,
        "PRECIOARSM": 0.0, "BARRIOS_1": "PALERMO", "COMUNAS": 14, "MES": "JUNIO",
        "LATITUD": -34.58, "LONGITUD": -58.42,
    }
    return {**base, **cambios}


@pytest.fixture
def df_crudo() -> pd.DataFrame:
    return pd.DataFrame([
        fila(),                                             # válida
        fila(PRECIOUSD=160_000.0),                          # válida
        fila(OPERACION="ALQ"),                              # alquiler -> se elimina
        fila(TIPO="EDIFICIO"),                              # edificio -> se elimina
        fila(M2CUB=0.0),                                    # sin superficie -> se elimina
        fila(PRECIOUSD=1_000_000.0, M2CUB=20.0),            # 50.000 USD/m² -> se elimina
        fila(BARRIOS_1="NUÃ\x91EZ", COMUNAS=0, PRECIOUSD=170_000.0),  # se corrige
        fila(BARRIOS_1="NUÑEZ", COMUNAS=13, PRECIOUSD=180_000.0),
        fila(LATITUD=0.0, LONGITUD=0.0, PRECIOUSD=190_000.0),        # coordenadas -> nulo
    ])


# --- Funciones individuales -------------------------------------------------
@pytest.mark.parametrize("entrada", ["NUÃ\x91EZ", "NUÃƒâ€˜EZ"])
def test_corrige_codificacion_simple_y_doble(entrada):
    assert _corregir_codificacion(entrada) == "NUÑEZ"


def test_codificacion_no_altera_texto_correcto_ni_no_texto():
    assert _corregir_codificacion("PALERMO") == "PALERMO"
    assert _corregir_codificacion(np.nan) is np.nan


def test_barrio_vacio_se_elimina():
    df = pd.DataFrame([fila(BARRIOS_1=""), fila()])
    assert len(normalizar_barrios(df)) == 1


def test_comuna_cero_toma_la_del_barrio():
    df = pd.DataFrame([fila(BARRIOS_1="NUÑEZ", COMUNAS=13), fila(BARRIOS_1="NUÑEZ", COMUNAS=0)])
    assert completar_comunas(df)["COMUNAS"].tolist() == [13, 13]


def test_coordenadas_en_cero_pasan_a_nulo():
    resultado = anular_coordenadas_invalidas(pd.DataFrame([fila(LATITUD=0.0, LONGITUD=0.0)]))
    assert resultado[["LATITUD", "LONGITUD"]].isna().all(axis=None)


def test_fuera_de_dominio_pasa_a_nulo_sin_perder_la_fila():
    resultado = anular_fuera_de_dominio(pd.DataFrame([fila(BANOS=132.0, AMBIENTES=0.0)]))
    assert len(resultado) == 1
    assert resultado[["BANOS", "AMBIENTES"]].isna().all(axis=None)


def test_m2total_menor_que_cubierta_se_anula():
    resultado = anular_m2total_inconsistente(pd.DataFrame([fila(M2TOTAL=40.0, M2CUB=50.0)]))
    assert pd.isna(resultado.loc[0, "M2TOTAL"])


def test_antiguedad_cero_solo_es_valida_si_esta_marcada_a_estrenar():
    df = pd.DataFrame([fila(ANTIG=0.0, AGE="EST"), fila(ANTIG=0.0, AGE="")])
    resultado = anular_antiguedad_sin_dato(df)
    assert resultado.loc[0, "ANTIG"] == 0
    assert pd.isna(resultado.loc[1, "ANTIG"])


# --- Composición completa ---------------------------------------------------
def test_no_modifica_el_dataframe_de_entrada(df_crudo):
    copia = df_crudo.copy()
    limpiar_datos(df_crudo)
    pd.testing.assert_frame_equal(df_crudo, copia)


def test_contrato_de_salida(df_crudo):
    df, registro = limpiar_datos(df_crudo)
    assert df[COLUMNA_ID].is_unique
    assert not df.columns.isin(list(COLUMNAS_DESCARTADAS)).any()
    assert {"USD_M2", "TRAMO_ANTIG"}.issubset(df.columns)
    assert df["USD_M2"].between(*LIMITES_USD_M2).all()
    assert df["COMUNAS"].between(1, 15).all()
    assert registro[0] == ("Datos originales", len(df_crudo))
    assert registro[-1][1] == len(df)


def test_resultado_esperado_sobre_datos_sinteticos(df_crudo):
    df, _ = limpiar_datos(df_crudo)
    assert len(df) == 5  # se eliminan alquiler, edificio, sin superficie y precio absurdo
    assert set(df["BARRIOS_1"]) == {"PALERMO", "NUÑEZ"}
    assert df["USD_M2"].iloc[0] == pytest.approx(150_000 / 50)  # recalculado sobre M2CUB


def test_caso_limite_sin_filas_validas():
    df, registro = limpiar_datos(pd.DataFrame([fila(OPERACION="ALQ")]))
    assert df.empty
    assert registro[-1] == ("Sin duplicados tras descartar columnas", 0)
