"""Limpieza de datos inválidos del dataset de departamentos en venta (BA Data 2019).

Criterio general (ver notebook 02_eda):
- Aquí solo se corrigen datos **inválidos**: filas que no corresponden, valores
  imposibles y errores de carga. Todas las reglas son fijas (no se "aprenden" de
  los datos), por lo que pueden aplicarse antes de separar train/test sin fuga
  de información.
- La imputación, el escalado y la codificación se hacen dentro del Pipeline de
  scikit-learn, ajustado solo con train.

Cada paso es una función pura: recibe un DataFrame, devuelve uno nuevo y no
modifica el original. ``limpiar_datos`` compone los pasos y registra cuántas
filas quedan después de cada uno.
"""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Constantes de dominio -------------------------------------------------
COLUMNA_ID: str = "ID_PROPIEDAD"

# Rangos válidos por variable (mínimo, máximo). Fuera de rango -> nulo.
LIMITES: dict[str, tuple[float, float]] = {
    "M2CUB": (15, 1000),
    "M2TOTAL": (15, 2000),
    "AMBIENTES": (1, 15),
    "BANOS": (1, 10),
    "ANTIG": (0, 150),
}

# Rango válido del precio por m² cubierto. Límites de dominio fijos en lugar de
# IQR: el IQR (≈ 724 a 5.565) eliminaba el 74 % de Puerto Madero y gran parte de
# Palermo, que son precios reales y no errores de carga.
# Justificación:
# - Index Zonaprop (ago-2026): barrio más barato Lugano ≈ 1.056 USD/m²,
#   más caro Puerto Madero ≈ 6.148 USD/m². En este dataset (2019) las medianas
#   por barrio van de ≈ 963 (Villa Soldati) a ≈ 6.263 (Puerto Madero).
# - Proyectos de lujo publicados en Puerto Madero ≈ 10.000 USD/m².
# - Revisión de extremos: los 7 avisos < 500 y los 3 > 20.000 son errores de
#   carga evidentes (p. ej. 15 m² a USD 590.000); entre 15.000 y 20.000 hay
#   pisos de lujo plausibles en Palermo y Puerto Madero, que se conservan.
LIMITES_USD_M2: tuple[float, float] = (500, 20_000)

# Rectángulo que contiene a CABA. Coordenadas fuera de él -> nulo.
LATITUD_CABA: tuple[float, float] = (-34.71, -34.52)
LONGITUD_CABA: tuple[float, float] = (-58.54, -58.33)

# Tramos de antigüedad en años (límites derechos incluidos).
BORDES_TRAMO_ANTIG: tuple[float, ...] = (-0.5, 0.5, 10.5, 30.5, 60.5, 150.5)
ETIQUETAS_TRAMO_ANTIG: tuple[str, ...] = ("A estrenar", "1 a 10", "11 a 30", "31 a 60", "Más de 60")

COLUMNAS_DESCARTADAS: dict[str, str] = {
    "OPERACION": "Constante tras filtrar (solo ventas).",
    "TIPO": "Constante tras filtrar (solo departamentos).",
    "AGE": "81 % vacía; su único valor (EST = a estrenar) se usa para validar ANTIG = 0 "
           "y queda reflejado en TRAMO_ANTIG.",
    "DIRECCION": "Texto libre; la ubicación ya está en barrio, comuna y coordenadas.",
    "CAMBIO": "Tipo de cambio del mes; formato inconsistente y no describe al inmueble.",
    "PRECIOARS": "Mismo precio en pesos: redundante con PRECIOUSD.",
    "PRECIOARSM": "Mismo precio por m² en pesos: redundante.",
    "PRECIOUSDM": "Calculado con criterio distinto según el mes (FEBRERO sobre m² totales); "
                  "se reemplaza por USD_M2.",
}


# --- Pasos de limpieza (funciones puras) -----------------------------------
def asignar_id(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega un identificador estable por fila (posición en el archivo original).

    Permite vincular cada propiedad entre SQLite y MongoDB.
    """
    return df.reset_index(drop=True).rename_axis(COLUMNA_ID).reset_index()


def filtrar_ventas_departamentos(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo las publicaciones de venta de departamentos (excluye alquileres y edificios)."""
    return df[(df["OPERACION"] == "VTA") & (df["TIPO"] == "DTO")].copy()


def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas repetidas en todas las columnas salvo el identificador."""
    columnas = [c for c in df.columns if c != COLUMNA_ID]
    return df.drop_duplicates(subset=columnas).copy()


def _decodificar_una_vez(texto: str) -> str:
    """Deshace una capa de UTF-8 mal leído como Latin-1/CP1252; si no puede, devuelve el texto."""
    for codec in ("latin-1", "cp1252"):
        try:
            return texto.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return texto


def _corregir_codificacion(texto: object, max_capas: int = 3) -> object:
    """Corrige texto con errores de codificación, incluso si fue mal decodificado varias veces.

    En el dataset aparecen 'NUÃ‘EZ' (una capa) y 'NUÃƒâ€˜EZ' (dos capas); ambos -> 'NUÑEZ'.
    """
    if not isinstance(texto, str):
        return texto
    for _ in range(max_capas):
        if "Ã" not in texto:
            break
        corregido = _decodificar_una_vez(texto)
        if corregido == texto:
            break
        texto = corregido
    return texto


def normalizar_barrios(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige errores de codificación en los nombres de barrio y elimina filas sin barrio.

    Los barrios vacíos del DBF llegan como texto vacío, no como nulo: se tratan como faltantes.
    """
    barrios = (
        df["BARRIOS_1"].map(_corregir_codificacion).str.strip().str.upper().replace("", np.nan)
    )
    return df.assign(BARRIOS_1=barrios).dropna(subset=["BARRIOS_1"])


def anular_antiguedad_sin_dato(df: pd.DataFrame) -> pd.DataFrame:
    """Pasa a nulo ANTIG = 0 cuando el aviso no está marcado como 'a estrenar' (AGE = 'EST').

    Evidencia: en ENERO, MARZO y ABRIL no hay ningún ANTIG = 0 sin la marca EST, y en
    el resto de los meses hay entre 700 y 2.200 por mes. El patrón depende del mes de
    carga, no del inmueble: es un dato faltante cargado como 0. Se imputa en el Pipeline
    (con indicador de faltante).
    """
    sin_dato = (df["ANTIG"] == 0) & (df["AGE"] != "EST")
    return df.assign(ANTIG=df["ANTIG"].mask(sin_dato))


def completar_comunas(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza la comuna 0 (inexistente) por la comuna más frecuente de su barrio."""
    comuna_por_barrio = (
        df[df["COMUNAS"].between(1, 15)]
        .groupby("BARRIOS_1")["COMUNAS"]
        .agg(lambda serie: serie.mode().iloc[0])
    )
    invalida = ~df["COMUNAS"].between(1, 15)
    comunas = df["COMUNAS"].mask(invalida, df["BARRIOS_1"].map(comuna_por_barrio))
    return df.assign(COMUNAS=comunas).dropna(subset=["COMUNAS"]).astype({"COMUNAS": int})


def anular_coordenadas_invalidas(
    df: pd.DataFrame,
    latitud: tuple[float, float] = LATITUD_CABA,
    longitud: tuple[float, float] = LONGITUD_CABA,
) -> pd.DataFrame:
    """Pasa a nulo las coordenadas fuera de CABA (incluye las cargadas en 0)."""
    fuera = ~(df["LATITUD"].between(*latitud) & df["LONGITUD"].between(*longitud))
    return df.assign(
        LATITUD=df["LATITUD"].mask(fuera, np.nan),
        LONGITUD=df["LONGITUD"].mask(fuera, np.nan),
    )


def anular_fuera_de_dominio(
    df: pd.DataFrame, limites: dict[str, tuple[float, float]] = LIMITES
) -> pd.DataFrame:
    """Pasa a nulo los valores fuera de su rango válido (se imputan luego en el Pipeline)."""
    corregidas = {
        columna: df[columna].where(df[columna].between(minimo, maximo))
        for columna, (minimo, maximo) in limites.items()
    }
    return df.assign(**corregidas)


def anular_m2total_inconsistente(df: pd.DataFrame) -> pd.DataFrame:
    """Si la superficie cubierta supera a la total, se anula la total (se conserva la cubierta)."""
    return df.assign(M2TOTAL=df["M2TOTAL"].mask(df["M2CUB"] > df["M2TOTAL"]))


def exigir_superficie_cubierta(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas sin superficie cubierta válida: sin ella no hay precio por m²."""
    return df.dropna(subset=["M2CUB"])


def calcular_usd_m2(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcula el precio por m² cubierto con un único criterio para todos los meses."""
    return df.assign(USD_M2=df["PRECIOUSD"] / df["M2CUB"])


def filtrar_usd_m2(
    df: pd.DataFrame, limites: tuple[float, float] = LIMITES_USD_M2
) -> pd.DataFrame:
    """Elimina precios por m² imposibles para CABA (errores de carga)."""
    return df[df["USD_M2"].between(*limites)].copy()


def agregar_tramo_antiguedad(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega TRAMO_ANTIG: la antigüedad agrupada en tramos interpretables."""
    tramo = pd.cut(
        df["ANTIG"],
        bins=list(BORDES_TRAMO_ANTIG),
        labels=list(ETIQUETAS_TRAMO_ANTIG),
        ordered=True,
    )
    return df.assign(TRAMO_ANTIG=tramo)


def descartar_columnas(
    df: pd.DataFrame, columnas: dict[str, str] = COLUMNAS_DESCARTADAS
) -> pd.DataFrame:
    """Elimina columnas que no aportan al análisis (el motivo de cada una está documentado)."""
    return df.drop(columns=[c for c in columnas if c in df.columns])


# --- Composición -----------------------------------------------------------
Paso = tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]

PASOS_LIMPIEZA: tuple[Paso, ...] = (
    ("Solo ventas de departamentos", filtrar_ventas_departamentos),
    ("Sin duplicados exactos", eliminar_duplicados),
    ("Barrios normalizados", normalizar_barrios),
    ("Comunas completas", completar_comunas),
    ("Coordenadas fuera de CABA a nulo", anular_coordenadas_invalidas),
    ("Antigüedad 0 sin marca 'a estrenar' a nulo", anular_antiguedad_sin_dato),
    ("Valores fuera de dominio a nulo", anular_fuera_de_dominio),
    ("M2TOTAL inconsistente a nulo", anular_m2total_inconsistente),
    ("Con superficie cubierta válida", exigir_superficie_cubierta),
    ("USD_M2 recalculado", calcular_usd_m2),
    ("USD_M2 dentro de límites de dominio", filtrar_usd_m2),
    ("Tramo de antigüedad agregado", agregar_tramo_antiguedad),
    ("Columnas descartadas", descartar_columnas),
    ("Sin duplicados tras descartar columnas", eliminar_duplicados),
)


def limpiar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Aplica todos los pasos de limpieza en orden.

    Args:
        df: DataFrame con las columnas originales del DBF. No se modifica.

    Returns:
        Una tupla con el DataFrame limpio y el registro ``(paso, filas)``
        de cuántas filas quedan después de cada paso.
    """
    logger.info("Limpieza: inicio con %d filas.", len(df))
    resultado = asignar_id(df)
    registro = [("Datos originales", len(resultado))]

    for nombre, paso in PASOS_LIMPIEZA:
        filas_antes = len(resultado)
        resultado = paso(resultado)
        registro.append((nombre, len(resultado)))
        eliminadas = filas_antes - len(resultado)
        if eliminadas:
            logger.info("Limpieza: %s -> %d filas eliminadas.", nombre, eliminadas)
        else:
            logger.debug("Limpieza: %s -> sin filas eliminadas.", nombre)

    nulos = resultado.isna().sum()
    for columna, cantidad in nulos[nulos > 0].items():
        logger.warning("Limpieza: %s con %d nulos (se imputan en el Pipeline).", columna, cantidad)

    logger.info(
        "Limpieza: fin. %d de %d filas conservadas (%.1f %%).",
        len(resultado), len(df), 100 * len(resultado) / max(len(df), 1),
    )
    return resultado.reset_index(drop=True), registro
