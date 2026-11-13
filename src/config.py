"""Configuración central del proyecto: rutas, semillas y constantes.

Todas las rutas se resuelven a partir de la ubicación de este archivo, de modo que
el proyecto funciona igual sin importar desde qué carpeta se lo ejecute.
"""

from pathlib import Path

# --- Rutas -----------------------------------------------------------------
RAIZ: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = RAIZ / "data"
OUTPUTS_DIR: Path = RAIZ / "outputs"

RUTA_DBF: Path = DATA_DIR / "Departamentos_venta_2019.dbf"
ENCODING_DBF: str = "latin-1"

# --- Reproducibilidad ------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# --- Variable objetivo -----------------------------------------------------
# Segmentos de valor definidos por terciles de USD/m² (calculados solo con train).
CLASES: tuple[str, ...] = ("Accesible", "Estándar", "Alta")

# Columnas de precio: se usan para construir la clase y NUNCA como features,
# porque permitirían reconstruir la respuesta (data leakage).
COLUMNAS_PRECIO: tuple[str, ...] = ("PRECIOUSD", "PRECIOARS", "PRECIOUSDM", "PRECIOARSM")
