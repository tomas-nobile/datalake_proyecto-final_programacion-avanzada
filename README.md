# Segmentación de valor de departamentos en venta — CABA

Proyecto Final de **Programación Avanzada en Ciencia de Datos** (Licenciatura en Ciencia de Datos, UNICABA, 2026).

## Objetivo

Clasificar departamentos en venta de la Ciudad de Buenos Aires en tres segmentos de valor
(**Económico / Medio / Premium**) según superficie, antigüedad, ambientes, baños y ubicación.
El segmento se define por terciles del precio por m² cubierto (calculados solo con el conjunto
de entrenamiento); las columnas de precio **no** se usan como variables de entrada para evitar
*data leakage*.

**Uso previsto:** orientar la tasación antes de publicar y detectar avisos con precio atípico
para sus características.

## Datos

- **Departamentos en venta 2019** — Portal de Datos Abiertos de la Ciudad de Buenos Aires (BA Data):
  <https://data.buenosaires.gob.ar/dataset/departamentos-venta>
  Descargar el archivo `Departamentos_venta_2019.dbf` y ubicarlo en `data/` (no se versiona por su tamaño, 139 MB).
- **Contexto de mercado:** Index Zonaprop CABA, agosto 2026 (`data/_context/`).

## Estructura del proyecto

```
├── data/                  # DBF original (no versionado) y _context/ (Index Zonaprop)
├── notebooks/
│   └── 02_eda.ipynb       # EDA, hallazgos de calidad y decisiones de limpieza
├── src/
│   ├── config.py          # rutas, semillas y constantes
│   ├── logger.py          # logging a consola (INFO) y archivo (DEBUG)
│   ├── carga.py           # lectura del DBF
│   ├── calidad.py         # diagnóstico: auditoría por columna y outliers IQR
│   └── limpieza.py        # limpieza de datos inválidos (funciones puras)
├── tests/                 # tests con datos sintéticos (pytest)
└── outputs/               # logs y artefactos de cada ejecución (no versionado)
```

## Decisiones de limpieza (resumen)

Detalle y evidencia en `notebooks/02_eda.ipynb`. Solo se corrigen datos **inválidos** con reglas
fijas; la imputación, el escalado y la codificación van dentro del `Pipeline` (ajustado con train).

- **Precio por m² recalculado** como `PRECIOUSD / M2CUB`: en FEBRERO `PRECIOUSDM` venía
  calculado sobre superficie total (≈ 9,5 % más bajo).
- **Límites de dominio fijos, no IQR**: el IQR sobre USD/m² eliminaba el 74 % de Puerto Madero
  (precios reales). Se descartan solo avisos fuera de 500–20.000 USD/m² (10 errores de carga).
- **Antigüedad 0 sin marca "a estrenar" → faltante**: en 8 de 11 meses el dato faltante se cargó como 0.
- **Codificación de barrios** corregida (`NUÃ‘EZ` y `NUÃƒâ€˜EZ` → `NUÑEZ`), comuna 0 completada
  por barrio, coordenadas fuera de CABA (SEPTIEMBRE) a nulo.
- Resultado: **85.357 de 88.640 avisos (96,3 %)**.

## Arquitectura de datos

| Base | Tabla / colección | Contenido |
|---|---|---|
| SQLite | `propiedades_limpias` | Datos de entrada limpios, clase objetivo y partición train/test |
| MongoDB Atlas | `predicciones` | Predicción por registro de test, con ubicación GeoJSON (índice `2dsphere`) |
| MongoDB Atlas | `metricas` | Métricas globales y por clase, matriz de confusión embebida |
| MongoDB Atlas | `modelos_config` | Algoritmo, hiperparámetros óptimos y cortes de los terciles |

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # completar credenciales de MongoDB Atlas
```

## Uso

```bash
pytest tests                                            # tests de limpieza
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda.ipynb   # reejecutar el EDA
```

## Estado

🚧 En desarrollo.
