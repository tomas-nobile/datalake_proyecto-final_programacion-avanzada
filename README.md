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
│   ├── limpieza.py        # limpieza de datos inválidos (funciones puras)
│   ├── target.py          # split train/test y segmento por terciles (cortes de train)
│   ├── modelado.py        # Pipeline, imputador de coordenadas por barrio, GridSearchCV
│   └── evaluacion.py      # métricas, matriz de confusión, importancia por permutación
├── scripts/
│   └── entrenar.py        # flujo completo con run_id: datos -> modelos -> artefactos
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

## Modelado

- **Partición:** 80/20 estratificada por deciles de USD/m² (`random_state=42`). Los cortes de los
  terciles se calculan **solo con train** (2.727 y 3.489 USD/m²) y se aplican al test.
- **Pipeline:** imputación de coordenadas faltantes con el centroide del barrio (transformer propio,
  aprendido en train) → `SimpleImputer(add_indicator=True)` + `StandardScaler` en numéricas →
  `OneHotEncoder(handle_unknown='ignore')` en categóricas → modelo.
- **Selección:** `GridSearchCV` con `StratifiedKFold(5)` sobre train y F1 macro como criterio;
  el test se usa solo para la evaluación final.

| Modelo | F1 macro CV | Accuracy test | F1 macro test |
|---|---|---|---|
| Dummy (línea base) | 0,167 ± 0,000 | 0,335 | 0,167 |
| Regresión logística | 0,642 ± 0,002 | 0,634 | 0,633 |
| **Random Forest** (elegido) | **0,737 ± 0,001** | **0,746** | **0,746** |
| HistGradientBoosting | 0,716 ± 0,002 | 0,715 | 0,716 |

Las confusiones son casi siempre entre clases vecinas (Económico↔Premium < 4 %). Las variables
con más peso (importancia por permutación) son latitud, antigüedad y longitud.

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
pytest tests                                            # tests
python -m scripts.entrenar                              # entrena y evalúa; artefactos en outputs/<run_id>/
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda.ipynb   # reejecutar el EDA
```

## Estado

🚧 En desarrollo.
