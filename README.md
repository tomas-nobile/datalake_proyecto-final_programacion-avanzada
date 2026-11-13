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
├── data/                  # DBF original (no versionado), tp_final.sqlite y _context/ (Index Zonaprop)
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
│   ├── evaluacion.py      # métricas, matriz de confusión, importancia por permutación
│   └── persistencia.py    # repositorios SQLite y MongoDB, documentos y comprobaciones
├── scripts/
│   ├── cargar_bd.py       # script de carga: datos limpios -> SQLite (Tabla 1)
│   └── entrenar.py        # SQLite -> modelos -> artefactos locales y MongoDB (Tablas 2 y 3)
├── tests/                 # tests con datos sintéticos (pytest)
├── app.py                 # dashboard en Streamlit (MongoDB con usuario de solo lectura + SQLite)
├── presentacion/          # filminas de la defensa (PPTX y PDF)
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

Las confusiones son casi siempre entre clases vecinas (Económico↔Premium < 4 %). Las variables
con más peso (importancia por permutación) son latitud, antigüedad y longitud.

## Bases de datos

```
DBF ─► limpieza ─► split + segmento ─► SQLite: propiedades_limpias (Tabla 1)
                                              │  scripts/cargar_bd.py
                                              ▼
                               Pipeline + GridSearchCV (scripts/entrenar.py)
                                              │
              ┌───────────────────────────────┼──────────────────────────────┐
              ▼                               ▼                              ▼
   Mongo: modelos_config (Tabla 3) ◄── metricas (Tabla 2) ◄── predicciones (Tabla 2)
          hiperparámetros, CV,          matriz de confusión     GeoJSON + índice 2dsphere
          importancias                  embebida                 referencia a modelo_id
```

| Base | Tabla / colección | Contenido | Documentos / filas |
|---|---|---|---|
| SQLite | `propiedades_limpias` | Datos limpios, `SPLIT` (train/test) y `SEGMENTO` | 85.357 |
| MongoDB Atlas | `modelos_config` | Algoritmo, hiperparámetros óptimos, resultados de CV, cortes, importancias | 1 por modelo y ejecución |
| MongoDB Atlas | `metricas` | Accuracy, precision, recall, F1 (macro y por clase), matriz de confusión | 1 por modelo y ejecución |
| MongoDB Atlas | `predicciones` | Segmento real y predicho por propiedad de test, ubicación GeoJSON | 17.072 por modelo y ejecución |

### Decisión de modelado (registro según M5-2)

- **Preguntas de acceso frecuentes:** "leer todos los datos limpios para entrenar", "¿qué métricas y
  qué matriz de confusión tuvo cada modelo en una ejecución?", "¿dónde acierta y dónde se equivoca
  el modelo?" (por barrio y por cercanía a un punto).
- **SQLite para la Tabla 1:** datos tabulares de esquema fijo que se leen completos y se filtran por
  columnas. Las restricciones `CHECK` hacen que la base rechace valores imposibles (por ejemplo un
  segmento inexistente o un USD/m² fuera de 500–20.000).
- **MongoDB para resultados:**
  - **Embeber** la matriz de confusión y las métricas por clase en `metricas`: se leen y cambian juntas.
  - **Referenciar** `modelos_config` desde `predicciones` (`modelo_id`): miles de predicciones
    comparten un mismo modelo; duplicar sus hiperparámetros en cada una no tiene sentido.
  - **GeoJSON + índice `2dsphere`** en `predicciones`: permite consultas `$near` ("predicciones a menos
    de 1,5 km del Obelisco"), que SQLite no resuelve sin extensiones.
- **Transformación aplicada:** las coordenadas se guardan como `{"type": "Point", "coordinates": [lon, lat]}`
  y los hiperparámetros sin el prefijo `modelo__` del Pipeline.
- **Comprobaciones después de cada escritura** (`MongoRepositorio.validar_ejecucion`):
  1. columnas esperadas en los documentos de predicción;
  2. sin segmentos imposibles (fuera de Económico/Medio/Premium);
  3. totales reconciliados: cantidad de predicciones = filas de test, y accuracy recalculada con un
     pipeline de agregación = accuracy guardada en `metricas`.
- **Frase de cierre:** *elegiríamos MongoDB para los resultados porque las consultas requieren leer
  cada modelo como una unidad y filtrar por ubicación; no lo elegiríamos para la Tabla 1 si necesitáramos
  integridad referencial entre tablas o consultas SQL ad hoc sobre los datos de entrada.*

### Acceso para la revisión

- **SQLite:** el archivo `data/tp_final.sqlite` está versionado en el repositorio.
- **MongoDB Atlas:** base `tp_final` en el clúster del proyecto. Se comparte por privado un usuario
  **de solo lectura** (`profesor_ro`); con él se puede usar el dashboard o MongoDB Compass.

## Dashboard

`streamlit run app.py` abre un tablero con cinco pestañas:

- **Resumen:** modelo elegido, F1 y accuracy en test contra la línea base, comparación CV vs test,
  métricas por clase e importancia de variables.
- **Matriz de confusión** de cada modelo.
- **Mapa de predicciones:** aciertos y errores sobre el mapa de CABA y % de aciertos por barrio
  (pipeline de agregación de MongoDB).
- **Buscar por zona:** consulta geográfica `$near` sobre el índice `2dsphere` (predicciones a menos
  de N metros de un punto).
- **Datos:** distribución del precio por m² por segmento y composición de los barrios (Tabla 1, SQLite).

Lee MongoDB con el usuario de **solo lectura**, por lo que puede usarse con las credenciales de revisión.

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
python -m scripts.cargar_bd                             # DBF -> limpieza -> SQLite (+ índices en Mongo)
python -m scripts.entrenar                              # entrena, evalúa y guarda resultados en Mongo
python -m scripts.entrenar --muestra 5000               # prueba rápida con una muestra (~20 s)
python -m scripts.entrenar --sin-mongo                  # sin conexión a Atlas (solo outputs/)
streamlit run app.py                                    # dashboard de resultados
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda.ipynb   # reejecutar el EDA
```

## Estado

🚧 En desarrollo.
