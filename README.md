# Segmentación de valor de departamentos en venta — CABA

Proyecto Final de **Programación Avanzada en Ciencia de Datos** (Licenciatura en Ciencia de Datos, UNICABA, 2026).

## Objetivo

Clasificar departamentos en venta de la Ciudad de Buenos Aires en tres segmentos de valor
(**Accesible / Estándar / Alta**) según superficie, antigüedad, ambientes, baños y ubicación.
El segmento se define por terciles de precio en USD/m²; las columnas de precio **no** se usan
como variables de entrada para evitar *data leakage*.

## Datos

- **Departamentos en venta 2019** — Portal de Datos Abiertos de la Ciudad de Buenos Aires (BA Data):
  <https://data.buenosaires.gob.ar/dataset/departamentos-venta>
  Descargar el archivo `Departamentos_venta_2019.dbf` y ubicarlo en `data/` (no se versiona por su tamaño, 139 MB).
- **Contexto de mercado:** Index Zonaprop CABA, agosto 2026 (`data/_context/`).

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

## Estado

🚧 En desarrollo.
