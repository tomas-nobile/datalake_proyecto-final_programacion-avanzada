"""Dashboard del proyecto: resultados del modelo (MongoDB Atlas) y datos limpios (SQLite).

Uso (desde la raíz del proyecto):
    streamlit run app.py

Lee MongoDB con el usuario de solo lectura (MONGO_RO_USER en .env).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import CLASES, RUTA_SQLITE
from src.persistencia import MongoRepositorio, SQLiteRepositorio

COLORES_SEGMENTO = {"Económico": "#55a868", "Medio": "#4c72b0", "Premium": "#c44e52"}
COLORES_ACIERTO = {"Acierto": "#4c72b0", "Error": "#dd8452"}
CENTRO_CABA = {"lat": -34.615, "lon": -58.445}
LUGARES = {
    "Obelisco": (-34.6037, -58.3816),
    "Plaza Italia (Palermo)": (-34.5810, -58.4210),
    "Puerto Madero (Dique 3)": (-34.6110, -58.3640),
    "Parque Centenario (Caballito)": (-34.6066, -58.4357),
    "Plaza de Flores": (-34.6283, -58.4637),
    "Villa Lugano (Av. Riestra)": (-34.6720, -58.4720),
}

st.set_page_config(page_title="Segmentación de valor — CABA", page_icon="🏙️", layout="wide")


# --- Acceso a datos (con caché) ----------------------------------------------
@st.cache_resource
def repositorio_mongo() -> MongoRepositorio:
    return MongoRepositorio.desde_entorno(solo_lectura=True)


@st.cache_data(ttl=600)
def cargar_resultados() -> tuple[str | None, list[dict], list[dict]]:
    mongo = repositorio_mongo()
    run_id = mongo.ultima_ejecucion()
    if run_id is None:
        return None, [], []
    return run_id, mongo.configuraciones(run_id), mongo.metricas(run_id)


@st.cache_data(ttl=600)
def cargar_predicciones(run_id: str, modelo: str) -> pd.DataFrame:
    return repositorio_mongo().predicciones(run_id, modelo)


@st.cache_data(ttl=600)
def cargar_aciertos_por_barrio(run_id: str, modelo: str) -> pd.DataFrame:
    return repositorio_mongo().aciertos_por_barrio(run_id, modelo)


@st.cache_data
def cargar_propiedades() -> pd.DataFrame:
    return SQLiteRepositorio(RUTA_SQLITE).leer_propiedades()


# --- Gráficos ------------------------------------------------------------------
def grafico_comparativa(configs: list[dict], metricas: list[dict]) -> go.Figure:
    filas = []
    for config in configs:
        test = next(m for m in metricas if m["modelo"] == config["modelo"])
        filas += [
            {"modelo": config["modelo"], "conjunto": "Validación cruzada (train)",
             "F1 macro": config["cv"]["f1_macro"]["media"]},
            {"modelo": config["modelo"], "conjunto": "Test", "F1 macro": test["f1_macro"]},
        ]
    tabla = pd.DataFrame(filas).sort_values("F1 macro")
    figura = px.bar(tabla, x="F1 macro", y="modelo", color="conjunto", barmode="group",
                    orientation="h", text_auto=".3f", range_x=[0, 1],
                    color_discrete_sequence=["#8c8c8c", "#4c72b0"])
    figura.update_layout(height=320, legend_title_text="", yaxis_title="", margin=dict(t=20))
    return figura


def grafico_matriz(metrica: dict) -> go.Figure:
    matriz = pd.DataFrame(metrica["matriz_confusion"], index=metrica["clases"], columns=metrica["clases"])
    porcentaje = matriz.div(matriz.sum(axis=1), axis=0) * 100
    texto = matriz.astype(str) + "<br>(" + porcentaje.round(1).astype(str) + " %)"
    figura = go.Figure(go.Heatmap(
        z=porcentaje.values, x=matriz.columns, y=matriz.index, text=texto.values,
        texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=100,
        colorbar=dict(title="% de la<br>clase real"),
    ))
    figura.update_layout(height=420, xaxis_title="Clase predicha", yaxis_title="Clase real",
                         yaxis_autorange="reversed", margin=dict(t=20))
    return figura


def mapa_predicciones(df: pd.DataFrame, colorear_por: str, zoom: float = 10.8,
                      centro: dict | None = None) -> go.Figure:
    datos = df.dropna(subset=["latitud", "longitud"]).assign(
        resultado=lambda d: d["acierto"].map({True: "Acierto", False: "Error"}))
    colores = COLORES_ACIERTO if colorear_por == "resultado" else COLORES_SEGMENTO
    figura = px.scatter_map(
        datos, lat="latitud", lon="longitud", color=colorear_por, color_discrete_map=colores,
        hover_data={"barrio": True, "segmento_real": True, "segmento_predicho": True,
                    "usd_m2": ":,.0f", "latitud": False, "longitud": False},
        zoom=zoom, center=centro or CENTRO_CABA, height=560, opacity=0.6,
        map_style="carto-positron",
    )
    figura.update_traces(marker=dict(size=5))
    figura.update_layout(margin=dict(t=0, b=0, l=0, r=0), legend_title_text="")
    return figura


# --- Página ---------------------------------------------------------------------
st.title("Segmentación de valor de departamentos en venta — CABA 2019")
st.caption("Proyecto Final · Programación Avanzada en Ciencia de Datos · UNICABA")

try:
    run_id, configs, metricas = cargar_resultados()
except Exception as error:  # sin red, sin credenciales o IP no habilitada en Atlas
    st.error(f"No se pudo conectar a MongoDB Atlas: {error}")
    st.stop()

if run_id is None:
    st.warning("Todavía no hay resultados en MongoDB. Ejecutar `python -m scripts.entrenar`.")
    st.stop()

elegido = next(c for c in configs if c["es_elegido"])
metrica_elegida = next(m for m in metricas if m["modelo"] == elegido["modelo"])
metrica_dummy = next((m for m in metricas if m["modelo"] == "Dummy"), None)
modelos_con_predicciones = [c["modelo"] for c in configs if c["modelo"] != "Dummy"]

tab_resumen, tab_matriz, tab_mapa, tab_cerca, tab_datos = st.tabs(
    ["Resumen", "Matriz de confusión", "Mapa de predicciones", "Buscar por zona", "Datos"])

with tab_resumen:
    columnas = st.columns(4)
    columnas[0].metric("Modelo elegido", elegido["modelo"])
    columnas[1].metric("F1 macro (test)", f"{metrica_elegida['f1_macro']:.3f}",
                       delta=f"{metrica_elegida['f1_macro'] - metrica_dummy['f1_macro']:+.3f} vs. Dummy"
                       if metrica_dummy else None)
    columnas[2].metric("Accuracy (test)", f"{metrica_elegida['accuracy']:.1%}")
    columnas[3].metric("Propiedades de test", f"{sum(metrica_elegida['por_clase'][c]['soporte'] for c in CLASES):,}".replace(",", "."))

    st.subheader("Comparación de modelos")
    st.plotly_chart(grafico_comparativa(configs, metricas), width="stretch")
    st.caption("El modelo se elige por F1 macro en validación cruzada sobre train; "
               "el test se usa solo para la evaluación final.")

    izquierda, derecha = st.columns(2)
    with izquierda:
        st.subheader("Métricas por clase (test)")
        por_clase = pd.DataFrame(metrica_elegida["por_clase"]).T[["precision", "recall", "f1", "soporte"]]
        st.dataframe(por_clase.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}",
                                             "soporte": "{:,.0f}"}), width="stretch")
        st.markdown("**Hiperparámetros elegidos:** " + ", ".join(
            f"`{k} = {v}`" for k, v in elegido["hiperparametros"].items()))
    with derecha:
        st.subheader("Importancia de variables")
        importancias = pd.DataFrame(elegido.get("importancias", []))
        if not importancias.empty:
            figura = px.bar(importancias.sort_values("caida_f1_media"), x="caida_f1_media", y="variable",
                            orientation="h", error_x="caida_f1_desvio",
                            labels={"caida_f1_media": "Caída del F1 al desordenar la variable", "variable": ""})
            figura.update_layout(height=360, margin=dict(t=10))
            st.plotly_chart(figura, width="stretch")
            st.caption("Importancia por permutación en test: mide dependencia del modelo, no causalidad.")

with tab_matriz:
    modelo = st.selectbox("Modelo", [m["modelo"] for m in metricas],
                          index=[m["modelo"] for m in metricas].index(elegido["modelo"]))
    metrica = next(m for m in metricas if m["modelo"] == modelo)
    st.plotly_chart(grafico_matriz(metrica), width="stretch")
    st.caption("Filas: clase real · columnas: clase predicha · % sobre el total de cada clase real.")

with tab_mapa:
    izquierda, derecha = st.columns([1, 3])
    with izquierda:
        modelo_mapa = st.selectbox("Modelo", modelos_con_predicciones,
                                   index=modelos_con_predicciones.index(elegido["modelo"]), key="modelo_mapa")
        colorear = st.radio("Colorear por", ["resultado", "segmento_predicho", "segmento_real"],
                            format_func={"resultado": "Acierto / error", "segmento_predicho": "Segmento predicho",
                                         "segmento_real": "Segmento real"}.get)
        segmentos = st.multiselect("Segmento real", list(CLASES), default=list(CLASES))
    predicciones = cargar_predicciones(run_id, modelo_mapa)
    filtradas = predicciones[predicciones["segmento_real"].isin(segmentos)]
    with izquierda:
        st.metric("Propiedades", f"{len(filtradas):,}".replace(",", "."))
        st.metric("% de aciertos", f"{filtradas['acierto'].mean():.1%}" if len(filtradas) else "—")
        st.caption("Las publicaciones de SEPTIEMBRE no tienen coordenadas y no aparecen en el mapa.")
    with derecha:
        st.plotly_chart(mapa_predicciones(filtradas, colorear), width="stretch")

    st.subheader("Aciertos por barrio")
    st.caption("Pipeline de agregación en MongoDB (`$match` → `$group` → `$project` → `$sort`).")
    por_barrio = cargar_aciertos_por_barrio(run_id, modelo_mapa)
    por_barrio = por_barrio[por_barrio["predicciones"] >= 30]
    figura = px.bar(por_barrio, x="pct_aciertos", y="barrio", orientation="h", height=900,
                    color="pct_aciertos", color_continuous_scale="RdYlBu", range_color=[40, 100],
                    hover_data=["predicciones"], labels={"pct_aciertos": "% de aciertos", "barrio": ""})
    figura.update_layout(margin=dict(t=10), coloraxis_showscale=False)
    st.plotly_chart(figura, width="stretch")
    st.caption("Solo barrios con al menos 30 propiedades en test.")

with tab_cerca:
    st.markdown("Consulta geográfica `$near` sobre el índice `2dsphere` de la colección `predicciones`: "
                "propiedades de test a menos de cierta distancia de un punto, de la más cercana a la más lejana.")
    izquierda, derecha = st.columns([1, 3])
    with izquierda:
        lugar = st.selectbox("Punto de referencia", list(LUGARES))
        metros = st.slider("Radio (metros)", 200, 3000, 1000, step=100)
        modelo_cerca = st.selectbox("Modelo", modelos_con_predicciones,
                                    index=modelos_con_predicciones.index(elegido["modelo"]), key="modelo_cerca")
    latitud, longitud = LUGARES[lugar]
    cercanas = repositorio_mongo().predicciones_cercanas(run_id, modelo_cerca, longitud, latitud, metros)
    with izquierda:
        st.metric("Propiedades encontradas", len(cercanas))
        if len(cercanas):
            st.metric("% de aciertos", f"{cercanas['acierto'].mean():.1%}")
            st.markdown("**Segmento real**")
            st.dataframe(cercanas["segmento_real"].value_counts().reindex(CLASES, fill_value=0),
                         width="stretch")
    with derecha:
        if len(cercanas):
            coordenadas = cercanas["ubicacion"].map(lambda u: u["coordinates"])
            vista = cercanas.assign(longitud=coordenadas.str[0], latitud=coordenadas.str[1])
            st.plotly_chart(mapa_predicciones(vista, "resultado", zoom=14, centro={"lat": latitud, "lon": longitud}),
                            width="stretch")
        else:
            st.info("No hay propiedades de test en ese radio.")

with tab_datos:
    propiedades = cargar_propiedades()
    cortes = elegido["configuracion"]["cortes_usd_m2"]
    st.markdown(
        f"Tabla 1 (`propiedades_limpias`, SQLite): **{len(propiedades):,}** propiedades. ".replace(",", ".")
        + f"Cortes de los terciles calculados con train: **{cortes['economico_medio']:,.0f}** y "
          f"**{cortes['medio_premium']:,.0f}** USD/m².".replace(",", "."))
    izquierda, derecha = st.columns(2)
    with izquierda:
        figura = px.histogram(propiedades, x="USD_M2", color="SEGMENTO", nbins=120, range_x=[0, 9000],
                              color_discrete_map=COLORES_SEGMENTO, category_orders={"SEGMENTO": list(CLASES)},
                              labels={"USD_M2": "USD por m² cubierto", "SEGMENTO": "Segmento"})
        figura.update_layout(height=380, yaxis_title="Propiedades", margin=dict(t=10))
        st.plotly_chart(figura, width="stretch")
    with derecha:
        composicion = (propiedades.groupby("BARRIOS_1")["SEGMENTO"].value_counts(normalize=True)
                       .rename("proporcion").reset_index())
        orden = (propiedades.groupby("BARRIOS_1")["USD_M2"].median().sort_values().index.tolist())
        top = orden[:8] + orden[-8:]
        figura = px.bar(composicion[composicion["BARRIOS_1"].isin(top)], x="proporcion", y="BARRIOS_1",
                        color="SEGMENTO", orientation="h", color_discrete_map=COLORES_SEGMENTO,
                        category_orders={"BARRIOS_1": top, "SEGMENTO": list(CLASES)},
                        labels={"proporcion": "Proporción", "BARRIOS_1": "", "SEGMENTO": "Segmento"})
        figura.update_layout(height=380, margin=dict(t=10), xaxis_tickformat=".0%")
        st.plotly_chart(figura, width="stretch")
        st.caption("Los 8 barrios más baratos y los 8 más caros (mediana de USD/m²).")

st.divider()
st.caption(f"Ejecución mostrada: `{run_id}` · Datos: BA Data, Departamentos en venta 2019.")
