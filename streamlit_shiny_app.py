import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(page_title="Dashboard Estudiantil", layout="wide")

# === CARGAR DATOS ===
@st.cache_data
def load_data():
    return pd.read_csv("dummy_estudiantes_dataset.csv", encoding="latin1")

df = load_data()

# === DEFINICIONES DE VARIABLES ===
demograficas = ["Procedencia", "1st Fall Enrollment", "Índice General", "Índice Científico", "PCAT"]
excluir_cat = ["Nombre", "Numero de Estudiante", "Email UPR", "Número de Expediente"]
notas_cursos = [col for col in df.columns if "Nota" in col or "(D)" in col or "(F)" in col or "(W)" in col]
continuas = ["Índice General", "Índice Científico", "PCAT"]
categoricas = [col for col in df.select_dtypes(include=["object", "category"]).columns if col not in excluir_cat and col not in continuas]
nota_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
df[notas_cursos] = df[notas_cursos].apply(lambda col: col.map(lambda x: nota_map.get(str(x).strip().upper(), np.nan)))

# === VALORES POR DEFECTO ===
default_cat = "1st Fall Enrollment"
default_val = "All Enrollment"
default_proc = "Todas"
default_x = "Índice General"
default_y = "Índice Científico"

# === SIDEBAR: CONTROLES ===
with st.sidebar:
    st.header("📊 Filtros")

    if st.button("🔄 Resetear filtros"):
        st.session_state.clear()

    col_cat = st.selectbox("Filtrar por categoría", options=[default_cat] + sorted([c for c in categoricas if c != default_cat]), key="col_cat")
    valores = sorted(df[col_cat].dropna().astype(str).unique())
    if col_cat == default_cat:
        valores = [default_val] + valores
    val_cat = st.selectbox(f"Valor en '{col_cat}'", valores, key="val_cat")

    val_proc = st.selectbox("Procedencia", options=["Todas"] + sorted(df["Procedencia"].dropna().astype(str).unique()), key="val_proc")

    col_x = st.selectbox("Variable continua (eje X)", options=continuas, index=0, key="col_x")
    col_y = st.selectbox("Variable continua (eje Y)", options=[c for c in continuas if c != col_x], key="col_y")

    slider_min = float(df[col_x].min())
    slider_max = float(df[col_x].max())
    slider_step = 1.0 if col_x == "PCAT" else 0.1
    selected_range = st.slider(f"Rango de '{col_x}'", min_value=slider_min, max_value=slider_max, value=(slider_min, slider_max), step=slider_step, key="slider")

# === FILTRADO DE DATOS ===
df_filtrado = df.copy()
if col_cat == default_cat and val_cat != default_val:
    df_filtrado = df_filtrado[df_filtrado[col_cat].astype(str) == val_cat]
elif col_cat != default_cat:
    df_filtrado = df_filtrado[df_filtrado[col_cat].astype(str) == val_cat]
if val_proc != "Todas":
    df_filtrado = df_filtrado[df_filtrado["Procedencia"].astype(str) == val_proc]
df_filtrado = df_filtrado[(df_filtrado[col_x] >= selected_range[0]) & (df_filtrado[col_x] <= selected_range[1])]

# === GRAFICO DE DISPERSIÓN ===
st.subheader("📈 Gráfico de Dispersión")
fig = px.scatter(df_filtrado, x=col_x, y=col_y, color="Procedencia", title=f"{col_y} vs {col_x}")
st.plotly_chart(fig, use_container_width=True)

# === CHAT CON LOS DATOS ===
st.header("🤖 Chat con tus datos")

st.markdown("Haz preguntas en lenguaje natural sobre los datos. Ejemplos:")
st.markdown("""
- ¿Cuál es el promedio del Índice General por Procedencia?
- ¿Cuántos estudiantes hay por cada año de ingreso?
- Haz un histograma del PCAT.
- ¿Cuál es la nota más común en Biología?
""")

# Inicializar LLM
llm = OpenAI(api_token=st.secrets["OPENAI_API_KEY"])
sdf = SmartDataframe(df_filtrado, config={"llm": llm})

# Entrada del usuario
user_input = st.text_input("Tu pregunta")

if user_input:
    with st.spinner("Pensando..."):
        try:
            response = sdf.chat(user_input)
            st.write(response)
        except Exception as e:
            st.error(f"Ocurrió un error al responder: {e}")
