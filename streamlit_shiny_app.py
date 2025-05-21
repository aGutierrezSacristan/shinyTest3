import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import requests 
from io import BytesIO
import io
import contextlib

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(page_title="Dashboard Estudiantil", layout="wide")

# === CARGAR DATOS ===
@st.cache_data
def load_data_from_gdrive(file_id: str) -> pd.DataFrame:
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    r = requests.get(url)
    return pd.read_csv(BytesIO(r.content), encoding="latin1")

FILE_ID = st.secrets["FILE_ID"]
df = load_data_from_gdrive(FILE_ID)

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

# === CHAT INTELIGENTE USANDO OPENAI ===
st.header("🤖 Chat con tus datos (compatible con Streamlit Cloud)")

st.markdown("Haz preguntas en lenguaje natural sobre los datos. Ejemplos:")
st.markdown("""
- ¿Cuál es el promedio del Índice General por Procedencia?
- ¿Cuántos estudiantes hay por cada año de ingreso?
- ¿Cuál es la nota más común en Biología?
- Haz un resumen estadístico del PCAT.
""")

# Configurar la clave OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Entrada de usuario
user_question = st.text_input("Tu pregunta")

# Función que genera código con OpenAI
def generate_code_from_question(question, df_sample):
    prompt = f"""
Actúa como un asistente de análisis de datos. Se te da un DataFrame llamado df con las siguientes columnas:

{', '.join(df_sample.columns)}

El usuario preguntó: "{question}"

Devuelve solo el código Python necesario para responder esa pregunta. Usa pandas y guarda el resultado en una variable llamada 'resultado'.
Si corresponde, usa plotly express para gráficas. No devuelvas explicaciones.
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message["content"]

if user_question:
    try:
        code = generate_code_from_question(user_question, df_filtrado)
        st.code(code, language="python")

        # Ejecutar el código
        local_vars = {"df": df_filtrado.copy(), "px": px, "pd": pd, "np": np}
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(code, {}, local_vars)

        resultado = local_vars.get("resultado", None)

        if resultado is not None:
            if hasattr(resultado, "to_plotly_json") or "plotly" in str(type(resultado)).lower():
                st.plotly_chart(resultado, use_container_width=True)
            else:
                st.write(resultado)
        else:
            st.warning("No se generó ninguna variable llamada 'resultado'.")
    except Exception as e:
        st.error(f"Error al procesar tu pregunta: {e}")

