import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
demograficas = [
    "Procedencia", "1st Fall Enrollment", "Índice General", "Índice Científico", "PCAT"
]
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

# === FILTRADO ===
df_filtrado = df.copy()
if col_cat == default_cat and val_cat != default_val:
    df_filtrado = df_filtrado[df_filtrado[col_cat].astype(str) == val_cat]
elif col_cat != default_cat:
    df_filtrado = df_filtrado[df_filtrado[col_cat].astype(str) == val_cat]
if val_proc != "Todas":
    df_filtrado = df_filtrado[df_filtrado["Procedencia"].astype(str) == val_proc]
df_filtrado = df_filtrado[(df_filtrado[col_x] >= selected_range[0]) & (df_filtrado[col_x] <= selected_range[1])]

# === MÉTRICAS ===
st.title("📈 Dashboard Estudiantil")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total registros", f"{len(df_filtrado):,}")
col2.metric("Promedio General", f"{df_filtrado['Índice General'].mean():.2f}")
col3.metric("Promedio Científico", f"{df_filtrado['Índice Científico'].mean():.2f}")
col4.metric("Promedio PCAT", f"{df_filtrado['PCAT'].mean():.2f}")

# === GRÁFICO: HISTOGRAMA ===
hist = go.Figure()
hist.add_trace(go.Histogram(x=df_filtrado[col_x], nbinsx=10, marker_color="#1f77b4"))
hist.update_layout(title=f"Distribución de {col_x}", xaxis_title=col_x, yaxis_title="Frecuencia")

# === GRÁFICO: BARRAS CATEGÓRICAS ===
valores_barras = df_filtrado[col_cat].dropna().astype(str).value_counts().sort_index()
bars = go.Figure()
bars.add_trace(go.Bar(x=valores_barras.index, y=valores_barras.values, marker_color="#2c3e50"))
bars.update_layout(title=f"Distribución de {col_cat}", xaxis_title=col_cat, yaxis_title="Cantidad", xaxis_type='category')

# === MATRIZ DE CORRELACIÓN ===
columnas_cor = notas_cursos + continuas
datos_cor = df_filtrado[columnas_cor].copy()
matriz = datos_cor.corr()

heatmap = go.Figure(data=go.Heatmap(
    z=matriz.values,
    x=matriz.columns,
    y=matriz.index,
    colorscale="Blues",
    zmin=-1,
    zmax=1,
    colorbar=dict(title="Correlación")
))
heatmap.update_layout(
    title="Correlación entre notas y métricas",
    xaxis=dict(tickangle=45, tickfont=dict(size=10), automargin=True),
    yaxis=dict(tickfont=dict(size=10), automargin=True),
    width=1200,
    height=1000,
    margin=dict(t=80, l=200, r=50, b=200)
)

# === SCATTER + REGRESIÓN ===
x_vals = df_filtrado[col_x].dropna().values.reshape(-1, 1)
y_vals = df_filtrado[col_y].dropna().values.reshape(-1, 1)
valid_idx = (~np.isnan(x_vals.flatten())) & (~np.isnan(y_vals.flatten()))
x_clean = x_vals[valid_idx].reshape(-1, 1)
y_clean = y_vals[valid_idx].reshape(-1, 1)

model = LinearRegression()
model.fit(x_clean, y_clean)
y_pred = model.predict(x_clean)
r2 = r2_score(y_clean, y_pred)
slope = model.coef_[0][0]
intercept = model.intercept_[0]
equation = f"y = {slope:.2f}x + {intercept:.2f}<br>R² = {r2:.3f}"

scatter = go.Figure()
scatter.add_trace(go.Scatter(x=x_clean.flatten(), y=y_clean.flatten(), mode='markers', name='Datos'))
scatter.add_trace(go.Scatter(x=x_clean.flatten(), y=y_pred.flatten(), mode='lines', name='Regresión', line=dict(color='orange')))
scatter.update_layout(title=f"{col_x} vs {col_y} con regresión<br><sub>{equation}</sub>", xaxis_title=col_x, yaxis_title=col_y)

# === VISUALIZACIÓN DE PLOTS ===
g1, g2 = st.columns(2)
g1.plotly_chart(hist, use_container_width=True)
g2.plotly_chart(bars, use_container_width=True)

g3, g4 = st.columns(2)
g3.plotly_chart(scatter, use_container_width=True)
g4.plotly_chart(heatmap, use_container_width=True)

# === SECCIÓN DE TABLA DE DATOS — ELIMINADA ===
# (Removida según solicitud)
