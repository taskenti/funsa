import streamlit as st
import gpxpy
import pandas as pd
import folium
from folium.plugins import HeatMap, Fullscreen
from streamlit_folium import st_folium
import sqlite3
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="MicoData: Dashboard", layout="wide", page_icon="🍄")

# --- CSS PARA MAXIMIZAR ESPACIO ---
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        h1 {margin-bottom: 0rem;}
    </style>
""", unsafe_allow_html=True)

# --- GESTOR DE BD OPTIMIZADO PARA BIG DATA ---
class DBManager:
    def __init__(self):
        self.conn = sqlite3.connect('micodata_global.db', check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        c = self.conn.cursor()
        # Tabla optimizada: Guardamos 'points_json' pero también metadatos clave para filtrado rápido
        c.execute('''
            CREATE TABLE IF NOT EXISTS rutas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                date TEXT,
                min_lat REAL, max_lat REAL, min_lon REAL, max_lon REAL,
                points_json TEXT
            )
        ''')
        self.conn.commit()

    def save_route(self, data):
        c = self.conn.cursor()
        # Calcular Bounding Box de la ruta para búsquedas rápidas futuras
        lats = [p['lat'] for p in data['points']]
        lons = [p['lon'] for p in data['points']]
        
        c.execute('''
            INSERT INTO rutas (filename, date, min_lat, max_lat, min_lon, max_lon, points_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (data['filename'], data['date'], min(lats), max(lats), min(lons), max(lons), json.dumps(data['points'])))
        self.conn.commit()

    # Función clave: Cargar TODOS los puntos en memoria (con caché)
    @st.cache_data(ttl=600) 
    def get_all_points_dataframe(_self):
        # El guion bajo en _self es para que streamlit ignore el objeto DB en el hash de caché
        df_routes = pd.read_sql_query("SELECT * FROM rutas", _self.conn)
        
        all_points = []
        for _, row in df_routes.iterrows():
            points = json.loads(row['points_json'])
            # Añadimos metadatos de la ruta a cada punto (para filtrar por fecha/año después)
            route_date = pd.to_datetime(row['date'])
            for p in points:
                # Downsampling: Si tienes miles de rutas, coge 1 de cada 5 puntos para ir rápido
                # Si quieres precisión total, quita el 'if'
                p['route_date'] = route_date
                p['year'] = route_date.year
                p['month'] = route_date.month
                all_points.append(p)
        
        return pd.DataFrame(all_points)

db = DBManager()

# --- PARSER GPX ---
def parse_gpx(file_buffer, filename):
    try:
        gpx = gpxpy.parse(file_buffer)
        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                # Simplificación ligera para no explotar la BD
                for i, point in enumerate(segment.points):
                    if i % 3 == 0: # Guardar 1 de cada 3 puntos
                        points.append({'lat': point.latitude, 'lon': point.longitude, 'ele': point.elevation, 'speed': 0}) # Speed se calcula luego si hace falta
        if points:
            date_obj = gpx.time if gpx.time else datetime.now()
            return {'filename': filename, 'date': date_obj.strftime('%Y-%m-%d'), 'points': points}
    except: return None

# --- METEO ---
@st.cache_data(ttl=3600)
def get_climate_data(lat, lon):
    end = datetime.now().date()
    start = end - timedelta(days=45) # 45 días de histórico
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ["precipitation_sum", "soil_temperature_0_to_7cm_mean"],
        "timezone": "auto"
    }
    try:
        r = requests.get(url, params=params).json()
        return pd.DataFrame(r['daily'])
    except: return None

# --- UI PRINCIPAL ---

# 1. BARRA LATERAL (Solo para cargar datos)
with st.sidebar:
    st.title("📂 Cargar GPX")
    uploaded_files = st.file_uploader("Arrastra tus archivos aquí", accept_multiple_files=True)
    if uploaded_files and st.button("Procesar"):
        bar = st.progress(0)
        for i, f in enumerate(uploaded_files):
            s_io = io.StringIO(f.getvalue().decode("utf-8"))
            data = parse_gpx(s_io, f.name)
            if data: db.save_route(data)
            bar.progress((i+1)/len(uploaded_files))
        st.success("Procesado.")
        st.cache_data.clear() # Limpiar caché para recargar datos nuevos
        st.rerun()

# 2. CARGA DE DATOS MASIVA
df_all = db.get_all_points_dataframe()

if df_all.empty:
    st.warning("No hay rutas. Sube tus GPX en el menú lateral.")
    st.stop()

# --- LAYOUT DASHBOARD ---

# Fila 1: El Mapa y los Controles de Análisis
col_map, col_kpi = st.columns([3, 1])

with col_kpi:
    st.subheader("🔭 Análisis Local")
    st.info("Mueve el mapa a la zona que quieras investigar (ej: Soria, Pirineos) y pulsa el botón.")
    
    # Estado de la sesión para guardar los límites del mapa
    if 'map_bounds' not in st.session_state:
        st.session_state.map_bounds = None
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [40.416, -3.703] # Madrid default

    analyze_btn = st.button("📍 ANALIZAR ZONA VISIBLE", type="primary", use_container_width=True)
    
    # Filtros visuales (no afectan a los datos, solo al mapa)
    heatmap_intensity = st.slider("Intensidad Heatmap", 0.1, 1.0, 0.6)
    
    # KPIs Globales (De todo el histórico)
    st.divider()
    st.metric("Total Puntos Trackeados", f"{len(df_all):,}")
    st.metric("Años de registros", f"{df_all['year'].min()} - {df_all['year'].max()}")

with col_map:
    # Lógica del mapa
    m = folium.Map(location=st.session_state.map_center, zoom_start=6, tiles=None)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name="Satélite").add_to(m)
    folium.TileLayer('OpenStreetMap', name="Callejero").add_to(m)
    
    # Capa 1: HEATMAP GLOBAL (Lo que pedías: ver todo junto)
    # Convertimos lat/lon a lista para el plugin
    heat_data = df_all[['lat', 'lon']].values.tolist()
    HeatMap(heat_data, radius=12, blur=15, min_opacity=heatmap_intensity, 
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)

    # Control de capas
    folium.LayerControl().add_to(m)
    
    # Capturamos interacción
    map_data = st_folium(m, height=500, width="100%", key="main_map")

# --- LÓGICA DE FILTRADO GEOGRÁFICO ---
filtered_df = df_all # Por defecto todo
climate_df = None
location_msg = "Mostrando datos globales"

if analyze_btn and map_data and map_data['bounds']:
    # 1. Obtener límites de la pantalla
    bounds = map_data['bounds']
    sw = bounds['_southWest']
    ne = bounds['_northEast']
    center = map_data['center']
    
    st.session_state.map_center = [center['lat'], center['lng']] # Guardar centro para no resetear mapa
    
    # 2. Filtrar DataFrame "Big Data" por coordenadas
    mask = (
        (df_all['lat'] >= sw['lat']) & (df_all['lat'] <= ne['lat']) &
        (df_all['lon'] >= sw['lng']) & (df_all['lon'] <= ne['lng'])
    )
    filtered_df = df_all.loc[mask]
    
    # 3. Obtener Clima del CENTRO del mapa
    with st.spinner(f"Analizando clima en {center['lat']:.2f}, {center['lng']:.2f}..."):
        climate_df = get_climate_data(center['lat'], center['lng'])
    
    location_msg = f"📍 Análisis de zona visible ({len(filtered_df)} puntos encontrados)"
else:
    # Si no se ha filtrado, coger clima de un punto promedio global
    pass 

# --- PANELES DE RESULTADOS (ABAJO DEL MAPA) ---

st.subheader(location_msg)

if filtered_df.empty:
    st.warning("No hay rutas en la zona del mapa que estás viendo. Haz zoom out o muévete a una zona con tracks.")
else:
    # Creamos 3 columnas de análisis Pro
    c_climate, c_topo, c_pattern = st.columns([1.5, 1, 1])

    # PANEL 1: CLIMATOLOGÍA EN TIEMPO REAL (DE LA ZONA)
    with c_climate:
        st.markdown("#### 🌦️ Clima (Últimos 45 días)")
        if climate_df is not None:
            climate_df['time'] = pd.to_datetime(climate_df['time'])
            
            # Cálculo SMI simple (Soil Moisture Index)
            smi = []
            curr = 0
            for rain in climate_df['precipitation_sum']:
                curr = (curr * 0.9) + rain
                smi.append(curr)
            climate_df['SMI'] = smi
            
            # Gráfico combinado
            fig = go.Figure()
            fig.add_trace(go.Bar(x=climate_df['time'], y=climate_df['precipitation_sum'], name="Lluvia (mm)", marker_color="skyblue"))
            fig.add_trace(go.Scatter(x=climate_df['time'], y=climate_df['SMI'], name="Humedad Suelo", line=dict(color="blue", width=3), yaxis="y2"))
            fig.add_trace(go.Scatter(x=climate_df['time'], y=climate_df['soil_temperature_0_to_7cm_mean'], name="Temp Suelo", line=dict(color="orange", dash="dot"), yaxis="y3"))
            
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=30, b=0),
                yaxis=dict(title="Lluvia", domain=[0, 0.6]),
                yaxis2=dict(title="Humedad", overlaying="y", side="right"),
                yaxis3=dict(title="Temp", overlaying="y", side="left", position=0.05, domain=[0.65, 1]),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Semáforo rápido
            last_rain_15 = climate_df['precipitation_sum'].tail(15).sum()
            last_smi = smi[-1]
            st.caption(f"Lluvia acumulada (15d): **{last_rain_15:.1f}mm** | Índice Humedad: **{last_smi:.1f}**")
        else:
            st.info("Pulsa 'Analizar Zona Visible' para cargar el clima de esta región.")

    # PANEL 2: TOPOGRAFÍA DE LA ZONA VISIBLE
    with c_topo:
        st.markdown("#### ⛰️ Cota y Topografía")
        # Histograma de Altitud de los puntos VISIBLES
        fig_ele = px.histogram(filtered_df, x="ele", nbins=20, 
                               title="Distribución de Altitud (m)",
                               color_discrete_sequence=['#8e44ad'])
        fig_ele.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig_ele, use_container_width=True)
        
        # Estadística simple
        mean_ele = filtered_df['ele'].mean()
        st.metric("Cota Media en esta zona", f"{mean_ele:.0f} m")

    # PANEL 3: PATRONES TEMPORALES (¿CUÁNDO VAS AQUÍ?)
    with c_pattern:
        st.markdown("#### 📅 ¿Cuándo funciona esta zona?")
        # Extraer meses de los puntos filtrados
        month_counts = filtered_df['month'].value_counts().sort_index()
        
        # Mapear números a nombres
        month_names = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
        df_months = pd.DataFrame({
            'Mes': [month_names.get(x, x) for x in month_counts.index],
            'Actividad': month_counts.values
        })
        
        fig_time = px.bar(df_months, x='Mes', y='Actividad', 
                          title="Tus visitas históricas",
                          color='Actividad', color_continuous_scale='Viridis')
        fig_time.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Predicción simple
        best_month = df_months.loc[df_months['Actividad'].idxmax()]['Mes']
        st.info(f"Históricamente, esta zona es de: **{best_month}**")
