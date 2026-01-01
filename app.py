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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="MicoRutas Pro: Analyst", layout="wide", page_icon="🍄")

# --- GESTOR DE BD (SQLite Local para simplificar el ejemplo, compatible con Supabase) ---
class DBManager:
    def __init__(self):
        self.conn = sqlite3.connect('micorutas_v2.db', check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        c = self.conn.cursor()
        # Añadimos columnas para análisis avanzado
        c.execute('''
            CREATE TABLE IF NOT EXISTS rutas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                date TEXT,
                distance_km REAL,
                elevation_gain REAL,
                duration_h REAL,
                avg_speed_kmh REAL,
                forest_type TEXT,
                pct_umbria REAL,
                pct_solana REAL,
                points_json TEXT
            )
        ''')
        self.conn.commit()

    def save_route(self, data):
        c = self.conn.cursor()
        points_str = json.dumps(data['points'])
        c.execute('''
            INSERT INTO rutas (filename, date, distance_km, elevation_gain, duration_h, avg_speed_kmh, forest_type, pct_umbria, pct_solana, points_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data['filename'], data['date'], data['distance_km'], data['elevation_gain'], 
              data['duration_h'], data['avg_speed_kmh'], data['forest_type'],
              data['pct_umbria'], data['pct_solana'], points_str))
        self.conn.commit()

    def get_routes(self):
        df = pd.read_sql_query("SELECT * FROM rutas", self.conn)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['points'] = df['points_json'].apply(json.loads)
        return df
    
    def delete_route(self, id):
        c = self.conn.cursor()
        c.execute("DELETE FROM rutas WHERE id=?", (id,))
        self.conn.commit()

db = DBManager()

# --- ALGORITMOS DE ANÁLISIS GPX AVANZADO ---
def calculate_bearing(lat1, lon1, lat2, lon2):
    # Calcula la dirección (azimut) entre dos puntos para saber la orientación de la ladera
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def get_aspect_category(bearing):
    # Simplificación: Norte/Este -> Umbría (Húmedo), Sur/Oeste -> Solana (Seco)
    # N(315-45), E(45-135), S(135-225), W(225-315)
    if 0 <= bearing < 45 or 315 <= bearing <= 360: return "Norte (Umbría)"
    elif 45 <= bearing < 135: return "Este (Umbría)"
    elif 135 <= bearing < 225: return "Sur (Solana)"
    else: return "Oeste (Solana)"

def parse_gpx_advanced(file_buffer, filename, forest_type_input):
    try:
        gpx = gpxpy.parse(file_buffer)
        points_data = []
        
        # Iterar sobre puntos para cálculos vectoriales
        prev_point = None
        umbria_count = 0
        solana_count = 0
        total_segments = 0
        
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    
                    # Datos básicos del punto
                    p_data = {
                        'lat': point.latitude,
                        'lon': point.longitude,
                        'ele': point.elevation,
                        'time': point.time,
                        'speed': 0, # Se calculará
                        'aspect': None
                    }

                    if prev_point:
                        # Distancia 3D
                        dist = point.distance_3d(prev_point)
                        
                        # Velocidad instantánea (m/s -> km/h)
                        time_diff = (point.time - prev_point.time).total_seconds() if point.time and prev_point.time else 0
                        if time_diff > 0:
                            speed_kmh = (dist / 1000) / (time_diff / 3600)
                            p_data['speed'] = speed_kmh
                        
                        # Orientación (Aspecto) - Solo si nos movemos lo suficiente (>10m)
                        if dist > 10:
                            bearing = calculate_bearing(prev_point.latitude, prev_point.longitude, point.latitude, point.longitude)
                            category = get_aspect_category(bearing)
                            p_data['aspect'] = category
                            
                            if "Umbría" in category: umbria_count += 1
                            else: solana_count += 1
                            total_segments += 1
                    
                    # Serializar fecha para JSON
                    if p_data['time']: p_data['time'] = p_data['time'].isoformat()
                    points_data.append(p_data)
                    prev_point = point

        # Estadísticas Globales
        moving_data = gpx.get_moving_data()
        uphill_downhill = gpx.get_uphill_downhill()
        
        pct_umbria = (umbria_count / total_segments * 100) if total_segments > 0 else 0
        pct_solana = (solana_count / total_segments * 100) if total_segments > 0 else 0

        # Fecha inteligente
        date_obj = gpx.time if gpx.time else datetime.now()
        
        return {
            'filename': filename,
            'date': date_obj.strftime('%Y-%m-%d'),
            'distance_km': round(moving_data.moving_distance / 1000, 2),
            'elevation_gain': round(uphill_downhill.uphill, 2),
            'duration_h': round(moving_data.moving_time / 3600, 2),
            'avg_speed_kmh': round(moving_data.max_speed * 3.6, 2) if moving_data.max_speed else 0, # Usamos max speed como ref o average
            'forest_type': forest_type_input,
            'pct_umbria': round(pct_umbria, 1),
            'pct_solana': round(pct_solana, 1),
            'points': points_data # Ahora incluye velocidad y aspecto punto a punto
        }
    except Exception as e:
        st.error(f"Error analizando {filename}: {e}")
        return None

# --- MÓDULO METEO AVANZADO (SMI) ---
@st.cache_data(ttl=3600)
def get_advanced_weather(lat, lon):
    # Pedimos 60 días atrás para construir el modelo de humedad del suelo
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=60)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ["precipitation_sum", "temperature_2m_max", "soil_temperature_0_to_7cm_mean"],
        "timezone": "Europe/Madrid"
    }
    
    try:
        r = requests.get(url, params=params)
        data = r.json()
        
        if 'daily' not in data: return None
        
        df_meteo = pd.DataFrame(data['daily'])
        df_meteo['time'] = pd.to_datetime(df_meteo['time'])
        
        # --- ALGORITMO SMI (Soil Moisture Index Simplificado) ---
        # Simula la acumulación de agua. Cada día el suelo pierde X% (evaporación/drenaje)
        # y gana lo que llueva.
        # Factor de retención: 0.85 (Suelo boscoso retiene bien, pero drena)
        # Factor alto (0.95) = Arcilla (se encharca)
        # Factor bajo (0.6) = Arena (se seca rápido)
        
        retention_factor = 0.90 
        smi_values = []
        current_moisture = 0 # Empezamos asumiendo 0 o un valor base
        
        for rain in df_meteo['precipitation_sum']:
            # Fórmula: HumedadHoy = (HumedadAyer * Retención) + LluviaHoy
            current_moisture = (current_moisture * retention_factor) + rain
            smi_values.append(current_moisture)
            
        df_meteo['SMI'] = smi_values
        return df_meteo
        
    except Exception:
        return None

# --- INTERFAZ ---
st.title("🍄 MicoRutas Pro: Analizador de Biotopos")

with st.sidebar:
    st.header("Importar Datos")
    forest_type = st.selectbox("Tipo de Bosque (para el archivo actual)", 
                               ["Pinar", "Robledal", "Hayedo", "Encinar", "Pradera/Mxto", "Desconocido"])
    uploaded_files = st.file_uploader("Subir GPX", accept_multiple_files=True)
    
    if uploaded_files and st.button("Procesar GPX"):
        bar = st.progress(0)
        for i, f in enumerate(uploaded_files):
            s_io = io.StringIO(f.getvalue().decode("utf-8"))
            # Pasamos el tipo de bosque
            data = parse_gpx_advanced(s_io, f.name, forest_type)
            if data: db.save_route(data)
            bar.progress((i+1)/len(uploaded_files))
        st.success("Procesado con análisis topográfico.")
        st.rerun()

df = db.get_routes()

if not df.empty:
    tab1, tab2, tab3 = st.tabs(["🗺️ Mapa de Búsqueda", "💧 Hidrología y Suelo", "🌲 Topografía y Bosque"])

    # --- TAB 1: MAPA TÁCTICO ---
    with tab1:
        st.markdown("El mapa resalta en **Amarillo/Rojo** las zonas donde anduviste lento (< 1.5 km/h). Esas suelen ser las zonas de recolección.")
        
        col_list, col_map = st.columns([1, 4])
        with col_list:
            selected_id = st.selectbox("Centrar en ruta:", df['id'], format_func=lambda x: df[df['id']==x]['filename'].values[0])
            route_data = df[df['id'] == selected_id].iloc[0]
            
            if st.button("🗑️ Borrar Ruta"):
                db.delete_route(selected_id)
                st.rerun()

        with col_map:
            center_pt = route_data['points'][len(route_data['points'])//2]
            m = folium.Map(location=[center_pt['lat'], center_pt['lon']], zoom_start=14)
            folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri').add_to(m)

            # Dibujar ruta principal
            pts = [(p['lat'], p['lon']) for p in route_data['points']]
            folium.PolyLine(pts, color="#3498db", weight=3, opacity=0.6).add_to(m)
            
            # Dibujar "HOTSPOTS" (Velocidad baja = Recolección)
            slow_points = [p for p in route_data['points'] if p.get('speed', 10) < 1.5 and p.get('speed', 10) > 0.1]
            if slow_points:
                heat_data = [[p['lat'], p['lon']] for p in slow_points]
                # Heatmap específico de "zonas lentas"
                HeatMap(heat_data, radius=10, gradient={0.4: 'yellow', 1: 'red'}, name="Zonas de Recolección").add_to(m)

            st_folium(m, height=500, width="100%")

    # --- TAB 2: HIDROLOGÍA AVANZADA ---
    with tab2:
        st.subheader("Análisis de Humedad Real del Suelo (SMI)")
        st.info("Este gráfico no muestra solo lluvia, muestra **cuánta agua retiene el suelo**. Si la curva azul sube, el micelio se activa.")
        
        # Usamos coordenadas de la ruta seleccionada
        if st.button("Analizar Hidrología (Últimos 60 días)"):
            with st.spinner("Calculando retención de agua y temperaturas..."):
                meteo_df = get_advanced_weather(center_pt['lat'], center_pt['lon'])
                
                if meteo_df is not None:
                    # Crear gráfico de doble eje
                    fig = go.Figure()

                    # 1. Barras de Lluvia (Eje Y derecho, invertido para que parezca que cae del cielo)
                    fig.add_trace(go.Bar(
                        x=meteo_df['time'], 
                        y=meteo_df['precipitation_sum'], 
                        name="Lluvia Diaria (mm)",
                        marker_color='lightblue',
                        opacity=0.4,
                        yaxis='y2'
                    ))

                    # 2. Curva SMI (Humedad Acumulada) - LA CLAVE
                    fig.add_trace(go.Scatter(
                        x=meteo_df['time'], 
                        y=meteo_df['SMI'], 
                        name="Humedad en Suelo (SMI)",
                        line=dict(color='blue', width=4),
                        fill='tozeroy',
                        fillcolor='rgba(0, 0, 255, 0.1)'
                    ))

                    # 3. Temperatura del suelo
                    fig.add_trace(go.Scatter(
                        x=meteo_df['time'], 
                        y=meteo_df['soil_temperature_0_to_7cm_mean'], 
                        name="Temp. Suelo",
                        line=dict(color='orange', width=2, dash='dot')
                    ))

                    # Configurar ejes
                    fig.update_layout(
                        title="Evolución del Biotopo: Lluvia vs Retención",
                        yaxis=dict(title="Índice SMI / Temp (°C)"),
                        yaxis2=dict(title="Lluvia (mm)", overlaying='y', side='right', range=[0, 100]),
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretación automática
                    last_smi = meteo_df['SMI'].iloc[-1]
                    smi_trend = meteo_df['SMI'].iloc[-1] - meteo_df['SMI'].iloc[-5] # Tendencia 5 días
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Índice Humedad Actual", f"{last_smi:.1f}", delta=f"{smi_trend:.1f} ult. 5 días")
                    
                    interpretation = ""
                    if last_smi < 10: interpretation = "🌵 Suelo SECO. Probabilidad baja salvo zonas de ribera."
                    elif 10 <= last_smi < 30: interpretation = "⚠️ Recuperando humedad. Faltan unos días."
                    elif 30 <= last_smi < 60: interpretation = "🍄 Condiciones IDEALES de humedad."
                    else: interpretation = "💧 Suelo saturado/encharcado."
                    
                    c2.info(interpretation)

    # --- TAB 3: TOPOGRAFÍA Y BOSQUE ---
    with tab3:
        st.subheader("Caracterización del Terreno")
        
        c1, c2 = st.columns(2)
        
        with c1:
            # Gráfico de sectores: Solana vs Umbría
            # Agrupar datos de todas las rutas o la seleccionada
            fig_sun = px.pie(values=[route_data['pct_umbria'], route_data['pct_solana']], 
                             names=["Umbría (Norte/Este)", "Solana (Sur/Oeste)"],
                             title="Exposición Solar de la Ruta",
                             color_discrete_sequence=['#2ecc71', '#f1c40f'])
            st.plotly_chart(fig_sun, use_container_width=True)
            
        with c2:
            st.metric("Tipo de Bosque", route_data['forest_type'])
            st.metric("Desnivel", f"{route_data['elevation_gain']} m")
            
            # Histograma de Altitud (Donde has estado más tiempo)
            elevations = [p['ele'] for p in route_data['points'] if p['ele']]
            if elevations:
                fig_ele = px.histogram(x=elevations, nbins=20, 
                                       title="Distribución de Altitud (Cotas frecuentadas)",
                                       labels={'x':'Altitud (m)'}, color_discrete_sequence=['#8e44ad'])
                st.plotly_chart(fig_ele, use_container_width=True)

else:
    st.info("Sube un GPX y selecciona el tipo de bosque para empezar.")
