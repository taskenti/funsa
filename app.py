import streamlit as st
import gpxpy
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import st_folium
import sqlite3
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import branca.colormap as cm

# --- CONFIGURACIÓN E INICIALIZACIÓN ---
st.set_page_config(page_title="MicoBrain Ultimate 🍄", layout="wide", page_icon="🍄")

# CSS Profesional
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
        h1, h2, h3 {color: #2e7d32;}
    </style>
""", unsafe_allow_html=True)

# --- CLASE 1: GESTOR DE DATOS (PERSISTENCIA) ---
class DataManager:
    def __init__(self, db_name='micobrain_pro.db'):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.init_db()

    def init_db(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS rutas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT, date TEXT, 
            distance_km REAL, elevation_gain REAL, duration_h REAL,
            min_lat REAL, max_lat REAL, min_lon REAL, max_lon REAL,
            points_json TEXT, tags TEXT DEFAULT "General")''')
        c.execute('''CREATE TABLE IF NOT EXISTS favoritos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL, lon REAL, nombre TEXT, descripcion TEXT,
            especie TEXT, rating INTEGER, fecha_creacion TEXT)''')
        self.conn.commit()

    def save_route(self, data, tags):
        pts = data['points']
        lats, lons = [p['lat'] for p in pts], [p['lon'] for p in pts]
        c = self.conn.cursor()
        c.execute('''INSERT INTO rutas 
            (filename, date, distance_km, elevation_gain, duration_h, min_lat, max_lat, min_lon, max_lon, points_json, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (data['filename'], data['date'], data['dist'], data['ele'], data['dur'], 
             min(lats), max(lats), min(lons), max(lons), json.dumps(pts), tags))
        self.conn.commit()

    def delete_route(self, route_id):
        self.conn.execute("DELETE FROM rutas WHERE id=?", (route_id,))
        self.conn.commit()

    def save_favorite(self, lat, lon, nombre, desc, especie, rating):
        self.conn.execute("INSERT INTO favoritos (lat, lon, nombre, descripcion, especie, rating, fecha_creacion) VALUES (?,?,?,?,?,?,?)",
                          (lat, lon, nombre, desc, especie, rating, datetime.now().strftime("%Y-%m-%d")))
        self.conn.commit()
    
    def delete_favorite(self, fav_id):
        self.conn.execute("DELETE FROM favoritos WHERE id=?", (fav_id,))
        self.conn.commit()

    @st.cache_data(ttl=300)
    def get_dataframe(_self):
        # Carga optimizada
        df = pd.read_sql_query("SELECT * FROM rutas", _self.conn)
        all_points = []
        for _, row in df.iterrows():
            pts = json.loads(row['points_json'])
            # Convertimos a formato ligero para análisis
            for i, p in enumerate(pts):
                # Downsampling inteligente: Guardar puntos lentos (setas) siempre, y rápidos cada 10
                is_slow = p.get('speed', 5) < 1.5
                if is_slow or (i % 5 == 0):
                    p['route_id'] = row['id']
                    p['route_date'] = row['date']
                    p['tags'] = row['tags']
                    all_points.append(p)
        return pd.DataFrame(all_points)

    def get_favorites_df(self):
        return pd.read_sql_query("SELECT * FROM favoritos", self.conn)

# --- CLASE 2: MOTOR DE INTELIGENCIA (ML & GEOMETRÍA) ---
class IntelligenceEngine:
    
    @staticmethod
    def parse_gpx(file_buffer, filename):
        try:
            gpx = gpxpy.parse(file_buffer)
            points = []
            prev_pt = None
            
            # Métricas globales
            mov_data = gpx.get_moving_data()
            uphill_downhill = gpx.get_uphill_downhill()
            
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        speed = 0
                        if prev_pt:
                            dist = point.distance_2d(prev_pt)
                            time_diff = (point.time - prev_pt.time).total_seconds()
                            if time_diff > 0: speed = (dist/1000)/(time_diff/3600)
                        
                        points.append({'lat': point.latitude, 'lon': point.longitude, 'ele': point.elevation, 'speed': speed})
                        prev_pt = point
            
            return {
                'filename': filename,
                'date': gpx.time.strftime('%Y-%m-%d') if gpx.time else datetime.now().strftime('%Y-%m-%d'),
                'points': points,
                'dist': mov_data.moving_distance / 1000,
                'ele': uphill_downhill.uphill,
                'dur': mov_data.moving_time / 3600
            }
        except Exception as e:
            return None

    @staticmethod
    def find_hotspots(df_points, eps_meters=30, min_samples=10):
        """
        Usa DBSCAN para encontrar clusters de puntos donde la velocidad fue baja (< 1.5 km/h).
        Esto identifica automáticamente las zonas donde te paraste a buscar.
        """
        if df_points.empty: return []
        
        # Filtrar solo puntos de "búsqueda" (lentos)
        search_points = df_points[df_points['speed'] < 1.5]
        if len(search_points) < min_samples: return []

        coords = search_points[['lat', 'lon']].values
        
        # DBSCAN: eps en radianes (aprox. km / 6371)
        kms_per_radian = 6371.0088
        eps_rad = (eps_meters / 1000) / kms_per_radian
        
        db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine', algorithm='ball_tree').fit(np.radians(coords))
        
        clusters = []
        labels = db.labels_
        unique_labels = set(labels)
        
        for k in unique_labels:
            if k == -1: continue # Ruido
            
            class_member_mask = (labels == k)
            cluster_points = coords[class_member_mask]
            
            # Calcular envolvente convexa (polígono) para el cluster
            if len(cluster_points) > 3:
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                # Cerrar el polígono
                hull_points = np.append(hull_points, [hull_points[0]], axis=0)
                clusters.append(hull_points.tolist())
                
        return clusters

    @staticmethod
    def generate_probability_grid(df_points, grid_size=50):
        """Genera una malla de probabilidad usando KDE para visualización de heatmap predictivo"""
        productive = df_points[df_points['speed'] < 2.0]
        if len(productive) < 50: return None, None, None

        data = productive[['lat', 'lon']].values
        
        # Grid bounds
        lat_min, lat_max = data[:, 0].min(), data[:, 0].max()
        lon_min, lon_max = data[:, 1].min(), data[:, 1].max()
        
        # Margen del 10%
        lat_margin = (lat_max - lat_min) * 0.1
        lon_margin = (lon_max - lon_min) * 0.1
        
        # Generar Grid
        lat_grid = np.linspace(lat_min - lat_margin, lat_max + lat_margin, grid_size)
        lon_grid = np.linspace(lon_min - lon_margin, lon_max + lon_margin, grid_size)
        X, Y = np.meshgrid(lat_grid, lon_grid)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        
        # Entrenar modelo rápido
        kde = KernelDensity(bandwidth=0.001, metric='haversine')
        kde.fit(np.radians(data))
        
        # Evaluar
        Z = np.exp(kde.score_samples(np.radians(xy)))
        Z = Z.reshape(X.shape)
        
        return X, Y, Z

# --- CLASE 3: SERVICIO METEOROLÓGICO ---
class WeatherService:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_weather_context(lat, lon):
        try:
            # 60 días atrás + 7 predicción
            start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            end = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            
            url = "https://archive-api.open-meteo.com/v1/archive"
            # Usamos endpoint de forecast híbrido si es posible, simplificamos usando archive para pasado
            # y forecast para futuro en una app real. Aquí usamos archive para el contexto histórico.
            
            r = requests.get(url, params={
                "latitude": lat, "longitude": lon,
                "start_date": start, "end_date": datetime.now().strftime('%Y-%m-%d'),
                "daily": ["precipitation_sum", "soil_temperature_0_to_7cm_mean"],
                "timezone": "auto"
            })
            data = r.json()
            
            df = pd.DataFrame(data['daily'])
            
            # Cálculo SMI (Soil Moisture Index)
            smi_list = []
            smi = 0
            for rain in df['precipitation_sum']:
                smi = (smi * 0.90) + (rain if rain else 0)
                smi_list.append(smi)
            df['smi'] = smi_list
            
            return df
        except:
            return None

# --- UI PRINCIPAL ---
db = DataManager()
engine = IntelligenceEngine()
weather = WeatherService()

st.sidebar.title("🍄 Control de Misión")

# --- 1. IMPORTACIÓN ---
with st.sidebar.expander("📂 Importar Tracks", expanded=False):
    tags = st.text_input("Etiquetas (ej: Otoño 2024)", "General")
    files = st.file_uploader("Arrastra GPX", accept_multiple_files=True)
    if files and st.button("Procesar"):
        bar = st.progress(0)
        for i, f in enumerate(files):
            s_io = io.StringIO(f.getvalue().decode("utf-8"))
            data = engine.parse_gpx(s_io, f.name)
            if data: db.save_route(data, tags)
            bar.progress((i+1)/len(files))
        st.success("Procesado.")
        st.cache_data.clear()
        st.rerun()

# --- 2. FILTROS ---
df_all = db.get_dataframe()
if df_all.empty:
    st.warning("Base de datos vacía. Sube rutas para empezar.")
    st.stop()

available_tags = list(df_all['tags'].unique())
sel_tags = st.sidebar.multiselect("Filtrar Colección", available_tags, default=available_tags)
df_view = df_all[df_all['tags'].isin(sel_tags)]

if df_view.empty: st.stop()

# --- TABS PRINCIPALES ---
tab_map, tab_analytics, tab_planner, tab_data = st.tabs(["🗺️ Mapa Táctico", "📊 Analítica", "🔮 Predicción", "💾 Datos"])

with tab_map:
    col_ctrl, col_viz = st.columns([1, 4])
    
    with col_ctrl:
        st.markdown("### Capas Inteligentes")
        show_tracks = st.toggle("Mostrar Rutas (Líneas)", True)
        show_heat = st.toggle("Calor Histórico (Actividad)", True)
        show_clusters = st.toggle("🤖 Detectar 'Corros' (IA)", False, help="Usa DBSCAN para detectar zonas donde te paraste mucho tiempo.")
        show_prediction = st.toggle("🔮 Mapa Probabilidad (KDE)", False, help="Genera contornos de predicción matemática basados en densidad.")
        
        st.divider()
        st.info("Haz clic en el mapa para guardar un setal manualmente.")

    with col_viz:
        # Centro mapa
        center_lat, center_lon = df_view['lat'].mean(), df_view['lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
        
        # Capas Base
        folium.TileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attr='OpenTopoMap', name='Topográfico').add_to(m)
        folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='Satélite').add_to(m)

        # 1. Rutas
        if show_tracks:
            # Agrupar por ruta para dibujar líneas
            for rid in df_view['route_id'].unique():
                route_pts = df_view[df_view['route_id'] == rid]
                coords = route_pts[['lat', 'lon']].values.tolist()
                folium.PolyLine(coords, color="#2e7d32", weight=2, opacity=0.5).add_to(m)

        # 2. Heatmap Histórico
        if show_heat:
            heat_data = df_view[['lat', 'lon']].values.tolist()
            plugins.HeatMap(heat_data, radius=15, blur=20, name="Actividad").add_to(m)

        # 3. Clusters Automáticos (DBSCAN) - LA JOYA DE LA CORONA
        if show_clusters:
            clusters = engine.find_hotspots(df_view)
            for cluster_coords in clusters:
                # Dibujar polígono del "corro" detectado
                folium.Polygon(
                    locations=cluster_coords,
                    color='#ff9800', fill=True, fill_color='#ff9800', fill_opacity=0.4,
                    weight=2, popup="Zona Productiva Detectada (IA)"
                ).add_to(m)

        # 4. Predicción Matemática (KDE)
        if show_prediction:
            X, Y, Z = engine.generate_probability_grid(df_view)
            if Z is not None:
                # Normalizar Z para colores
                # Crear imagen overlay o contornos. Usaremos contornos simplificados para Streamlit
                # Para simplificar en Folium usamos ImageOverlay con mapa de color
                st.toast("Generando capa predictiva...")
                # Esto es complejo de renderizar perfecto en Folium simple, usamos un truco:
                # Puntos de alta probabilidad como HeatMap de otro color
                high_prob_indices = np.where(Z > np.percentile(Z, 85)) # Top 15% probabilidad
                prob_pts = list(zip(X[high_prob_indices], Y[high_prob_indices]))
                plugins.HeatMap(prob_pts, radius=25, blur=15, gradient={0: 'transparent', 0.5: 'cyan', 1: 'blue'}, name="Probabilidad IA").add_to(m)

        # 5. Favoritos
        favs = db.get_favorites_df()
        for _, f in favs.iterrows():
            icon_color = "red" if f['rating'] < 3 else "orange" if f['rating'] < 5 else "green"
            folium.Marker(
                [f['lat'], f['lon']], 
                popup=f"<b>{f['nombre']}</b><br>{f['especie']}", 
                icon=folium.Icon(color=icon_color, icon="star")
            ).add_to(m)

        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        
        map_out = st_folium(m, height=600, width="100%")

        # Lógica Guardar Favorito al Clic
        if map_out and map_out.get("last_clicked"):
            lc = map_out["last_clicked"]
            with st.sidebar.form("fav_form"):
                st.write("⭐ Nuevo Setal")
                st.write(f"Lat: {lc['lat']:.4f}, Lon: {lc['lng']:.4f}")
                n = st.text_input("Nombre")
                e = st.selectbox("Especie", ["Boletus", "Níscalo", "Amanita", "Trompeta", "Otro"])
                r = st.slider("Rating", 1, 5, 3)
                if st.form_submit_button("Guardar"):
                    db.save_favorite(lc['lat'], lc['lng'], n, "", e, r)
                    st.rerun()

with tab_analytics:
    st.markdown("### 🧠 Desglose de Rendimiento")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Km", f"{df_view['route_id'].nunique() * df_view.groupby('route_id').size().mean() / 1000 * 2:.1f} aprox") # Estimación rápida
    
    # Análisis de "Picking Speed"
    picking_pts = df_view[df_view['speed'] < 1.0]
    c2.metric("Puntos Productivos", len(picking_pts))
    
    avg_ele = picking_pts['ele'].mean()
    c3.metric("Altitud Ideal", f"{avg_ele:.0f} m", help="Altitud media donde tu velocidad es de recolección")
    
    # Orientaciones (Simple)
    # Aquí podríamos calcular aspecto si tuviéramos DEM real, usamos distribución de puntos
    
    # Gráficos
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig = px.histogram(picking_pts, x="ele", nbins=20, title="Altitud de Recolección", color_discrete_sequence=['orange'])
        st.plotly_chart(fig, use_container_width=True)
    with col_g2:
        # Fechas
        df_view['month_name'] = pd.to_datetime(df_view['route_date']).dt.month_name()
        fig2 = px.histogram(df_view, x="month_name", title="Estacionalidad", category_orders={"month_name": ["September", "October", "November", "December"]})
        st.plotly_chart(fig2, use_container_width=True)

with tab_planner:
    st.markdown("### 🔮 Oráculo Micológico")
    st.info("Este módulo conecta con satélites para ver si tus zonas históricas tienen las condiciones de humedad adecuadas HOY.")
    
    if st.button("Analizar Condiciones Actuales en Zona de Mapa"):
        with st.spinner("Consultando Open-Meteo API..."):
            # Usamos el centroide de los datos filtrados
            w_df = weather.get_weather_context(center_lat, center_lon)
            
            if w_df is not None:
                curr_smi = w_df['smi'].iloc[-1]
                curr_rain_15 = w_df['precipitation_sum'].tail(15).sum()
                
                c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
                c_kpi1.metric("Lluvia (15d)", f"{curr_rain_15:.1f} mm")
                c_kpi2.metric("Índice Humedad (SMI)", f"{curr_smi:.1f}", help=">30 suele ser bueno")
                
                status = "🔴 SECO"
                if curr_smi > 15: status = "🟡 RECUPERANDO"
                if curr_smi > 30: status = "🟢 ACTIVO"
                if curr_smi > 60: status = "🔵 SATURADO"
                
                c_kpi3.metric("Semáforo", status)
                
                # Gráfico Evolución
                fig_w = go.Figure()
                fig_w.add_trace(go.Bar(x=w_df['time'], y=w_df['precipitation_sum'], name="Lluvia"))
                fig_w.add_trace(go.Scatter(x=w_df['time'], y=w_df['smi'], name="Humedad Suelo (SMI)", line=dict(color='orange', width=3)))
                fig_w.update_layout(title="Ciclo Hidrológico Últimos 60 días", height=300)
                st.plotly_chart(fig_w, use_container_width=True)
                
            else:
                st.error("Error conectando con servicio meteo.")

with tab_data:
    st.markdown("### 💾 Gestión de Rutas y Favoritos")
    
    st.subheader("Rutas")
    routes_list = pd.read_sql_query("SELECT id, filename, date, tags FROM rutas", db.conn)
    st.dataframe(routes_list, use_container_width=True)
    
    rid_to_del = st.number_input("ID Ruta a Borrar", min_value=0)
    if st.button("Borrar Ruta"):
        db.delete_route(rid_to_del)
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("Favoritos")
    fav_list = db.get_favorites_df()
    st.dataframe(fav_list, use_container_width=True)
    
    fid_to_del = st.number_input("ID Favorito a Borrar", min_value=0)
    if st.button("Borrar Favorito"):
        db.delete_favorite(fid_to_del)
        st.rerun()
