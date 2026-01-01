import streamlit as st
import gpxpy
import pandas as pd
import folium
from folium.plugins import HeatMap, Fullscreen, MarkerCluster
from streamlit_folium import st_folium
import sqlite3
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="MicoData: Dashboard", layout="wide", page_icon="🍄")

# --- CSS PARA MAXIMIZAR ESPACIO ---
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        h1 {margin-bottom: 0rem;}
        .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

# --- GESTOR DE BD OPTIMIZADO PARA BIG DATA ---
class DBManager:
    def __init__(self):
        self.conn = sqlite3.connect('micodata_global.db', check_same_thread=False)
        self.create_tables()
        self.init_species_data()

    def create_tables(self):
        c = self.conn.cursor()
        
        # Tabla de rutas principal
        c.execute('''
            CREATE TABLE IF NOT EXISTS rutas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                date TEXT,
                min_lat REAL, max_lat REAL, min_lon REAL, max_lon REAL,
                points_json TEXT,
                productivity_rating INTEGER DEFAULT NULL,
                notes TEXT DEFAULT NULL,
                species_found TEXT DEFAULT NULL
            )
        ''')
        
        # Tabla de categorías
        c.execute('''
            CREATE TABLE IF NOT EXISTS categorias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                color TEXT,
                icon TEXT,
                description TEXT
            )
        ''')
        
        # Tabla de relación rutas-categorías
        c.execute('''
            CREATE TABLE IF NOT EXISTS ruta_categorias (
                ruta_id INTEGER,
                categoria_id INTEGER,
                PRIMARY KEY (ruta_id, categoria_id),
                FOREIGN KEY (ruta_id) REFERENCES rutas(id),
                FOREIGN KEY (categoria_id) REFERENCES categorias(id)
            )
        ''')
        
        # Tabla de favoritos/POIs
        c.execute('''
            CREATE TABLE IF NOT EXISTS favoritos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                lat REAL,
                lon REAL,
                description TEXT,
                species TEXT,
                productivity_rating INTEGER,
                elevation REAL,
                created_at TEXT,
                updated_at TEXT,
                last_visit TEXT,
                visit_count INTEGER DEFAULT 1
            )
        ''')
        
        # Tabla de predicciones aprendidas
        c.execute('''
            CREATE TABLE IF NOT EXISTS prediction_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                zone_center_lat REAL,
                zone_center_lon REAL,
                optimal_months TEXT,
                optimal_elevation_min REAL,
                optimal_elevation_max REAL,
                optimal_temp_min REAL,
                optimal_temp_max REAL,
                humidity_threshold REAL,
                success_rate REAL,
                sample_count INTEGER
            )
        ''')
        
        self.conn.commit()

    def init_species_data(self):
        """Inicializa categorías por defecto"""
        c = self.conn.cursor()
        default_cats = [
            ('Boletus', '#8B0000', '🍄', 'Rutas enfocadas a Boletus'),
            ('Níscalos', '#FF8C00', '🍊', 'Rutas de Lactarius deliciosus'),
            ('Champiñones', '#808080', '🍽️', 'Agaricus y similares'),
            ('Setas de Primavera', '#90EE90', '🌸', 'Especies de temporada primavera'),
            ('Alta Montaña', '#4169E1', '🏔️', 'Tracks por encima de 1500m'),
            ('Bosque de Coníferas', '#228B22', '🌲', 'Pinares y abetales'),
            ('Bosque de Frondosas', '#8B4513', '🍂', 'Robles, hayas, castaños'),
            ('Zona Húmeda', '#00CED1', '💧', 'Cerca de ríos o zonas encharcadas'),
            ('Exploración', '#9932CC', '🔍', 'Zonas nuevas por explorar')
        ]
        for cat in default_cats:
            c.execute('INSERT OR IGNORE INTO categorias (name, color, icon, description) VALUES (?, ?, ?, ?)', cat)
        self.conn.commit()

    def save_route(self, data, categories=None, productivity=None, notes=None, species=None):
        c = self.conn.cursor()
        lats = [p['lat'] for p in data['points']]
        lons = [p['lon'] for p in data['points']]
        
        c.execute('''
            INSERT INTO rutas (filename, date, min_lat, max_lat, min_lon, max_lon, points_json, productivity_rating, notes, species_found)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data['filename'], data['date'], min(lats), max(lats), min(lons), max(lons), 
              json.dumps(data['points']), productivity, notes, species))
        
        ruta_id = c.lastrowid
        
        if categories:
            for cat_id in categories:
                c.execute('INSERT INTO ruta_categorias (ruta_id, categoria_id) VALUES (?, ?)', (ruta_id, cat_id))
        
        self.conn.commit()
        return ruta_id

    def save_favorite(self, data):
        c = self.conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute('''
            INSERT INTO favoritos (name, lat, lon, description, species, productivity_rating, elevation, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data['name'], data['lat'], data['lon'], data.get('description', ''), 
              data.get('species', ''), data.get('rating', 0), data.get('elevation', 0), now, now))
        self.conn.commit()
        return c.lastrowid

    def update_favorite_visit(self, fav_id):
        c = self.conn.cursor()
        c.execute('UPDATE favoritos SET last_visit = ?, visit_count = visit_count + 1 WHERE id = ?', 
                  (datetime.now().strftime('%Y-%m-%d'), fav_id))
        self.conn.commit()

    def get_all_rutas(self):
        return pd.read_sql_query("SELECT * FROM rutas", self.conn)

    def get_all_favorites(self):
        return pd.read_sql_query("SELECT * FROM favoritos", self.conn)

    def get_rutas_with_categories(self):
        df = pd.read_sql_query('''
            SELECT r.*, GROUP_CONCAT(c.name) as categories
            FROM rutas r
            LEFT JOIN ruta_categorias rc ON r.id = rc.ruta_id
            LEFT JOIN categorias c ON rc.categoria_id = c.id
            GROUP BY r.id
        ''', self.conn)
        return df

    def get_categories(self):
        return pd.read_sql_query("SELECT * FROM categorias", self.conn)

    @st.cache_data(ttl=600)
    def get_all_points_dataframe(_self):
        df_routes = _self.get_all_rutas()
        
        all_points = []
        for _, row in df_routes.iterrows():
            points = json.loads(row['points_json'])
            route_date = pd.to_datetime(row['date'])
            for p in points:
                p['route_id'] = row['id']
                p['route_date'] = route_date
                p['year'] = route_date.year
                p['month'] = route_date.month
                p['productivity'] = row['productivity_rating']
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
                for i, point in enumerate(segment.points):
                    if i % 3 == 0:
                        ele = point.elevation if point.elevation else 0
                        points.append({'lat': point.latitude, 'lon': point.longitude, 'ele': ele, 'speed': 0})
        if points:
            date_obj = gpx.time if gpx.time else datetime.now()
            return {'filename': filename, 'date': date_obj.strftime('%Y-%m-%d'), 'points': points}
    except Exception as e:
        st.error(f"Error parseando GPX: {e}")
        return None

# --- METEO ---
@st.cache_data(ttl=3600)
def get_climate_data(lat, lon, days=90):
    end = datetime.now().date()
    start = end - timedelta(days=days)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ["precipitation_sum", "temperature_2m_mean", "soil_temperature_0_to_7cm_mean", "relative_humidity_2m_mean"],
        "timezone": "auto"
    }
    try:
        r = requests.get(url, params=params).json()
        return pd.DataFrame(r['daily'])
    except Exception as e:
        st.error(f"Error obteniendo datos climáticos: {e}")
        return None

@st.cache_data(ttl=3600)
def get_current_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": ["temperature_2m", "precipitation", "relative_humidity_2m", "soil_temperature_0_to_7cm"],
        "timezone": "auto"
    }
    try:
        r = requests.get(url, params=params).json()
        return r.get('current', {})
    except:
        return None

# --- MODELO PREDICTIVO DE ZONAS ---
class ZonePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def prepare_features(self, df):
        """Extrae características de cada zona para predicción"""
        if df.empty:
            return None
        
        features = []
        zones = []
        
        # Clustering de puntos para identificar zonas
        coords = df[['lat', 'lon']].values
        clustering = DBSCAN(eps=0.01, min_samples=10).fit(coords)
        df['cluster'] = clustering.labels_
        
        for cluster_id in df['cluster'].unique():
            if cluster_id == -1:
                continue  # Ignorar ruido
            cluster_data = df[df['cluster'] == cluster_id]
            
            zone_features = {
                'center_lat': cluster_data['lat'].mean(),
                'center_lon': cluster_data['lon'].mean(),
                'elevation_mean': cluster_data['ele'].mean(),
                'elevation_std': cluster_data['ele'].std(),
                'elevation_min': cluster_data['ele'].min(),
                'elevation_max': cluster_data['ele'].max(),
                'activity_count': len(cluster_data),
                'productivity_mean': cluster_data['productivity'].mean() if 'productivity' in df.columns else 0,
                'months_active': cluster_data['month'].nunique()
            }
            features.append(list(zone_features.values()))
            zones.append(zone_features)
        
        if features:
            X = np.array(features)
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled, zones
        return None, None

    def get_best_zones(self, df, top_n=5):
        """Identifica las zonas de mayor éxito históricamente"""
        if df.empty:
            return []
        
        features, zones = self.prepare_features(df)
        if features is None:
            return []
        
        # Calcular score de zona basado en actividad y productividad
        zone_scores = []
        for i, zone in enumerate(zones):
            score = (
                zone['activity_count'] * 0.3 +
                (zone['productivity_mean'] or 0) * 50 +
                zone['months_active'] * 2 +
                (1 if zone['elevation_mean'] > 1000 else 0) * 10
            )
            zone_scores.append({
                'lat': zone['center_lat'],
                'lon': zone['center_lon'],
                'elevation': zone['elevation_mean'],
                'activity': zone['activity_count'],
                'productivity': zone['productivity_mean'],
                'score': score,
                'months': zone['months_active']
            })
        
        zone_scores.sort(key=lambda x: x['score'], reverse=True)
        return zone_scores[:top_n]

    def predict_optimal_zones(self, current_weather, df):
        """Predice zonas óptimas basadas en condiciones actuales"""
        best_zones = self.get_best_zones(df, top_n=10)
        if not best_zones:
            return []
        
        # Filtrar zonas por condiciones actuales
        recommended = []
        for zone in best_zones:
            # Lógica simple de filtrado
            elevation_factor = 1
            if current_weather:
                temp = current_weather.get('temperature_2m', 15)
                humidity = current_weather.get('relative_humidity_2m', 50)
                rain = current_weather.get('precipitation', 0)
                
                # Ajustar score por condiciones
                if 10 <= temp <= 20:
                    elevation_factor *= 1.2
                if humidity > 60:
                    elevation_factor *= 1.1
                if rain > 0:
                    elevation_factor *= 1.3
            
            zone['recommended_score'] = zone['score'] * elevation_factor
            recommended.append(zone)
        
        recommended.sort(key=lambda x: x['recommended_score'], reverse=True)
        return recommended[:5]

# --- PREDICTOR DE FECHAS ---
class DatePredictor:
    def __init__(self):
        self.optimal_conditions = {}

    def analyze_patterns(self, df_rutas, climate_history):
        """Analiza patrones históricos para predecir fechas óptimas"""
        if df_rutas.empty:
            return {}
        
        # Agrupar por mes y analizar productividad
        monthly_stats = df_rutas.groupby('month').agg({
            'productivity_rating': ['mean', 'count']
        }).reset_index()
        monthly_stats.columns = ['month', 'avg_productivity', 'count']
        
        # Identificar meses óptimos
        optimal_months = monthly_stats[monthly_stats['avg_productivity'] > monthly_stats['avg_productivity'].quantile(0.7)]['month'].tolist()
        
        return {
            'optimal_months': optimal_months,
            'monthly_stats': monthly_stats,
            'best_month': monthly_stats.loc[monthly_stats['avg_productivity'].idxmax(), 'month'] if not monthly_stats.empty else None
        }

    def predict_next_optimal_date(self, patterns, current_weather):
        """Predice la próxima fecha óptima basada en patrones"""
        if not patterns or 'optimal_months' not in patterns:
            return None
        
        now = datetime.now()
        current_month = now.month
        
        # Buscar el próximo mes óptimo
        optimal_months = sorted(patterns['optimal_months'])
        
        for month in optimal_months:
            if month > current_month:
                # Próximo mes óptimo
                next_date = datetime(now.year, month, 1)
                days_ahead = (next_date - now).days
                return {
                    'target_month': month,
                    'target_month_name': ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'][month-1],
                    'days_ahead': days_ahead,
                    'reason': f"Típicamente productivo en {['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'][month-1]}"
                }
        
        # Si no hay mes óptimo adelante, buscar en el próximo año
        if optimal_months:
            next_month = optimal_months[0]
            next_date = datetime(now.year + 1, next_month, 1)
            days_ahead = (next_date - now).days
            return {
                'target_month': next_month,
                'target_month_name': ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'][next_month-1],
                'days_ahead': days_ahead,
                'reason': f"Típicamente productivo en {['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'][next_month-1]} (próximo año)"
            }
        
        return None

# --- OBTENER DATOS DE ELEVACION ---
def get_elevation_data(lat, lon):
    """Obtiene datos de elevación para un punto"""
    # Usar la API de Open-Elevation o similar
    url = f"https://api.open-elevation.com/api/v1/lookup?points=%7B%22latitude%22%3A{lat}%2C%22longitude%22%3A{lon}%7D"
    try:
        response = requests.get(url).json()
        if 'results' in response and response['results']:
            return response['results'][0].get('elevation', 0)
    except:
        pass
    return 0

# --- UI PRINCIPAL ---

# Inicializar predictores
zone_predictor = ZonePredictor()
date_predictor = DatePredictor()

# 1. BARRA LATERAL
with st.sidebar:
    st.title("📂 Cargar GPX")
    uploaded_files = st.file_uploader("Arrastra tus archivos aquí", accept_multiple_files=True)
    
    if uploaded_files and st.button("Procesar", type="primary"):
        with st.expander("Opciones de carga", expanded=True):
            st.write("Asigna categorías a estos tracks:")
            all_cats = db.get_categories()
            selected_cats = st.multiselect("Categorías", all_cats['name'].tolist())
            cat_ids = [all_cats[all_cats['name'] == c]['id'].values[0] for c in selected_cats]
            
            col1, col2 = st.columns(2)
            with col1:
                productivity = st.slider("Productividad", 1, 5, 3, help="¿Cuánta suerte tuviste en esta ruta?")
            with col2:
                species = st.text_input("Especies encontradas", placeholder="Boletus, Níscalos...")
            
            notes = st.text_area("Notas", placeholder="Observaciones sobre la ruta...")
            
            if st.button("Guardar todo"):
                bar = st.progress(0)
                for i, f in enumerate(uploaded_files):
                    s_io = io.StringIO(f.getvalue().decode("utf-8"))
                    data = parse_gpx(s_io, f.name)
                    if data:
                        db.save_route(data, cat_ids if cat_ids else None, productivity, notes, species)
                    bar.progress((i+1)/len(uploaded_files))
                st.success("¡Rutas procesadas!")
                st.cache_data.clear()
                st.rerun()
    
    st.divider()
    
    # Gestión de categorías
    with st.expander("🏷️ Gestionar Categorías"):
        new_cat_name = st.text_input("Nueva categoría")
        new_cat_color = st.color_picker("Color", "#4169E1")
        new_cat_desc = st.text_input("Descripción")
        if st.button("Crear categoría"):
            if new_cat_name:
                c = db.conn.cursor()
                try:
                    c.execute('INSERT INTO categorias (name, color, icon, description) VALUES (?, ?, ?, ?)',
                              (new_cat_name, new_cat_color, "📁", new_cat_desc))
                    db.conn.commit()
                    st.success("Categoría creada")
                    st.rerun()
                except:
                    st.error("Ya existe esa categoría")
    
    # Filtrar por categoría
    st.divider()
    st.write("### 🔍 Filtrar por categoría")
    all_cats = db.get_categories()
    cat_filter = st.multiselect("Categorías", all_cats['name'].tolist())
    
    # Gestión de favoritos
    st.divider()
    st.write("### ⭐ Añadir Favorito")
    with st.form("add_favorite"):
        fav_name = st.text_input("Nombre del lugar")
        fav_lat = st.number_input("Latitud", value=40.4168, format="%.6f")
        fav_lon = st.number_input("Longitud", value=-3.7038, format="%.6f")
        fav_species = st.text_input("Especies típicas")
        fav_rating = st.slider("Productividad típica", 1, 5, 3)
        fav_desc = st.text_area("Descripción")
        
        if st.form_submit_button("Guardar Favorito"):
            fav_data = {
                'name': fav_name,
                'lat': fav_lat,
                'lon': fav_lon,
                'species': fav_species,
                'rating': fav_rating,
                'description': fav_desc,
                'elevation': get_elevation_data(fav_lat, fav_lon)
            }
            db.save_favorite(fav_data)
            st.success("Favorito guardado")
            st.rerun()

# 2. CARGA DE DATOS
df_all = db.get_all_points_dataframe()
df_rutas = db.get_rutas_with_categories()
df_favorites = db.get_all_favorites()

if df_all.empty:
    st.warning("No hay rutas. Sube tus GPX en el menú lateral.")
    st.stop()

# Aplicar filtros de categoría
if cat_filter:
    cat_filter_ids = [all_cats[all_cats['name'] == c]['id'].values[0] for c in cat_filter]
    valid_ruta_ids = pd.read_sql_query(f'''
        SELECT DISTINCT ruta_id FROM ruta_categorias WHERE categoria_id IN ({','.join(map(str, cat_filter_ids))})
    ''', db.conn)['ruta_id'].tolist()
    df_all = df_all[df_all['route_id'].isin(valid_ruta_ids)]
    df_rutas = df_rutas[df_rutas['id'].isin(valid_ruta_ids)]

# --- LAYOUT DASHBOARD ---

st.title("🍄 MicoData: Dashboard de Forrajeo")

# Fila 1: El Mapa y los Controles de Análisis
col_map, col_kpi = st.columns([3, 1])

with col_kpi:
    st.subheader("🔭 Análisis Local")
    st.info("Mueve el mapa a la zona que quieras investigar y pulsa el botón.")
    
    if 'map_bounds' not in st.session_state:
        st.session_state.map_bounds = None
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [40.416, -3.703]
    
    analyze_btn = st.button("📍 ANALIZAR ZONA", type="primary", use_container_width=True)
    
    heatmap_intensity = st.slider("Intensidad Heatmap", 0.1, 1.0, 0.5)
    show_topography = st.checkbox("Mostrar topografía", value=False)
    show_favorites = st.checkbox("Mostrar favoritos", value=True)
    show_predictions = st.checkbox("Mostrar predicciones", value=True)
    
    st.divider()
    st.metric("Total Puntos", f"{len(df_all):,}")
    st.metric("Rutas guardadas", len(df_rutas))
    st.metric("Años", f"{df_all['year'].min()} - {df_all['year'].max()}")
    if not df_favorites.empty:
        st.metric("Favoritos", len(df_favorites))

with col_map:
    m = folium.Map(location=st.session_state.map_center, zoom_start=7, tiles=None)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                     attr='Esri', name="Satélite").add_to(m)
    folium.TileLayer('OpenStreetMap', name="Callejero").add_to(m)
    
    # Capa 1: HEATMAP GLOBAL
    heat_data = df_all[['lat', 'lon']].values.tolist()
    HeatMap(heat_data, radius=12, blur=15, min_opacity=heatmap_intensity,
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
    
    # Capa 2: Curvas de nivel (simplificado)
    if show_topography:
        # Añadir marcadores de elevación en puntos estratégicos
        if not df_all.empty:
            elevation_sample = df_all.sample(min(50, len(df_all)))
            for _, row in elevation_sample.iterrows():
                if row['ele'] > 1000:
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=3,
                        color='brown',
                        fill=True,
                        popup=f"Cota: {row['ele']:.0f}m"
                    ).add_to(m)
    
    # Capa 3: Favoritos
    if show_favorites and not df_favorites.empty:
        fav_group = folium.FeatureGroup(name="⭐ Favoritos")
        for _, fav in df_favorites.iterrows():
            rating = fav.get('productivity_rating', 3)
            color = 'green' if rating >= 4 else 'orange' if rating >= 3 else 'red'
            folium.Marker(
                location=[fav['lat'], fav['lon']],
                popup=folium.Popup(f"""
                    <b>⭐ {fav['name']}</b><br>
                    <b>Especies:</b> {fav.get('species', 'N/A')}<br>
                    <b>Productividad:</b> {'★' * rating}<br>
                    <b>Visitas:</b> {fav.get('visit_count', 1)}<br>
                    <b>Última:</b> {fav.get('last_visit', 'N/A')}<br>
                    <i>{fav.get('description', '')}</i>
                """, max_width=300),
                icon=folium.Icon(color=color, icon='star', prefix='fa')
            ).add_to(fav_group)
        fav_group.add_to(m)
    
    # Capa 4: Zonas Predichas
    if show_predictions:
        best_zones = zone_predictor.get_best_zones(df_all, top_n=10)
        pred_group = folium.FeatureGroup(name="🎯 Zonas Predichas")
        for i, zone in enumerate(best_zones[:5]):
            folium.CircleMarker(
                location=[zone['lat'], zone['lon']],
                radius=15 - i * 2,
                color='purple',
                fill=True,
                fill_color='purple',
                fill_opacity=0.6,
                popup=folium.Popup(f"""
                    <b>Zona Recomendada #{i+1}</b><br>
                    <b>Score:</b> {zone['score']:.1f}<br>
                    <b>Altitud:</b> {zone['elevation']:.0f}m<br>
                    <b>Actividad:</b> {zone['activity']} puntos<br>
                    <b>Productividad:</b> {zone['productivity']:.1f}/5<br>
                    <b>Meses activos:</b> {zone['months']}
                """, max_width=250)
            ).add_to(pred_group)
        pred_group.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    map_data = st_folium(m, height=550, width="100%", key="main_map")

# --- LÓGICA DE FILTRADO Y ANÁLISIS ---
filtered_df = df_all
climate_df = None
current_weather = None
location_msg = "Mostrando datos globales"
prediction_patterns = None

if analyze_btn and map_data and map_data.get('bounds'):
    bounds = map_data['bounds']
    sw = bounds['_southWest']
    ne = bounds['_northEast']
    center = map_data['center']
    
    st.session_state.map_center = [center['lat'], center['lng']]
    
    # Filtrar por coordenadas
    mask = (
        (df_all['lat'] >= sw['lat']) & (df_all['lat'] <= ne['lat']) &
        (df_all['lon'] >= sw['lng']) & (df_all['lon'] <= ne['lng'])
    )
    filtered_df = df_all.loc[mask]
    
    # Obtener clima
    with st.spinner(f"Analizando clima en {center['lat']:.2f}, {center['lng']:.2f}..."):
        climate_df = get_climate_data(center['lat'], center['lng'])
        current_weather = get_current_weather(center['lat'], center['lng'])
    
    # Analizar patrones para predicciones
    prediction_patterns = date_predictor.analyze_patterns(df_rutas[df_rutas['id'].isin(filtered_df['route_id'].unique())], climate_df)
    
    location_msg = f"📍 Análisis de zona visible ({len(filtered_df)} puntos)"

# --- PANELES DE RESULTADOS ---

st.subheader(location_msg)

if filtered_df.empty:
    st.warning("No hay rutas en la zona visible.")
else:
    # Panel de Predicciones y Recomendaciones
    with st.expander("🎯 PREDICCIONES Y RECOMENDACIONES", expanded=True):
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        
        with col_pred1:
            st.markdown("#### 🌲 Mejores Zonas")
            best_zones = zone_predictor.get_best_zones(filtered_df, top_n=5)
            if best_zones:
                for i, zone in enumerate(best_zones):
                    st.write(f"**#{i+1}** Score: {zone['score']:.1f} | {zone['elevation']:.0f}m | {zone['activity']} pts")
            else:
                st.info("Más datos necesarios para predicciones")
        
        with col_pred2:
            st.markdown("#### 📅 Próxima Fecha Óptima")
            if prediction_patterns and prediction_patterns.get('best_month'):
                pred = date_predictor.predict_next_optimal_date(prediction_patterns, current_weather)
                if pred:
                    st.success(f"**{pred['target_month_name']}** (en {pred['days_ahead']} días)")
                    st.caption(pred['reason'])
                else:
                    st.info("Sin predicciones claras")
            else:
                st.info("Añade ratings de productividad para predicciones")
        
        with col_pred3:
            st.markdown("#### 🌦️ Condiciones Actuales")
            if current_weather:
                temp = current_weather.get('temperature_2m', 0)
                humidity = current_weather.get('relative_humidity_2m', 0)
                rain = current_weather.get('precipitation', 0)
                st.metric("Temperatura", f"{temp:.1f}°C")
                st.metric("Humedad", f"{humidity:.0f}%")
                st.metric("Lluvia actual", f"{rain:.1f}mm")
            else:
                st.info("Sin datos meteorológicos")

    # Fila de análisis: Clima, Topografía, Patrones
    c_climate, c_topo, c_pattern = st.columns([1.5, 1, 1])

    with c_climate:
        st.markdown("#### 🌦️ Clima (Últimos 90 días)")
        if climate_df is not None and not climate_df.empty:
            climate_df['time'] = pd.to_datetime(climate_df['time'])
            
            # Calcular índices
            smi = []
            curr = 0
            for rain in climate_df['precipitation_sum']:
                curr = (curr * 0.9) + rain
                smi.append(curr)
            climate_df['SMI'] = smi
            
            # Gráfico climático
            fig = go.Figure()
            fig.add_trace(go.Bar(x=climate_df['time'], y=climate_df['precipitation_sum'], 
                                name="Lluvia (mm)", marker_color="skyblue"))
            fig.add_trace(go.Scatter(x=climate_df['time'], y=climate_df['SMI'], 
                                    name="Humedad Suelo", line=dict(color="blue", width=3), yaxis="y2"))
            fig.add_trace(go.Scatter(x=climate_df['time'], y=climate_df['temperature_2m_mean'], 
                                    name="Temp Media", line=dict(color="orange", dash="dot"), yaxis="y3"))
            
            fig.update_layout(
                height=280, margin=dict(l=0, r=0, t=30, b=0),
                yaxis=dict(title="Lluvia", domain=[0, 0.55]),
                yaxis2=dict(title="Humedad", overlaying="y", side="right"),
                yaxis3=dict(title="Temp", overlaying="y", side="left", position=0.05, domain=[0.6, 1]),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Semáforo
            last_rain_15 = climate_df['precipitation_sum'].tail(15).sum()
            last_smi = smi[-1]
            last_temp = climate_df['temperature_2m_mean'].iloc[-1]
            
            conditions = []
            if last_rain_15 > 20: conditions.append("🌧️ Lluvia reciente")
            if last_smi > 50: conditions.append("💧 Humedad alta")
            if 10 <= last_temp <= 20: conditions.append("🌡️ Temperatura óptima")
            
            if conditions:
                st.write("**Condiciones actuales:** " + " | ".join(conditions))
            else:
                st.caption("Condiciones no óptimas actualmente")

    with c_topo:
        st.markdown("#### ⛰️ Topografía")
        if not filtered_df.empty:
            fig_ele = px.histogram(filtered_df, x="ele", nbins=25, 
                                   title="Distribución de Altitud (m)",
                                   color_discrete_sequence=['#8B4513'])
            fig_ele.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            st.plotly_chart(fig_ele, use_container_width=True)
            
            mean_ele = filtered_df['ele'].mean()
            min_ele = filtered_df['ele'].min()
            max_ele = filtered_df['ele'].max()
            
            col_e1, col_e2 = st.columns(2)
            col_e1.metric("Cota Media", f"{mean_ele:.0f} m")
            col_e2.metric("Rango", f"{min_ele:.0f}-{max_ele:.0f}")
            
            # Especies potenciales según elevación
            if mean_ele > 1500:
                st.caption("🧊 **Alta montaña:** Boletus, Lactarius)
            elif mean_ele > 800:
                st.caption("🌲 **Montaña media:** Níscalos, Boletus, Robellones")
            else:
                st.caption("🍂 **Tierras bajas:** Champiñones, setas de otoño")

    with c_pattern:
        st.markdown("#### 📅 Patrón Temporal")
        month_counts = filtered_df['month'].value_counts().sort_index()
        month_names = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        df_months = pd.DataFrame({
            'Mes': [month_names[x-1] for x in month_counts.index],
            'Actividad': month_counts.values
        })
        
        fig_time = px.bar(df_months, x='Mes', y='Actividad', 
                          title="Tus visitas históricas",
                          color='Actividad', color_continuous_scale='Viridis')
        fig_time.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_time, use_container_width=True)
        
        if not df_months.empty:
            best_month = df_months.loc[df_months['Actividad'].idxmax()]['Mes']
            st.success(f"Mejor mes: **{best_month}**")

# --- SECCIÓN DE ANÁLISIS AVANZADO ---
with st.expander("📊 Análisis Avanzado de Productividad", expanded=False):
    col_adv1, col_adv2 = st.columns(2)
    
    with col_adv1:
        st.markdown("##### 📈 Productividad por Categoría")
        if cat_filter:
            df_rutas_filt = df_rutas[df_rutas['id'].isin(filtered_df['route_id'].unique())]
            if not df_rutas_filt.empty and 'categories' in df_rutas_filt.columns:
                # Desglose por categoría
                cat_stats = []
                for _, row in df_rutas_filt.iterrows():
                    if row['categories']:
                        for cat in str(row['categories']).split(','):
                            cat_stats.append({'Categoria': cat, 'Productividad': row.get('productivity_rating', 0)})
                
                if cat_stats:
                    df_cat_stats = pd.DataFrame(cat_stats)
                    fig_cat = px.box(df_cat_stats, x='Categoria', y='Productividad', 
                                     title="Distribución de Productividad por Categoría",
                                     color='Categoria')
                    fig_cat.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig_cat, use_container_width=True)
    
    with col_adv2:
        st.markdown("##### 🗺️ Zonas de Alta Productividad")
        # Scatter map de zonas productivas
        if not df_rutas.empty:
            productive_routes = df_rutas[df_rutas['productivity_rating'] >= 4]
            if not productive_routes.empty:
                fig_scatter = px.scatter_mapbox(
                    pd.DataFrame({
                        'lat': [(r['min_lat'] + r['max_lat'])/2 for _, r in productive_routes.iterrows()],
                        'lon': [(r['min_lon'] + r['max_lon'])/2 for _, r in productive_routes.iterrows()],
                        'rating': productive_routes['productivity_rating'],
                        'date': productive_routes['date']
                    }),
                    lat='lat', lon='lon', color='rating', size='rating',
                    hover_data=['date'],
                    title="🌟 Zonas con Alta Productividad (rating ≥ 4)",
                    mapbox_style="open-street-map"
                )
                fig_scatter.update_layout(height=300)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No hay rutas con rating ≥ 4. Añade ratings para ver este análisis.")

# --- SECCIÓN DE FAVORITOS ---
if not df_favorites.empty:
    with st.expander("⭐ Mis Favoritos", expanded=False):
        col_fav1, col_fav2 = st.columns([2, 1])
        
        with col_fav1:
            st.write("### Lista de Favoritos")
            for _, fav in df_favorites.iterrows():
                with st.container():
                    col_f, col_d = st.columns([1, 4])
                    with col_f:
                        rating = fav.get('productivity_rating', 3)
                        st.write(f"{'⭐' * rating}")
                    with col_d:
                        st.write(f"**{fav['name']}** | {fav.get('species', 'N/A')} | {fav.get('visit_count', 1)} visitas")
                        if st.button(f"Registrar visita #{fav['id']}", key=f"visit_{fav['id']}"):
                            db.update_favorite_visit(fav['id'])
                            st.success("¡Visita registrada!")
                            st.rerun()
                    st.divider()
        
        with col_fav2:
            st.write("### Estadísticas de Favoritos")
            avg_rating = df_favorites['productivity_rating'].mean() if 'productivity_rating' in df_favorites.columns else 0
            total_visits = df_favorites['visit_count'].sum() if 'visit_count' in df_favorites.columns else len(df_favorites)
            st.metric("Rating promedio", f"{avg_rating:.1f}/5")
            st.metric("Total visitas", total_visits)
            
            # Mapa de favoritos
            if len(df_favorites) > 0:
                st.write("### Mapa de Favoritos")
                fav_map = folium.Map(location=[df_favorites['lat'].mean(), df_favorites['lon'].mean()], zoom_start=7)
                for _, fav in df_favorites.iterrows():
                    folium.Marker(
                        location=[fav['lat'], fav['lon']],
                        popup=fav['name']
                    ).add_to(fav_map)
                st_folium(fav_map, height=200, width="100%")
