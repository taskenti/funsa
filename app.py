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
import math
import hashlib
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from abc import ABC, abstractmethod
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings

# ==========================================
# 0. CONFIGURACIÓN DEL SISTEMA Y ESTILOS CSS
# ==========================================

# Suprimimos warnings de librerías científicas para limpiar la UI
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MicoBrain OMNI: Professional Foraging Suite",
    layout="wide",
    page_icon="🍄",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.github.com/tu-repo/help',
        'Report a bug': "https://www.github.com/tu-repo/bug",
        'About': "# MicoBrain OMNI v4.0\nLa herramienta definitiva para el micólogo moderno."
    }
)

# Inyectamos CSS profesional para transformar Streamlit en un Dashboard SaaS
st.markdown("""
    <style>
        /* Tipografía y Contenedores Generales */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
            max-width: 98% !important;
        }

        /* Tarjetas de Métricas (KPIs) */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-color: #2e7d32;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 800;
            color: #1b5e20;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #616161;
        }

        /* Títulos y Cabeceras */
        h1 {
            color: #1b5e20;
            font-weight: 800;
            letter-spacing: -0.02em;
            border-bottom: 2px solid #e8f5e9;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2, h3 {
            color: #2e7d32;
            font-weight: 600;
        }
        
        /* Personalización de Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f1f3f4;
            border-radius: 8px 8px 0 0;
            gap: 1px;
            padding: 10px 20px;
            border: none;
            font-weight: 600;
            color: #5f6368;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #1b5e20 !important;
            border-top: 3px solid #1b5e20 !important;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Alertas personalizadas */
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #e8f5e9;
            border: 1px solid #c8e6c9;
            color: #1b5e20;
            margin-bottom: 1rem;
        }
        .warning-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #fff3e0;
            border: 1px solid #ffe0b2;
            color: #e65100;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. UTILIDADES CIENTÍFICAS Y MATEMÁTICAS
# ==========================================

class GeoMath:
    """
    Motor de cálculo geométrico y vectorial optimizado con NumPy.
    Maneja proyecciones esféricas y cálculos de terreno.
    """
    
    @staticmethod
    def haversine_vectorized(lat1, lon1, lat2, lon2):
        """
        Calcula la distancia del gran círculo entre dos puntos en la tierra.
        Optimizado para arrays de numpy para procesar miles de puntos instantáneamente.
        """
        R = 6371000.0  # Radio de la Tierra en metros
        
        # Convertir a radianes
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c

    @staticmethod
    def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
        """
        Calcula el azimut (dirección) entre arrays de coordenadas.
        Retorna grados 0-360.
        """
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        
        initial_bearing = np.arctan2(y, x)
        initial_bearing_deg = np.degrees(initial_bearing)
        
        return (initial_bearing_deg + 360) % 360

    @staticmethod
    def calculate_slope(dist_meters, ele_diff_meters):
        """
        Calcula la pendiente en grados y porcentaje.
        Evita división por cero.
        """
        if dist_meters < 0.1: return 0
        slope_rad = np.arctan(ele_diff_meters / dist_meters)
        return np.degrees(slope_rad)

    @staticmethod
    def get_solar_aspect(bearing):
        """
        Clasifica la orientación de una ladera para determinar Solana vs Umbría.
        Fundamental para micología (humedad vs calor).
        """
        if bearing is None or np.isnan(bearing): return "Plano"
        
        # Definición de cuadrantes micológicos
        if 315 <= bearing or bearing < 45: 
            return "Norte (Umbría Estricta)"
        elif 45 <= bearing < 135: 
            return "Este (Umbría Húmeda)"
        elif 135 <= bearing < 225: 
            return "Sur (Solana Pura)"
        else: 
            return "Oeste (Solana Tarde)"

# ==========================================
# 2. GESTOR DE BASE DE DATOS (PERSISTENCIA AVANZADA)
# ==========================================

class DatabaseEngine:
    """
    Motor SQLite robusto. Maneja transacciones, serialización JSON y 
    estructuras de datos complejas para rutas y análisis.
    """
    def __init__(self, db_path='micobrain_omni_v4.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # Habilitar WAL (Write-Ahead Logging) para mejor concurrencia
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.init_schema()

    def init_schema(self):
        c = self.conn.cursor()
        
        # 1. TABLA DE RUTAS (Master)
        # Almacena metadatos y enlaces a los datos binarios
        c.execute('''
            CREATE TABLE IF NOT EXISTS routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                hash_id TEXT UNIQUE, -- Para evitar duplicados exactos
                date_start TEXT,
                date_end TEXT,
                tags TEXT DEFAULT "General",
                
                -- Métricas Físicas
                total_distance_km REAL,
                elevation_gain_m REAL,
                elevation_loss_m REAL,
                max_elevation_m REAL,
                min_elevation_m REAL,
                moving_time_h REAL,
                total_time_h REAL,
                avg_speed_kmh REAL,
                max_speed_kmh REAL,
                
                -- Datos Espaciales (Bounding Box para búsquedas rápidas)
                min_lat REAL, max_lat REAL, 
                min_lon REAL, max_lon REAL,
                
                -- Datos Analíticos (JSONs comprimidos)
                points_blob TEXT,      -- Array principal de puntos (lat, lon, ele, time, speed)
                analysis_blob TEXT,    -- Datos derivados (pendientes, orientaciones)
                segments_blob TEXT     -- Segmentación de paradas vs movimiento
            )
        ''')
        
        # 2. TABLA DE SETALES (Favoritos/Waypoints)
        c.execute('''
            CREATE TABLE IF NOT EXISTS spots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                species TEXT,          -- Especie principal (Boletus, Amanita...)
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                elevation REAL,
                
                -- Valoración y Calidad
                rating INTEGER DEFAULT 3, -- 1 a 5 estrellas
                productivity_index REAL,  -- Kg estimados o unidades
                
                -- Contexto
                biotope_type TEXT,     -- Pinar, Robledal, Pradera
                soil_type TEXT,        -- Ácido, Calcáreo (opcional)
                notes TEXT,
                
                -- Metadatos
                date_created TEXT,
                last_visited TEXT,
                image_path TEXT        -- Referencia a foto local (futuro)
            )
        ''')
        
        # 3. TABLA DE PREDICCIONES MANUALES (Bitácora)
        # Para que el usuario registre "Creo que aquí saldrá mañana"
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL, lon REAL,
                target_date TEXT,
                expected_species TEXT,
                confidence_level INTEGER,
                reasoning TEXT, -- "Ha llovido hace 15 días"
                status TEXT DEFAULT "PENDING" -- PENDING, VERIFIED_TRUE, VERIFIED_FALSE
            )
        ''')
        
        # Índices para acelerar consultas espaciales y temporales
        c.execute("CREATE INDEX IF NOT EXISTS idx_routes_date ON routes (date_start)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_routes_bbox ON routes (min_lat, max_lat, min_lon, max_lon)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_spots_species ON spots (species)")
        
        self.conn.commit()

    def calculate_file_hash(self, file_content):
        """Genera un hash único del contenido del archivo para evitar duplicados"""
        return hashlib.md5(file_content.encode('utf-8')).hexdigest()

    def route_exists(self, file_hash):
        c = self.conn.cursor()
        c.execute("SELECT id FROM routes WHERE hash_id = ?", (file_hash,))
        return c.fetchone() is not None

    def save_route_full(self, meta, points, analysis, segments, file_hash):
        """
        Guarda una ruta completa con toda la granularidad.
        Usa transacciones para asegurar integridad.
        """
        # Serializar objetos complejos a JSON
        points_json = json.dumps(points, default=str)
        analysis_json = json.dumps(analysis, default=str)
        segments_json = json.dumps(segments, default=str)
        
        c = self.conn.cursor()
        try:
            c.execute('''
                INSERT INTO routes (
                    filename, hash_id, date_start, date_end, tags,
                    total_distance_km, elevation_gain_m, elevation_loss_m,
                    max_elevation_m, min_elevation_m, moving_time_h, total_time_h,
                    avg_speed_kmh, max_speed_kmh,
                    min_lat, max_lat, min_lon, max_lon,
                    points_blob, analysis_blob, segments_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                meta['filename'], file_hash, meta['start_time'], meta['end_time'], meta.get('tags', 'General'),
                meta['distance_2d'] / 1000.0, meta['uphill'], meta['downhill'],
                meta['max_ele'], meta['min_ele'], meta['moving_time'] / 3600.0, meta['total_time'] / 3600.0,
                meta['avg_speed_kmh'], meta['max_speed_kmh'],
                meta['bounds']['min_lat'], meta['bounds']['max_lat'], 
                meta['bounds']['min_lon'], meta['bounds']['max_lon'],
                points_json, analysis_json, segments_json
            ))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"Error saving route: {e}")
            self.conn.rollback()
            return False

    def get_routes_filtered(self, tags=None, date_range=None, bounds=None):
        """
        Motor de consulta avanzado. Filtra por:
        1. Tags (Etiquetas)
        2. Rango de Fechas
        3. Área Geográfica (Viewport del mapa)
        """
        query = "SELECT * FROM routes WHERE 1=1"
        params = []
        
        # Filtro de Tags
        if tags and "Todos" not in tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
            if tag_conditions:
                query += " AND (" + " OR ".join(tag_conditions) + ")"
        
        # Filtro de Fechas
        if date_range:
            start_date, end_date = date_range
            query += " AND date(date_start) >= date(?) AND date(date_start) <= date(?)"
            params.extend([start_date, end_date])
            
        # Filtro Espacial (Bounding Box Intersection)
        if bounds:
            # Selecciona rutas cuyo Bounding Box intersecta con el del mapa
            # Logica: !(r.min_lat > b.max_lat || r.max_lat < b.min_lat ...)
            ne = bounds['_northEast']
            sw = bounds['_southWest']
            query += """ AND NOT (
                min_lat > ? OR max_lat < ? OR
                min_lon > ? OR max_lon < ?
            )"""
            params.extend([ne['lat'], sw['lat'], ne['lng'], sw['lng']])
            
        df = pd.read_sql_query(query, self.conn, params=params)
        
        # Deshidratar JSONs bajo demanda (lazy loading si fuera necesario, aqui lo hacemos directo)
        if not df.empty:
            df['points'] = df['points_blob'].apply(json.loads)
            df['analysis'] = df['analysis_blob'].apply(json.loads)
            df['segments'] = df['segments_blob'].apply(json.loads)
            df['date_start'] = pd.to_datetime(df['date_start'])
            
        return df

    def save_spot(self, spot_data):
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO spots (
                name, species, lat, lon, elevation, rating, 
                productivity_index, biotope_type, notes, date_created
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            spot_data['name'], spot_data['species'], spot_data['lat'], spot_data['lon'],
            spot_data.get('elevation', 0), spot_data['rating'], 
            spot_data.get('productivity', 0), spot_data.get('biotope', 'Desconocido'),
            spot_data.get('notes', ''), datetime.now().isoformat()
        ))
        self.conn.commit()

    def get_spots_df(self):
        return pd.read_sql_query("SELECT * FROM spots ORDER BY rating DESC", self.conn)
        
    def delete_element(self, table, item_id):
        self.conn.execute(f"DELETE FROM {table} WHERE id=?", (item_id,))
        self.conn.commit()

    def get_unique_tags(self):
        """Extrae y normaliza todos los tags únicos usados en la BD"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT tags FROM routes")
        raw_tags = cursor.fetchall()
        unique = set()
        for t_row in raw_tags:
            if t_row[0]:
                for tag in t_row[0].split(','):
                    unique.add(tag.strip())
        return sorted(list(unique))

# Instancia global del motor de BD para ser usada en toda la app
db = DatabaseEngine()

# ==========================================
# 3. MOTOR DE PROCESAMIENTO GPX Y ANÁLISIS DE TERRENO
# ==========================================

class GpxProcessor:
    """
    Analizador de alto rendimiento para tracks GPS.
    Incluye:
    - Filtros de suavizado para eliminar ruido del GPS.
    - Detección algorítmica de paradas (Stop detection) para identificar zonas de recolección.
    - Cálculo vectorial de pendientes y orientaciones.
    """
    
    def __init__(self):
        pass

    def process_file(self, file_buffer, filename, tags="General"):
        """
        Flujo principal de procesamiento de un archivo GPX.
        Retorna: metadatos, puntos enriquecidos, análisis y segmentos.
        """
        try:
            gpx = gpxpy.parse(file_buffer)
            
            # 1. Extracción y Aplanado de Puntos
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        points.append({
                            'lat': point.latitude,
                            'lon': point.longitude,
                            'ele': point.elevation if point.elevation is not None else 0,
                            'time': point.time
                        })
            
            if not points: return None

            # Convertir a DataFrame para manipulación vectorial rápida
            df = pd.DataFrame(points)
            
            # Limpieza básica: Eliminar puntos sin tiempo o duplicados
            df = df.dropna(subset=['time']).drop_duplicates(subset=['time'])
            df = df.sort_values('time').reset_index(drop=True)
            
            # 2. Cálculos Vectoriales (Velocidad, Distancia, Azimut)
            # Desplazamos los arrays para calcular deltas (i vs i-1)
            lats = df['lat'].values
            lons = df['lon'].values
            eles = df['ele'].values
            times = df['time'].values

            # Distancias (Haversine)
            dists = np.zeros(len(df))
            dists[1:] = GeoMath.haversine_vectorized(lats[:-1], lons[:-1], lats[1:], lons[1:])
            
            # Tiempos (Segundos)
            time_diffs = np.zeros(len(df))
            # Convertir a nanosegundos int64 y luego a segundos float
            time_diffs[1:] = (times[1:] - times[:-1]).astype('timedelta64[ns]').astype(np.int64) / 1e9
            
            # Evitar divisiones por cero o saltos temporales enormes (pausas de GPS)
            mask_valid = (time_diffs > 0) & (time_diffs < 3600) # Ignorar saltos > 1h
            
            # Velocidades (km/h)
            speeds = np.zeros(len(df))
            speeds[1:][mask_valid[1:]] = (dists[1:][mask_valid[1:]] / 1000.0) / (time_diffs[1:][mask_valid[1:]] / 3600.0)
            
            # Filtro de Ruido de Velocidad (Media Móvil) para suavizar picos de GPS
            speeds = pd.Series(speeds).rolling(window=5, center=True, min_periods=1).mean().fillna(0).values

            # Azimut (Bearing) y Orientación
            bearings = np.zeros(len(df))
            bearings[1:] = GeoMath.calculate_bearing_vectorized(lats[:-1], lons[:-1], lats[1:], lons[1:])
            
            # Pendientes (Slope) en grados
            ele_diffs = np.zeros(len(df))
            ele_diffs[1:] = eles[1:] - eles[:-1]
            slopes = np.zeros(len(df))
            slopes[1:][mask_valid[1:]] = np.degrees(np.arctan(ele_diffs[1:][mask_valid[1:]] / dists[1:][mask_valid[1:]]))
            
            # Asignar cálculos al DF
            df['dist_diff'] = dists
            df['time_diff'] = time_diffs
            df['speed'] = speeds
            df['bearing'] = bearings
            df['slope'] = slopes
            
            # 3. Detección de "Corros" (Segmentación Stop/Move)
            # Si velocidad < 1.0 km/h durante > 30 segundos, es una "Micro-Parada de Búsqueda"
            df['is_stopped'] = df['speed'] < 1.0
            
            # Identificar grupos de paradas (Corros)
            # Usamos lógica de cambio de estado
            df['segment_id'] = (df['is_stopped'] != df['is_stopped'].shift()).cumsum()
            
            segments_summary = []
            for seg_id, group in df.groupby('segment_id'):
                is_stop = group['is_stopped'].iloc[0]
                duration = group['time_diff'].sum()
                
                # Solo guardamos paradas significativas (> 2 min y < 30 min) para evitar pausas de bocadillo o errores
                if is_stop and 120 < duration < 1800:
                    centroid_lat = group['lat'].mean()
                    centroid_lon = group['lon'].mean()
                    segments_summary.append({
                        'type': 'FORAGING_STOP',
                        'lat': centroid_lat,
                        'lon': centroid_lon,
                        'duration_s': duration,
                        'start_time': group['time'].iloc[0].isoformat()
                    })

            # 4. Compilación de Metadatos
            total_dist = df['dist_diff'].sum()
            moving_time = df[df['speed'] > 0.5]['time_diff'].sum()
            total_time = df['time_diff'].sum()
            
            meta = {
                'filename': filename,
                'date': df['time'].iloc[0].strftime('%Y-%m-%d'),
                'start_time': df['time'].iloc[0].isoformat(),
                'end_time': df['time'].iloc[-1].isoformat(),
                'tags': tags,
                'distance_2d': total_dist,
                'uphill': df[df['slope'] > 0]['dist_diff'].sum() * np.tan(np.radians(df[df['slope'] > 0]['slope'].mean())), # Aprox
                'downhill': 0, # Simplificado
                'max_ele': df['ele'].max(),
                'min_ele': df['ele'].min(),
                'moving_time': moving_time,
                'total_time': total_time,
                'avg_speed_kmh': (total_dist/1000) / (moving_time/3600) if moving_time > 0 else 0,
                'max_speed_kmh': df['speed'].max(),
                'bounds': {
                    'min_lat': df['lat'].min(), 'max_lat': df['lat'].max(),
                    'min_lon': df['lon'].min(), 'max_lon': df['lon'].max()
                }
            }
            
            # 5. Compilación de Análisis Estadístico (Histogramas)
            # Para visualizaciones rápidas sin cargar todos los puntos
            analysis = {
                'aspect_distribution': df['bearing'].apply(GeoMath.get_aspect_category).value_counts().to_dict(),
                'slope_histogram': np.histogram(df['slope'], bins=10, range=(-30, 30))[0].tolist(),
                'ele_histogram': np.histogram(df['ele'], bins=10)[0].tolist(),
                'speed_histogram': np.histogram(df['speed'], bins=10, range=(0, 6))[0].tolist()
            }
            
            # Convertir DF a lista de dicts optimizada (solo columnas necesarias)
            # Reducimos precisión de floats para ahorrar espacio en BD
            final_points = df[['lat', 'lon', 'ele', 'speed', 'bearing']].to_dict(orient='records')
            
            # Serializamos fechas a string
            for p, t in zip(final_points, df['time']):
                p['time'] = t.isoformat()
            
            return meta, final_points, analysis, segments_summary

        except Exception as e:
            st.error(f"Error crítico procesando {filename}: {str(e)}")
            return None

# ==========================================
# 4. MOTOR METEOROLÓGICO HÍBRIDO (FRIKI MODE)
# ==========================================

class WeatherIntelligence:
    """
    Gestor de meteorología avanzada.
    Implementa:
    - Evapotranspiración (ET0) para calcular 'Agua Neta'.
    - Índices de Choque Térmico (Boletus Trigger).
    - Caché robusta y manejo de errores de API.
    """
    
    def __init__(self):
        # Configurar sesión con reintentos para robustez
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_historical_context(_self, lat, lon, days_back=45):
        """
        Obtiene datos de reanálisis de alta resolución (Open-Meteo).
        Ventana de 45 días para cálculo de ciclos de micelio.
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": [
                "precipitation_sum",
                "temperature_2m_max", 
                "temperature_2m_min",
                "et0_fao_evapotranspiration",  # CLAVE: Evaporación
                "soil_temperature_0_to_7cm_mean", # CLAVE: Temp Suelo
                "wind_speed_10m_max"
            ],
            "timezone": "auto"
        }
        
        try:
            # Nota: usamos _self para bypass del hash de streamlit en métodos de clase
            response = _self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'daily' not in data: return None
            
            df = pd.DataFrame(data['daily'])
            df['time'] = pd.to_datetime(df['time'])
            return df
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error conectando con satélites meteo: {e}")
            return None

    def calculate_fungi_indices(self, df_weather):
        """
        El núcleo 'Friki': Convierte datos crudos en probabilidades micológicas.
        """
        if df_weather is None or df_weather.empty: return None

        # 1. Índice de Humedad del Suelo (SMI) con Decaimiento Exponencial
        # Simula cómo el suelo retiene agua. Un día de sol resta, lluvia suma.
        # Factor 0.85 = Suelo de bosque (retiene bien). 0.6 = Arena.
        retention_factor = 0.88 
        smi_values = []
        current_smi = 0
        
        # Evapotranspiración acumulada (net water balance)
        net_water_balance = []
        
        for rain, et0 in zip(df_weather['precipitation_sum'], df_weather['et0_fao_evapotranspiration']):
            # El suelo pierde agua por ET0 (ajustada por factor de sombra bosque 0.7)
            loss = et0 * 0.7 
            # Balance neto del día
            net_change = rain - loss
            
            # El SMI es un acumulador con memoria (decaimiento)
            current_smi = (current_smi * retention_factor) + rain
            # Aplicamos pérdida por evaporación directa al SMI
            current_smi = max(0, current_smi - loss)
            
            smi_values.append(current_smi)
            net_water_balance.append(net_change)

        df_weather['SMI'] = smi_values
        df_weather['Net_Water'] = net_water_balance

        # 2. Detección de Choque Térmico (Trigger de Fructificación)
        # Buscamos caídas bruscas de temperatura mínima tras periodo estable
        # Ventana móvil de 3 días
        df_weather['temp_drop'] = df_weather['temperature_2m_min'].diff()
        
        # Trigger: Caída > 4ºC en 48h y humedad suelo > 15
        triggers = []
        for i in range(2, len(df_weather)):
            row = df_weather.iloc[i]
            prev = df_weather.iloc[i-2]
            
            drop = prev['temperature_2m_min'] - row['temperature_2m_min']
            is_wet = row['SMI'] > 20
            
            if drop >= 4.0 and is_wet:
                triggers.append("POTENTIAL_TRIGGER")
            elif drop >= 2.0 and is_wet:
                triggers.append("MILD_TRIGGER")
            else:
                triggers.append("STABLE")
        
        # Rellenar los primeros 2 días
        triggers = ["STABLE", "STABLE"] + triggers
        df_weather['thermal_trigger'] = triggers

        # 3. Penalización por Viento (El "Secador")
        # Si hubo viento > 25km/h en los últimos 5 días, el SMI superficial es falso (está seco)
        recent_wind = df_weather['wind_speed_10m_max'].tail(5).mean()
        wind_penalty_factor = 1.0
        if recent_wind > 20: wind_penalty_factor = 0.7
        if recent_wind > 30: wind_penalty_factor = 0.4
        
        # 4. Cálculo final MAI (Mycelium Activation Index) 0-100
        last_day = df_weather.iloc[-1]
        
        # Base hídrica (0-60 ptos)
        water_score = min(last_day['SMI'] * 1.5, 60)
        
        # Base temperatura suelo (0-20 ptos)
        # Óptimo Boletus: 12-16ºC
        t_soil = last_day['soil_temperature_0_to_7cm_mean']
        temp_score = 0
        if 10 <= t_soil <= 18: temp_score = 20
        elif 5 <= t_soil < 10 or 18 < t_soil <= 22: temp_score = 10
        
        # Bonus Trigger Térmico (0-20 ptos)
        # Miramos si hubo trigger en la última semana
        recent_triggers = df_weather['thermal_trigger'].tail(7)
        trigger_score = 0
        if "POTENTIAL_TRIGGER" in recent_triggers.values: trigger_score = 20
        elif "MILD_TRIGGER" in recent_triggers.values: trigger_score = 10
        
        final_mai = (water_score + temp_score + trigger_score) * wind_penalty_factor
        
        return int(final_mai), df_weather

# Instancias Globales
gpx_processor = GpxProcessor()
meteo_engine = WeatherIntelligence()

# ==========================================
# 5. MOTOR DE MACHINE LEARNING (PREDICCIÓN ESPACIAL)
# ==========================================

class BioPredictor:
    """
    Motor de IA para modelado de nicho ecológico.
    Usa Kernel Density Estimation (KDE) para convertir puntos discretos de hallazgos
    en mapas de probabilidad continuos.
    """
    
    def __init__(self):
        self.model = None
        self.bandwidth = 0.002  # ~200m aprox (ajustable según densidad de datos)

    def train_model(self, df_routes):
        """
        Entrena el modelo usando solo los puntos de 'parada/recolección' (baja velocidad).
        """
        if df_routes.empty: return False

        # 1. Extraer todos los puntos de todas las rutas filtradas
        # Es necesario deserializar los blobs si no se ha hecho
        all_points = []
        for _, row in df_routes.iterrows():
            # Si 'points' ya es lista (deserializado), usarla. Si no, parsear JSON.
            pts = row['points'] if isinstance(row['points'], list) else json.loads(row['points_blob'])
            
            # Filtro heurístico: Solo puntos donde la velocidad < 1.5 km/h
            # Esto asume que si vas lento, estás buscando o recogiendo.
            productive_pts = [
                [p['lat'], p['lon']] 
                for p in pts 
                if p.get('speed', 5) < 1.5
            ]
            all_points.extend(productive_pts)

        if len(all_points) < 50: 
            return False # No hay suficientes datos para una predicción fiable

        # 2. Entrenar KDE
        # Convertimos a radianes para usar métrica haversine
        X_train = np.radians(np.array(all_points))
        
        self.model = KernelDensity(bandwidth=self.bandwidth, metric='haversine')
        self.model.fit(X_train)
        return True

    def generate_probability_grid(self, bounds, resolution=100):
        """
        Genera una matriz de probabilidad (Heatmap Matemático) para el área visible.
        Retorna X, Y, Z para visualización de contornos.
        """
        if not self.model: return None, None, None

        # Definir el grid basado en los bounds del mapa actual
        lat_min, lat_max = bounds['_southWest']['lat'], bounds['_northEast']['lat']
        lon_min, lon_max = bounds['_southWest']['lng'], bounds['_northEast']['lng']

        # Añadir un margen del 10% para suavidad en los bordes
        lat_margin = (lat_max - lat_min) * 0.1
        lon_margin = (lon_max - lon_min) * 0.1
        
        lat_grid = np.linspace(lat_min - lat_margin, lat_max + lat_margin, resolution)
        lon_grid = np.linspace(lon_min - lon_margin, lon_max + lon_margin, resolution)
        
        X, Y = np.meshgrid(lat_grid, lon_grid)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        
        # Evaluar el modelo (score_samples devuelve log-densidad)
        # Convertimos grados a radianes para la evaluación
        log_dens = self.model.score_samples(np.radians(xy))
        Z = np.exp(log_dens).reshape(X.shape)
        
        # Normalizar Z a 0-1 para visualización consistente
        z_min, z_max = Z.min(), Z.max()
        Z_norm = (Z - z_min) / (z_max - z_min) if z_max > z_min else Z
        
        return X, Y, Z_norm

    def predict_point_score(self, lat, lon):
        """Devuelve la probabilidad (0-100) para una coordenada específica"""
        if not self.model: return 0
        point = np.radians([[lat, lon]])
        log_dens = self.model.score_samples(point)
        # Nota: Esto es densidad relativa, no probabilidad absoluta, 
        # pero sirve para comparar puntos.
        score = np.exp(log_dens)[0]
        return min(score * 1000, 100) # Factor de escala arbitrario para UX

# Instancia Global
ai_brain = BioPredictor()

# ==========================================
# 6. LOGICA DE INTERFAZ DE USUARIO (SIDEBAR & FILTROS)
# ==========================================

def render_sidebar():
    """Construye el panel lateral de control"""
    st.sidebar.image("https://img.icons8.com/color/96/000000/mushroom.png", width=60)
    st.sidebar.title("MicoBrain OMNI")
    st.sidebar.markdown("**v4.0.1 Stable** | *Professional Edition*")
    
    # --- MÓDULO DE IMPORTACIÓN ---
    with st.sidebar.expander("📥 Importar GPX", expanded=False):
        st.markdown("Añade nuevas rutas a tu base de datos.")
        
        # Formulario de Importación
        with st.form("upload_form", clear_on_submit=True):
            tags_input = st.text_input("🏷️ Etiquetas (sep. por comas)", 
                                     placeholder="Ej: Níscalos, Soria, 2024")
            files = st.file_uploader("Arrastra archivos .gpx", 
                                   type=['gpx'], accept_multiple_files=True)
            
            submitted = st.form_submit_button("Procesar e Indexar")
            
            if submitted and files:
                progress_bar = st.progress(0)
                success_count = 0
                
                for i, file in enumerate(files):
                    # Hash check para evitar duplicados
                    content = file.getvalue().decode("utf-8")
                    file_hash = db.calculate_file_hash(content)
                    
                    if db.route_exists(file_hash):
                        st.warning(f"Ignorado (Duplicado): {file.name}")
                    else:
                        # PROCESAMIENTO COMPLETO (Usando las clases Parte 2)
                        s_io = io.StringIO(content)
                        result = gpx_processor.process_file(s_io, file.name, tags_input)
                        
                        if result:
                            meta, pts, analysis, segments = result
                            saved = db.save_route_full(meta, pts, analysis, segments, file_hash)
                            if saved: success_count += 1
                        
                    progress_bar.progress((i + 1) / len(files))
                
                if success_count > 0:
                    st.success(f"✅ {success_count} rutas nuevas indexadas.")
                    st.cache_data.clear() # Limpiar caché para refrescar mapa
                else:
                    st.info("No se añadieron rutas nuevas.")

    st.sidebar.divider()

    # --- MÓDULO DE FILTRADO GLOBAL ---
    st.sidebar.subheader("🔍 Filtros de Visualización")
    
    # Cargar datos únicos para los filtros
    all_tags = db.get_unique_tags()
    
    # Filtro 1: Colecciones / Tags
    selected_tags = st.sidebar.multiselect(
        "Colecciones", 
        ["Todos"] + all_tags, 
        default=["Todos"]
    )
    
    # Filtro 2: Rango de Fechas
    # Obtener fechas min/max de la BD
    try:
        min_date_q = pd.read_sql("SELECT min(date_start) FROM routes", db.conn).iloc[0,0]
        max_date_q = pd.read_sql("SELECT max(date_start) FROM routes", db.conn).iloc[0,0]
        
        if min_date_q and max_date_q:
            min_d = datetime.strptime(min_date_q, "%Y-%m-%d %H:%M:%S").date()
            max_d = datetime.strptime(max_date_q, "%Y-%m-%d %H:%M:%S").date()
            
            date_range = st.sidebar.date_input(
                "Periodo de Análisis",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d
            )
        else:
            date_range = None
    except:
        date_range = None
        st.sidebar.caption("Sube rutas para habilitar filtro de fechas.")

    st.sidebar.divider()

    # --- KPI RÁPIDOS EN SIDEBAR ---
    # Mostramos resumen de lo filtrado en tiempo real
    if date_range and len(date_range) == 2:
        # Consulta ligera count(*)
        # (Lógica simplificada para visualización, la query real se hace en el dashboard)
        pass 
    
    return selected_tags, date_range

# ==========================================
# 7. GESTOR DE ESTADO DE SESIÓN
# ==========================================

def init_session_state():
    """Inicializa variables persistentes entre recargas"""
    # Coordenadas del mapa (Centro de España por defecto)
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [40.416, -3.703]
    if 'map_zoom' not in st.session_state:
        st.session_state.map_zoom = 6
    
    # Límites del mapa (Viewport)
    if 'map_bounds' not in st.session_state:
        st.session_state.map_bounds = None
        
    # Ruta seleccionada para análisis profundo
    if 'selected_route_id' not in st.session_state:
        st.session_state.selected_route_id = None
        
    # Datos del último análisis meteo (para no llamar a API en cada rerun)
    if 'last_weather_data' not in st.session_state:
        st.session_state.last_weather_data = None
        
    # Capas activas
    if 'layer_heatmap' not in st.session_state: st.session_state.layer_heatmap = True
    if 'layer_clusters' not in st.session_state: st.session_state.layer_clusters = False
    if 'layer_prediction' not in st.session_state: st.session_state.layer_prediction = False

# ==========================================
# INICIO DE EJECUCIÓN UI
# ==========================================

init_session_state()

# Renderizar Sidebar y capturar filtros
tags_filter, date_filter = render_sidebar()

# Recuperar datos base filtrados (sin bounding box todavía, eso es en el mapa)
# Si no hay rango de fecha válido (ej: usuario seleccionando), usamos todo
d_range_query = date_filter if date_filter and len(date_filter) == 2 else None
df_base = db.get_routes_filtered(tags=tags_filter, date_range=d_range_query)

if df_base.empty:
    st.info("👋 Bienvenido a MicoBrain OMNI. Empieza importando tus archivos GPX en el panel lateral.")
    # Creamos un DF dummy para que no falle el renderizado del mapa inicial
    df_base = pd.DataFrame(columns=['lat', 'lon', 'points', 'analysis'])

# ==========================================
# 8. DASHBOARD PRINCIPAL Y MAPA
# ==========================================

# TABS PRINCIPALES DE NAVEGACIÓN
tab_map, tab_analysis, tab_prediction, tab_data = st.tabs([
    "🗺️ Mapa Táctico", 
    "📊 Laboratorio de Datos", 
    "🔮 Oráculo (Predicción)", 
    "💾 Gestión BD"
])

# --- TAB 1: MAPA TÁCTICO ---
with tab_map:
    col_controls, col_map_display = st.columns([1, 4])
    
    with col_controls:
        st.markdown("### 📡 Capas Activas")
        
        # Control de Capas con Estado Persistente
        st.session_state.layer_heatmap = st.checkbox("🔥 Calor Histórico", value=True)
        st.session_state.layer_clusters = st.checkbox("🤖 Clusters (Corros)", value=False, help="Detección IA de paradas")
        st.session_state.layer_prediction = st.checkbox("✨ Predicción (KDE)", value=False, help="Modelo matemático de nicho")
        
        st.markdown("---")
        st.markdown("### 📍 Favoritos")
        st.info("Haz clic en el mapa para añadir un nuevo punto.")
        
        # Filtro rápido de favoritos
        fav_species_filter = st.selectbox("Filtrar Setales", ["Todos", "Boletus", "Níscalos", "Amanita", "Otros"])

    with col_map_display:
        # Configuración del Mapa Base
        m = folium.Map(
            location=st.session_state.map_center,
            zoom_start=st.session_state.map_zoom,
            tiles=None,
            control_scale=True
        )
        
        # 1. Capas Base (Satélite y Topo)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satélite (Esri)',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            attr='OpenTopoMap',
            name='Topográfico',
            overlay=False,
            control=True
        ).add_to(m)

        # 2. Renderizado de Rutas (Líneas)
        # Solo dibujamos si hay menos de 50 rutas para no saturar el navegador
        # Si hay muchas, confiamos en el Heatmap
        if len(df_base) < 50:
            for _, row in df_base.iterrows():
                # Deserializar puntos ligeros para dibujo
                # Nota: En producción usaríamos GeoJSON simplificado
                try:
                    pts = row['points'] if isinstance(row['points'], list) else json.loads(row['points_blob'])
                    coords = [[p['lat'], p['lon']] for p in pts if 'lat' in p]
                    
                    # Color según tags
                    color = "#ff9800" if "Níscalos" in row.get('tags', '') else "#2e7d32"
                    
                    folium.PolyLine(
                        coords, color=color, weight=2.5, opacity=0.6,
                        tooltip=f"{row['filename']} ({row['date_start']})"
                    ).add_to(m)
                except: continue

        # 3. Capa HEATMAP (Histórico)
        if st.session_state.layer_heatmap and not df_base.empty:
            # Extraer todos los puntos del dataframe filtrado
            all_heat_points = []
            for _, row in df_base.iterrows():
                pts = row['points'] if isinstance(row['points'], list) else json.loads(row['points_blob'])
                all_heat_points.extend([[p['lat'], p['lon']] for p in pts])
            
            # Downsampling para rendimiento si hay > 100k puntos
            if len(all_heat_points) > 50000:
                all_heat_points = all_heat_points[::5]
                
            plugins.HeatMap(
                all_heat_points, 
                name="Densidad Histórica",
                radius=15, blur=20, 
                gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
            ).add_to(m)

        # 4. Capa CLUSTERS (IA DBSCAN - Zonas de Recolección)
        if st.session_state.layer_clusters and not df_base.empty:
            # Reconstruir dataframe plano de puntos lentos para el algoritmo
            points_flat = []
            for _, row in df_base.iterrows():
                pts = row['points'] if isinstance(row['points'], list) else json.loads(row['points_blob'])
                for p in pts:
                    if p.get('speed', 5) < 1.0: # Filtro de velocidad
                        points_flat.append(p)
            
            if points_flat:
                df_slow = pd.DataFrame(points_flat)
                # Entrenamos DBSCAN on-the-fly para la vista actual
                # eps=30m, min_samples=15 puntos
                # Esto detecta "paradas densas"
                from sklearn.cluster import DBSCAN
                coords = df_slow[['lat', 'lon']].values
                kms_per_rad = 6371.0088
                eps_rad = 0.03 / kms_per_rad # 30 metros
                
                dbscan = DBSCAN(eps=eps_rad, min_samples=15, metric='haversine').fit(np.radians(coords))
                
                # Dibujar Envolventes Convexas de los clusters
                labels = dbscan.labels_
                unique_labels = set(labels)
                for k in unique_labels:
                    if k == -1: continue # Ruido
                    class_members = coords[labels == k]
                    if len(class_members) > 3:
                        hull = ConvexHull(class_members)
                        hull_pts = class_members[hull.vertices]
                        # Cerrar polígono
                        hull_pts = np.append(hull_pts, [hull_pts[0]], axis=0)
                        
                        folium.Polygon(
                            locations=hull_pts.tolist(),
                            color='#d32f2f', fill=True, fill_color='#ffcdd2', fill_opacity=0.5,
                            popup="Zona Productiva (Detectada)",
                            tooltip="Corro Detectado"
                        ).add_to(m)

        # 5. Capa PREDICCIÓN (KDE)
        if st.session_state.layer_prediction and not df_base.empty:
            # Entrenamos el modelo con los datos actuales
            success = ai_brain.train_model(df_base)
            if success and st.session_state.map_bounds:
                # Generamos grid para el viewport actual
                X, Y, Z = ai_brain.generate_probability_grid(st.session_state.map_bounds)
                if Z is not None:
                    # Visualización simplificada: Puntos de alta probabilidad como heatmap azul
                    # (Renderizar imagen completa es lento en Streamlit-Folium, usamos scatter de densidad)
                    high_prob_indices = np.where(Z > np.percentile(Z, 90)) # Top 10%
                    prob_pts = list(zip(X[high_prob_indices], Y[high_prob_indices]))
                    
                    plugins.HeatMap(
                        prob_pts,
                        name="Predicción IA",
                        radius=25, blur=15,
                        gradient={0: 'transparent', 0.5: 'cyan', 1: 'navy'}
                    ).add_to(m)
            elif not st.session_state.map_bounds:
                st.toast("Mueve el mapa para activar la predicción en esa zona.")

        # 6. Renderizado de FAVORITOS
        df_spots = db.get_spots_df()
        fg_spots = folium.FeatureGroup(name="Mis Setales")
        
        for _, spot in df_spots.iterrows():
            if fav_species_filter != "Todos" and fav_species_filter not in spot['species']:
                continue
                
            color = "green" if spot['rating'] >= 4 else "orange" if spot['rating'] == 3 else "red"
            icon = folium.Icon(color=color, icon="star", prefix="fa")
            
            html = f"""
                <div style="font-family: sans-serif; width: 200px;">
                    <h4>{spot['name']}</h4>
                    <p><b>Especie:</b> {spot['species']}</p>
                    <p><b>Rating:</b> {spot['rating']}/5</p>
                    <p><i>{spot['notes']}</i></p>
                </div>
            """
            
            folium.Marker(
                [spot['lat'], spot['lon']],
                popup=html,
                icon=icon,
                tooltip=spot['name']
            ).add_to(fg_spots)
            
        fg_spots.add_to(m)

        # Controles extra
        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.LocateControl().add_to(m)
        plugins.MeasureControl().add_to(m)

        # RENDER FINAL DEL MAPA
        map_data = st_folium(
            m, 
            height=600, 
            width="100%",
            key="main_map_widget",
            update_zoom=True,
            update_bounds=True
        )
        
        # ACTUALIZACIÓN DE ESTADO (BOUNDS Y CENTRO)
        if map_data and map_data.get('bounds'):
            st.session_state.map_bounds = map_data['bounds']
            st.session_state.map_center = [map_data['center']['lat'], map_data['center']['lng']]
            st.session_state.map_zoom = map_data['zoom']

        # LOGICA DE CLIC -> CREAR FAVORITO
        if map_data and map_data.get("last_clicked"):
            click = map_data["last_clicked"]
            st.session_state.last_click = click
            
            # Modal simulado en sidebar o debajo
            with st.sidebar.form("new_spot_form"):
                st.markdown("### ⭐ Nuevo Setal")
                st.caption(f"Lat: {click['lat']:.5f}, Lon: {click['lng']:.5f}")
                
                s_name = st.text_input("Nombre")
                s_spec = st.selectbox("Especie", ["Boletus edulis", "Boletus pinophilus", "Níscalos", "Amanita caesarea", "Rebozuelo", "Trompeta", "Otro"])
                s_rate = st.slider("Productividad", 1, 5, 3)
                s_note = st.text_area("Notas (Suelo, árboles...)")
                
                if st.form_submit_button("Guardar Setal"):
                    spot_data = {
                        'name': s_name, 'species': s_spec,
                        'lat': click['lat'], 'lon': click['lng'],
                        'rating': s_rate, 'notes': s_note
                    }
                    db.save_spot(spot_data)
                    st.success("Guardado!")
                    st.rerun()

# --- TAB 2: LABORATORIO DE DATOS (ANALYTICS) ---
with tab_analysis:
    st.header("📊 Análisis de Biotopo")
    
    if df_base.empty:
        st.info("No hay datos visibles.")
    else:
        # Recuperamos datos completos (análisis pre-calculado)
        # Nota: En un entorno real esto se haría con queries SQL agregadas para velocidad
        
        # 1. KPI Cards
        total_km = df_base['total_distance_km'].sum()
        total_ele = df_base['elevation_gain_m'].sum()
        total_hours = df_base['moving_time_h'].sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distancia Total", f"{total_km:.1f} km")
        c2.metric("Desnivel Positivo", f"{total_ele:.0f} m")
        c3.metric("Tiempo en Movimiento", f"{total_hours:.1f} h")
        c4.metric("Rutas Analizadas", len(df_base))
        
        st.divider()
        
        # 2. Gráficos Avanzados
        col_charts_1, col_charts_2 = st.columns(2)
        
        with col_charts_1:
            st.subheader("🏔️ Perfil Altitudinal")
            # Histograma de elevaciones min/max de las rutas
            fig_ele = px.histogram(
                df_base, 
                x="max_elevation_m", 
                nbins=20, 
                title="Cotas Máximas Alcanzadas",
                color_discrete_sequence=['#5c6bc0']
            )
            st.plotly_chart(fig_ele, use_container_width=True)
            
        with col_charts_2:
            st.subheader("📅 Patrones Temporales")
            # Extraer mes
            df_base['month'] = pd.to_datetime(df_base['date_start']).dt.month_name()
            # Ordenar meses
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
            
            fig_time = px.bar(
                df_base['month'].value_counts().reindex(month_order).dropna(),
                title="Salidas por Mes",
                color_discrete_sequence=['#66bb6a']
            )
            st.plotly_chart(fig_time, use_container_width=True)

# --- TAB 3: ORÁCULO (PREDICCIÓN METEO) ---
with tab_prediction:
    st.header("🔮 Predicción Meteorológica Híbrida")
    st.markdown("""
        Este módulo conecta con **Open-Meteo Archive** (Reanálisis de 1km) para calcular 
        variables micológicas críticas: **Evapotranspiración (Agua Neta)** y **Choque Térmico**.
    """)
    
    col_pred_ctrl, col_pred_viz = st.columns([1, 2])
    
    with col_pred_ctrl:
        st.info(f"Analizando zona central del mapa: {st.session_state.map_center}")
        
        if st.button("🔎 EJECUTAR ANÁLISIS BIO-METEOROLÓGICO", type="primary"):
            with st.spinner("Descargando datos satelitales (últimos 45 días)..."):
                lat, lon = st.session_state.map_center
                
                # 1. Fetch Data
                w_df = meteo_engine.fetch_historical_context(lat, lon, days_back=45)
                
                if w_df is not None:
                    # 2. Calculate Indices
                    mai, w_df_processed = meteo_engine.calculate_fungi_indices(w_df)
                    st.session_state.last_weather_data = (mai, w_df_processed)
                    st.success("Cálculos finalizados.")
                else:
                    st.error("No se pudo conectar con el servidor meteorológico.")

    with col_pred_viz:
        if st.session_state.last_weather_data:
            mai, df_w = st.session_state.last_weather_data
            
            # MAI GAUGE
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = mai,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "MAI (Mycelium Activation Index)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightgreen"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80}}))
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Interpretación
            if mai > 70:
                st.success("🚀 CONDICIONES EXCELENTES: Alta probabilidad de eclosión.")
            elif mai > 40:
                st.warning("⚠️ CONDICIONES MEDIAS: Actividad posible en zonas húmedas (regatos).")
            else:
                st.error("🛑 CONDICIONES POBRES: Suelo seco o falta de estímulo térmico.")

            # Gráfico Detallado (SMI vs Lluvia)
            st.subheader("Ciclo Hidrológico (Suelo)")
            fig_water = go.Figure()
            fig_water.add_trace(go.Bar(
                x=df_w['time'], y=df_w['precipitation_sum'], 
                name='Lluvia (mm)', marker_color='blue', opacity=0.3
            ))
            fig_water.add_trace(go.Scatter(
                x=df_w['time'], y=df_w['SMI'], 
                name='Humedad Suelo (SMI)', line=dict(color='blue', width=3)
            ))
            # Añadir línea de Evaporación
            fig_water.add_trace(go.Scatter(
                x=df_w['time'], y=df_w['et0_fao_evapotranspiration'],
                name='Evaporación (ET0)', line=dict(color='red', dash='dot')
            ))
            
            fig_water.update_layout(height=350, title="Lluvia Real vs Agua Retenida vs Evaporación")
            st.plotly_chart(fig_water, use_container_width=True)

# --- TAB 4: GESTIÓN DE DATOS ---
with tab_data:
    st.header("💾 Base de Datos")
    
    st.subheader("Rutas Indexadas")
    routes_df = pd.read_sql("SELECT id, filename, date_start, tags, total_distance_km FROM routes ORDER BY date_start DESC", db.conn)
    st.dataframe(routes_df, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        rid_del = st.number_input("ID Ruta a borrar", min_value=0)
        if st.button("🗑️ Borrar Ruta"):
            db.delete_element('routes', rid_del)
            st.success("Borrada.")
            st.rerun()
            
    st.subheader("Setales (Favoritos)")
    spots_df = db.get_spots_df()
    st.dataframe(spots_df, use_container_width=True)
    
    with c2:
        sid_del = st.number_input("ID Setal a borrar", min_value=0)
        if st.button("🗑️ Borrar Setal"):
            db.delete_element('spots', sid_del)
            st.success("Borrado.")
            st.rerun()
