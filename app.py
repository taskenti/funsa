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
import logging

# ==========================================
# 0. CONFIGURACIÓN DEL SISTEMA Y ESTILOS CSS
# ==========================================

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        'About': "# MicoBrain OMNI v4.0.2\nLa herramienta definitiva para el micólogo moderno."
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
    def get_aspect_category(bearing):
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
# 2. UTILIDADES DE SERIALIZACIÓN
# ==========================================

def safe_json_dumps(data):
    """
    Serializa datos complejos sin perder tipos numéricos.
    """
    def serializer(obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if pd.isna(obj):
            return None
        raise TypeError(f"Tipo {type(obj)} no serializable")
    
    return json.dumps(data, default=serializer, ensure_ascii=False)

def calculate_route_fingerprint(points):
    """
    Genera hash solo de la geometría (lat, lon) ignorando metadatos.
    Más robusto que hashear todo el archivo.
    """
    # Tomar solo coordenadas cada 10 puntos para eficiencia
    sampled = points[::10] if len(points) > 100 else points
    
    coords_str = ''.join([
        f"{p['lat']:.5f}{p['lon']:.5f}" 
        for p in sampled 
        if 'lat' in p and 'lon' in p
    ])
    
    return hashlib.sha256(coords_str.encode()).hexdigest()

# ==========================================
# 3. GESTOR DE BASE DE DATOS (PERSISTENCIA AVANZADA)
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
        c.execute('''
            CREATE TABLE IF NOT EXISTS routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                hash_id TEXT UNIQUE,
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
                points_blob TEXT,
                analysis_blob TEXT,
                segments_blob TEXT
            )
        ''')
        
        # 2. TABLA DE SETALES (Favoritos/Waypoints)
        c.execute('''
            CREATE TABLE IF NOT EXISTS spots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                species TEXT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                elevation REAL,
                
                -- Valoración y Calidad
                rating INTEGER DEFAULT 3,
                productivity_index REAL,
                
                -- Contexto
                biotope_type TEXT,
                soil_type TEXT,
                notes TEXT,
                
                -- Metadatos
                date_created TEXT,
                last_visited TEXT,
                image_path TEXT
            )
        ''')
        
        # 3. TABLA DE PREDICCIONES MANUALES (Bitácora)
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL, lon REAL,
                target_date TEXT,
                expected_species TEXT,
                confidence_level INTEGER,
                reasoning TEXT,
                status TEXT DEFAULT "PENDING"
            )
        ''')
        
        # Índices para acelerar consultas espaciales y temporales
        c.execute("CREATE INDEX IF NOT EXISTS idx_routes_date ON routes (date_start)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_routes_bbox ON routes (min_lat, max_lat, min_lon, max_lon)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_spots_species ON spots (species)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_routes_tags ON routes (tags)")
        
        self.conn.commit()

    def route_exists(self, file_hash):
        c = self.conn.cursor()
        c.execute("SELECT id FROM routes WHERE hash_id = ?", (file_hash,))
        return c.fetchone() is not None

    def save_route_full(self, meta, points, analysis, segments, file_hash):
        """
        Guarda una ruta completa con toda la granularidad.
        Usa transacciones para asegurar integridad.
        """
        # Serializar objetos complejos a JSON con función segura
        points_json = safe_json_dumps(points)
        analysis_json = safe_json_dumps(analysis)
        segments_json = safe_json_dumps(segments)
        
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
                meta['filename'], file_hash, meta['start_time'], meta['end_time'], 
                meta.get('tags', 'General').lower().strip(),  # Normalizar tags
                meta['distance_2d'] / 1000.0, meta['uphill'], meta['downhill'],
                meta['max_ele'], meta['min_ele'], meta['moving_time'] / 3600.0, meta['total_time'] / 3600.0,
                meta['avg_speed_kmh'], meta['max_speed_kmh'],
                meta['bounds']['min_lat'], meta['bounds']['max_lat'], 
                meta['bounds']['min_lon'], meta['bounds']['max_lon'],
                points_json, analysis_json, segments_json
            ))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError as e:
            logger.warning(f"Ruta duplicada: {e}")
            return False
        except Exception as e:
            logger.error(f"Error guardando ruta: {e}")
            self.conn.rollback()
            return False

    def get_routes_metadata_only(self, tags=None, date_range=None, bounds=None):
        """
        Devuelve solo metadatos SIN deserializar puntos.
        Mucho más eficiente para listados.
        """
        query = """
            SELECT id, filename, date_start, date_end, tags,
                   total_distance_km, elevation_gain_m, elevation_loss_m,
                   max_elevation_m, min_elevation_m, moving_time_h,
                   avg_speed_kmh, min_lat, max_lat, min_lon, max_lon
            FROM routes WHERE 1=1
        """
        params = []
        
        # Filtro de Tags (case-insensitive)
        if tags and "Todos" not in tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("LOWER(tags) LIKE LOWER(?)")
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
            ne = bounds['_northEast']
            sw = bounds['_southWest']
            query += """ AND NOT (
                min_lat > ? OR max_lat < ? OR
                min_lon > ? OR max_lon < ?
            )"""
            params.extend([ne['lat'], sw['lat'], ne['lng'], sw['lng']])
            
        df = pd.read_sql_query(query, self.conn, params=params)
        
        if not df.empty:
            # Convertir fechas de forma segura, manejando errores
            df['date_start'] = pd.to_datetime(df['date_start'], errors='coerce')
            
        return df

    def get_route_points(self, route_id, simplify_factor=1):
        """
        Carga puntos de UNA ruta específica con downsampling opcional.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT points_blob FROM routes WHERE id=?", (route_id,))
        result = cursor.fetchone()
        
        if not result:
            return []
        
        points = json.loads(result[0])
        
        # Simplificación: toma 1 de cada N puntos
        if simplify_factor > 1 and len(points) > 100:
            points = points[::simplify_factor]
        
        return points

    def get_all_slow_points(self, route_ids, speed_threshold=1.5):
        """
        Extrae puntos de baja velocidad de múltiples rutas.
        Optimizado para entrenamiento de IA.
        """
        if not route_ids:
            return []
        
        placeholders = ','.join(['?'] * len(route_ids))
        query = f"SELECT points_blob FROM routes WHERE id IN ({placeholders})"
        
        cursor = self.conn.cursor()
        cursor.execute(query, route_ids)
        results = cursor.fetchall()
        
        all_slow_points = []
        for (points_json,) in results:
            points = json.loads(points_json)
            slow_pts = [
                [p['lat'], p['lon']] 
                for p in points 
                if p.get('speed', 5) < speed_threshold
            ]
            all_slow_points.extend(slow_pts)
        
        return all_slow_points

    def save_spot(self, spot_data):
        c = self.conn.cursor()
        try:
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
            return True
        except Exception as e:
            logger.error(f"Error guardando setal: {e}")
            return False

    def get_spots_df(self):
        return pd.read_sql_query("SELECT * FROM spots ORDER BY rating DESC", self.conn)
        
    def delete_element(self, table, item_id):
        """Elimina elemento con validación de tabla"""
        allowed_tables = {'routes', 'spots', 'predictions'}
        if table not in allowed_tables:
            raise ValueError(f"Tabla {table} no permitida")
        
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
                    unique.add(tag.strip().title())  # Normalizar capitalización
        return sorted(list(unique))

# Instancia global del motor de BD
db = DatabaseEngine()

# ==========================================
# 4. MOTOR DE PROCESAMIENTO GPX Y ANÁLISIS DE TERRENO
# ==========================================

class GpxProcessor:
    """
    Analizador de alto rendimiento para tracks GPS.
    """
    
    def __init__(self):
        pass

    def calculate_speeds_robust(self, dists, time_diffs):
        """
        Calcula velocidades con protección contra outliers y divisiones por cero.
        """
        speeds = np.zeros(len(dists))
        
        # Filtro robusto: tiempo > 0.1s y < 1 hora
        mask_valid = (time_diffs > 0.1) & (time_diffs < 3600)
        
        # Calcular velocidad donde es válido
        valid_indices = np.where(mask_valid)[0]
        if len(valid_indices) > 0:
            speeds[valid_indices] = (dists[valid_indices] / 1000.0) / (time_diffs[valid_indices] / 3600.0)
            
            # Limitar velocidades absurdas (> 50 km/h caminando = error GPS)
            speeds = np.clip(speeds, 0, 50)
        
        # Suavizado con media móvil
        speeds_series = pd.Series(speeds)
        speeds = speeds_series.rolling(window=5, center=True, min_periods=1).mean().fillna(0).values
        
        return speeds

    def calculate_elevation_metrics(self, df):
        """
        Calcula desnivel acumulado correctamente usando trigonometría punto a punto.
        """
        # Elevación ganada (subidas)
        uphill_mask = df['slope'] > 0.5  # Filtro: pendiente > 0.5° (elimina ruido)
        uphill = (df[uphill_mask]['dist_diff'] * 
                  np.sin(np.radians(df[uphill_mask]['slope'].clip(upper=45)))).sum()
        
        # Elevación perdida (bajadas)
        downhill_mask = df['slope'] < -0.5
        downhill = abs((df[downhill_mask]['dist_diff'] * 
                        np.sin(np.radians(df[downhill_mask]['slope'].clip(lower=-45)))).sum())
        
        return uphill, downhill

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
            
            if not points: 
                logger.warning(f"No hay puntos en {filename}")
                return None

            # Convertir a DataFrame
            df = pd.DataFrame(points)
            
            # Limpieza básica
            df = df.dropna(subset=['time']).drop_duplicates(subset=['time'])
            df = df.sort_values('time').reset_index(drop=True)
            
            if len(df) < 10:
                logger.warning(f"Muy pocos puntos en {filename}")
                return None
            
            # 2. Cálculos Vectoriales
            lats = df['lat'].values
            lons = df['lon'].values
            eles = df['ele'].values
            times = df['time'].values

            # Distancias (Haversine)
            dists = np.zeros(len(df))
            dists[1:] = GeoMath.haversine_vectorized(lats[:-1], lons[:-1], lats[1:], lons[1:])
            
            # Tiempos (Segundos)
            time_diffs = np.zeros(len(df))
            time_diffs[1:] = (times[1:] - times[:-1]).astype('timedelta64[ns]').astype(np.int64) / 1e9
            
            # Velocidades (con función robusta)
            speeds = self.calculate_speeds_robust(dists, time_diffs)

            # Azimut (Bearing) y Orientación
            bearings = np.zeros(len(df))
            bearings[1:] = GeoMath.calculate_bearing_vectorized(lats[:-1], lons[:-1], lats[1:], lons[1:])
            
            # Pendientes (Slope)
            ele_diffs = np.zeros(len(df))
            ele_diffs[1:] = eles[1:] - eles[:-1]
            
            slopes = np.zeros(len(df))
            mask_valid_dist = dists > 0.1
            slopes[mask_valid_dist] = np.degrees(np.arctan(ele_diffs[mask_valid_dist] / dists[mask_valid_dist]))
            slopes = np.clip(slopes, -45, 45)  # Limitar pendientes absurdas
            
            # Asignar cálculos al DF
            df['dist_diff'] = dists
            df['time_diff'] = time_diffs
            df['speed'] = speeds
            df['bearing'] = bearings
            df['slope'] = slopes
            
            # 3. Detección de Paradas (Segmentación Stop/Move)
            df['is_stopped'] = df['speed'] < 1.0
            df['segment_id'] = (df['is_stopped'] != df['is_stopped'].shift()).cumsum()
            
            segments_summary = []
            for seg_id, group in df.groupby('segment_id'):
                is_stop = group['is_stopped'].iloc[0]
                duration = group['time_diff'].sum()
                
                # Paradas significativas (> 1 min para capturar todas las recolecciones)
                if is_stop and duration > 60:
                    centroid_lat = group['lat'].mean()
                    centroid_lon = group['lon'].mean()
                    segments_summary.append({
                        'type': 'FORAGING_STOP',
                        'lat': centroid_lat,
                        'lon': centroid_lon,
                        'duration_s': duration,
                        'start_time': group['time'].iloc[0].isoformat()
                    })

            # 4. Cálculo de Desnivel Corregido
            uphill, downhill = self.calculate_elevation_metrics(df)

            # 5. Compilación de Metadatos
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
                'uphill': uphill,
                'downhill': downhill,
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
            
            # 6. Análisis Estadístico
            analysis = {
                'aspect_distribution': df['bearing'].apply(GeoMath.get_aspect_category).value_counts().to_dict(),
                'slope_histogram': np.histogram(df['slope'], bins=10, range=(-30, 30))[0].tolist(),
                'ele_histogram': np.histogram(df['ele'], bins=10)[0].tolist(),
                'speed_histogram': np.histogram(df['speed'], bins=10, range=(0, 6))[0].tolist()
            }
            
            # Convertir DF a lista de dicts (solo columnas esenciales)
            final_points = df[['lat', 'lon', 'ele', 'speed', 'bearing']].round(5).to_dict(orient='records')
            
            # Añadir timestamps
            for p, t in zip(final_points, df['time']):
                p['time'] = t.isoformat()
            
            return meta, final_points, analysis, segments_summary

        except Exception as e:
            logger.error(f"Error crítico procesando {filename}: {str(e)}")
            st.error(f"Error procesando {filename}: {str(e)}")
            return None

# ==========================================
# 5. MOTOR METEOROLÓGICO HÍBRIDO
# ==========================================

class WeatherIntelligence:
    """
    Gestor de meteorología avanzada con caché.
    """
    
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_historical_context(_self, lat, lon, days_back=45):
        """
        Obtiene datos de reanálisis de alta resolución (Open-Meteo).
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
                "et0_fao_evapotranspiration",
                "soil_temperature_0_to_7cm_mean",
                "wind_speed_10m_max"
            ],
            "timezone": "auto"
        }
        
        try:
            response = _self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'daily' not in data: 
                return None
            
            df = pd.DataFrame(data['daily'])
            df['time'] = pd.to_datetime(df['time'])
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error API meteo: {e}")
            st.error(f"Error conectando con satélites meteo: {e}")
            return None

    def calculate_fungi_indices(self, df_weather):
        """
        Calcula índices micológicos avanzados.
        """
        if df_weather is None or df_weather.empty: 
            return None

        # 1. Índice de Humedad del Suelo (SMI)
        retention_factor = 0.88 
        smi_values = []
        current_smi = 0
        net_water_balance = []
        
        for rain, et0 in zip(df_weather['precipitation_sum'], df_weather['et0_fao_evapotranspiration']):
            loss = et0 * 0.7 
            net_change = rain - loss
            current_smi = (current_smi * retention_factor) + rain
            current_smi = max(0, current_smi - loss)
            smi_values.append(current_smi)
            net_water_balance.append(net_change)

        df_weather['SMI'] = smi_values
        df_weather['Net_Water'] = net_water_balance

        # 2. Detección de Choque Térmico
        df_weather['temp_drop'] = df_weather['temperature_2m_min'].diff()
        
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
        
        triggers = ["STABLE", "STABLE"] + triggers
        df_weather['thermal_trigger'] = triggers

        # 3. Penalización por Viento
        recent_wind = df_weather['wind_speed_10m_max'].tail(5).mean()
        wind_penalty_factor = 1.0
        if recent_wind > 20: 
            wind_penalty_factor = 0.7
        if recent_wind > 30: 
            wind_penalty_factor = 0.4
        
        # 4. Cálculo MAI
        last_day = df_weather.iloc[-1]
        
        water_score = min(last_day['SMI'] * 1.5, 60)
        
        t_soil = last_day['soil_temperature_0_to_7cm_mean']
        temp_score = 0
        if 10 <= t_soil <= 18: 
            temp_score = 20
        elif 5 <= t_soil < 10 or 18 < t_soil <= 22: 
            temp_score = 10
        
        recent_triggers = df_weather['thermal_trigger'].tail(7)
        trigger_score = 0
        if "POTENTIAL_TRIGGER" in recent_triggers.values: 
            trigger_score = 20
        elif "MILD_TRIGGER" in recent_triggers.values: 
            trigger_score = 10
        
        final_mai = (water_score + temp_score + trigger_score) * wind_penalty_factor
        
        return int(final_mai), df_weather

# ==========================================
# 6. MOTOR DE MACHINE LEARNING
# ==========================================

class BioPredictor:
    """
    Motor de IA para modelado de nicho ecológico.
    """
    
    def __init__(self):
        self.model = None
        self.bandwidth = 0.002
        self.trained_route_ids = None

    def train_model(self, route_ids):
        """
        Entrena el modelo usando IDs de rutas (más eficiente).
        """
        if not route_ids or len(route_ids) == 0:
            return False

        # Extraer solo puntos lentos
        all_points = db.get_all_slow_points(route_ids, speed_threshold=1.5)

        if len(all_points) < 50: 
            logger.warning("Insuficientes datos para predicción")
            return False

        # Entrenar KDE
        X_train = np.radians(np.array(all_points))
        self.model = KernelDensity(bandwidth=self.bandwidth, metric='haversine')
        self.model.fit(X_train)
        self.trained_route_ids = route_ids
        
        return True

    def generate_probability_grid(self, bounds, resolution=100):
        """
        Genera matriz de probabilidad para el área visible.
        """
        if not self.model: 
            return None, None, None

        lat_min, lat_max = bounds['_southWest']['lat'], bounds['_northEast']['lat']
        lon_min, lon_max = bounds['_southWest']['lng'], bounds['_northEast']['lng']

        lat_margin = (lat_max - lat_min) * 0.1
        lon_margin = (lon_max - lon_min) * 0.1
        
        lat_grid = np.linspace(lat_min - lat_margin, lat_max + lat_margin, resolution)
        lon_grid = np.linspace(lon_min - lon_margin, lon_max + lon_margin, resolution)
        
        X, Y = np.meshgrid(lat_grid, lon_grid)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        
        log_dens = self.model.score_samples(np.radians(xy))
        Z = np.exp(log_dens).reshape(X.shape)
        
        z_min, z_max = Z.min(), Z.max()
        Z_norm = (Z - z_min) / (z_max - z_min) if z_max > z_min else Z
        
        return X, Y, Z_norm

    def predict_point_score(self, lat, lon):
        """Probabilidad para una coordenada específica"""
        if not self.model: 
            return 0
        point = np.radians([[lat, lon]])
        log_dens = self.model.score_samples(point)
        score = np.exp(log_dens)[0]
        return min(score * 1000, 100)

# Instancias Globales
gpx_processor = GpxProcessor()
meteo_engine = WeatherIntelligence()
ai_brain = BioPredictor()

# ==========================================
# 7. GESTOR DE ESTADO DE SESIÓN
# ==========================================

def init_session_state():
    """Inicializa variables persistentes"""
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [40.416, -3.703]
    if 'map_zoom' not in st.session_state:
        st.session_state.map_zoom = 6
    if 'map_bounds' not in st.session_state:
        st.session_state.map_bounds = None
    if 'selected_route_id' not in st.session_state:
        st.session_state.selected_route_id = None
    if 'last_weather_data' not in st.session_state:
        st.session_state.last_weather_data = None
    if 'layer_heatmap' not in st.session_state: 
        st.session_state.layer_heatmap = True
    if 'layer_clusters' not in st.session_state: 
        st.session_state.layer_clusters = False
    if 'layer_prediction' not in st.session_state: 
        st.session_state.layer_prediction = False
    if 'add_spot_mode' not in st.session_state:
        st.session_state.add_spot_mode = False
    if 'pending_spot_coords' not in st.session_state:
        st.session_state.pending_spot_coords = None

# ==========================================
# 8. INTERFAZ DE USUARIO - SIDEBAR
# ==========================================

def render_sidebar():
    """Construye el panel lateral de control"""
    st.sidebar.image("https://img.icons8.com/color/96/000000/mushroom.png", width=60)
    st.sidebar.title("MicoBrain OMNI")
    st.sidebar.markdown("**v4.0.2 Fixed** | *Professional Edition*")
    
    # --- MÓDULO DE IMPORTACIÓN ---
    with st.sidebar.expander("📥 Importar GPX", expanded=False):
        st.markdown("Añade nuevas rutas a tu base de datos.")
        
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
                    content = file.getvalue().decode("utf-8")
                    s_io = io.StringIO(content)
                    
                    result = gpx_processor.process_file(s_io, file.name, tags_input)
                    
                    if result:
                        meta, pts, analysis, segments = result
                        file_hash = calculate_route_fingerprint(pts)
                        
                        if db.route_exists(file_hash):
                            st.warning(f"⚠️ Duplicado: {file.name}")
                        else:
                            saved = db.save_route_full(meta, pts, analysis, segments, file_hash)
                            if saved: 
                                success_count += 1
                        
                    progress_bar.progress((i + 1) / len(files))
                
                if success_count > 0:
                    st.success(f"✅ {success_count} rutas nuevas indexadas.")
                    st.cache_data.clear()
                else:
                    st.info("No se añadieron rutas nuevas.")

    st.sidebar.divider()

    # --- FILTROS ---
    st.sidebar.subheader("🔍 Filtros de Visualización")
    
    all_tags = db.get_unique_tags()
    
    selected_tags = st.sidebar.multiselect(
        "Colecciones", 
        ["Todos"] + all_tags, 
        default=["Todos"]
    )
    
    # Rango de Fechas
    try:
        min_date_q = pd.read_sql("SELECT min(date_start) FROM routes", db.conn).iloc[0,0]
        max_date_q = pd.read_sql("SELECT max(date_start) FROM routes", db.conn).iloc[0,0]
        
        if min_date_q and max_date_q:
            # Intentar parsear fechas de forma segura
            try:
                min_d = pd.to_datetime(min_date_q).date()
                max_d = pd.to_datetime(max_date_q).date()
                
                date_range = st.sidebar.date_input(
                    "Periodo de Análisis",
                    value=(min_d, max_d),
                    min_value=min_d,
                    max_value=max_d
                )
            except:
                date_range = None
                st.sidebar.caption("Error en formato de fechas de la BD.")
        else:
            date_range = None
    except Exception as e:
        date_range = None
        st.sidebar.caption("Sube rutas para habilitar filtro de fechas.")

    st.sidebar.divider()
    
    return selected_tags, date_range

# ==========================================
# INICIO APLICACIÓN
# ==========================================

init_session_state()

tags_filter, date_filter = render_sidebar()

# Cargar metadatos (sin puntos completos)
d_range_query = date_filter if date_filter and len(date_filter) == 2 else None
df_metadata = db.get_routes_metadata_only(tags=tags_filter, date_range=d_range_query)

if df_metadata.empty:
    st.info("👋 Bienvenido a MicoBrain OMNI. Empieza importando tus archivos GPX en el panel lateral.")

# ==========================================
# TABS PRINCIPALES
# ==========================================

tab_map, tab_analysis, tab_prediction, tab_data = st.tabs([
    "🗺️ Mapa Táctico", 
    "📊 Laboratorio de Datos", 
    "🔮 Oráculo (Predicción)", 
    "💾 Gestión BD"
])

# --- TAB 1: MAPA ---
with tab_map:
    col_controls, col_map_display = st.columns([1, 4])
    
    with col_controls:
        st.markdown("### 📡 Capas Activas")
        
        st.session_state.layer_heatmap = st.checkbox("🔥 Calor Histórico", value=st.session_state.layer_heatmap)
        st.session_state.layer_clusters = st.checkbox("🤖 Clusters (Corros)", value=st.session_state.layer_clusters)
        st.session_state.layer_prediction = st.checkbox("✨ Predicción (KDE)", value=st.session_state.layer_prediction)
        
        st.markdown("---")
        st.markdown("### 📍 Favoritos")
        
        # Modo añadir punto
        if st.button("➕ Activar Modo Añadir Punto"):
            st.session_state.add_spot_mode = not st.session_state.add_spot_mode
        
        if st.session_state.add_spot_mode:
            st.info("👆 Haz clic en el mapa para añadir un nuevo setal.")
        
        fav_species_filter = st.selectbox("Filtrar Setales", ["Todos", "Boletus", "Níscalos", "Amanita", "Otros"])

    with col_map_display:
        # Configuración del Mapa
        m = folium.Map(
            location=st.session_state.map_center,
            zoom_start=st.session_state.map_zoom,
            tiles=None,
            control_scale=True
        )
        
        # Capas Base
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

        # Renderizado de Rutas (solo si hay pocas)
        if len(df_metadata) < 50 and not df_metadata.empty:
            for _, row in df_metadata.iterrows():
                try:
                    # Cargar puntos simplificados
                    pts = db.get_route_points(row['id'], simplify_factor=10)
                    coords = [[p['lat'], p['lon']] for p in pts]
                    
                    color = "#ff9800" if "níscalo" in str(row.get('tags', '')).lower() else "#2e7d32"
                    
                    # Formatear fecha de forma segura
                    try:
                        date_str = pd.to_datetime(row['date_start']).strftime('%Y-%m-%d') if pd.notna(row['date_start']) else 'Sin fecha'
                    except:
                        date_str = 'Sin fecha'
                    
                    folium.PolyLine(
                        coords, color=color, weight=2.5, opacity=0.6,
                        tooltip=f"{row['filename']} ({date_str})"
                    ).add_to(m)
                except Exception as e:
                    logger.warning(f"Error dibujando ruta {row['id']}: {e}")
                    continue

        # Capa HEATMAP
        if st.session_state.layer_heatmap and not df_metadata.empty:
            all_heat_points = []
            for _, row in df_metadata.iterrows():
                try:
                    pts = db.get_route_points(row['id'], simplify_factor=5)
                    all_heat_points.extend([[p['lat'], p['lon']] for p in pts])
                except:
                    continue
            
            if all_heat_points:
                if len(all_heat_points) > 50000:
                    all_heat_points = all_heat_points[::5]
                    
                plugins.HeatMap(
                    all_heat_points, 
                    name="Densidad Histórica",
                    radius=15, blur=20, 
                    gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
                ).add_to(m)

        # Capa CLUSTERS
        if st.session_state.layer_clusters and not df_metadata.empty:
            all_slow_points = db.get_all_slow_points(df_metadata['id'].tolist())
            
            if len(all_slow_points) > 30:
                coords = np.array(all_slow_points)
                kms_per_rad = 6371.0088
                eps_rad = 0.03 / kms_per_rad
                
                dbscan = DBSCAN(eps=eps_rad, min_samples=15, metric='haversine').fit(np.radians(coords))
                
                labels = dbscan.labels_
                unique_labels = set(labels)
                
                for k in unique_labels:
                    if k == -1: 
                        continue
                    class_members = coords[labels == k]
                    if len(class_members) > 3:
                        try:
                            hull = ConvexHull(class_members)
                            hull_pts = class_members[hull.vertices]
                            hull_pts = np.append(hull_pts, [hull_pts[0]], axis=0)
                            
                            folium.Polygon(
                                locations=hull_pts.tolist(),
                                color='#d32f2f', fill=True, fill_color='#ffcdd2', fill_opacity=0.5,
                                popup="Zona Productiva (Detectada)",
                                tooltip="Corro Detectado"
                            ).add_to(m)
                        except:
                            continue

        # Capa PREDICCIÓN
        if st.session_state.layer_prediction and not df_metadata.empty and st.session_state.map_bounds:
            route_ids = df_metadata['id'].tolist()
            
            # Cachear entrenamiento basado en hash de IDs
            route_ids_hash = hashlib.md5(str(sorted(route_ids)).encode()).hexdigest()
            
            if ai_brain.trained_route_ids != route_ids:
                success = ai_brain.train_model(route_ids)
            else:
                success = True
            
            if success:
                X, Y, Z = ai_brain.generate_probability_grid(st.session_state.map_bounds, resolution=80)
                if Z is not None:
                    high_prob_indices = np.where(Z > np.percentile(Z, 85))
                    prob_pts = list(zip(X[high_prob_indices], Y[high_prob_indices]))
                    
                    if prob_pts:
                        plugins.HeatMap(
                            prob_pts,
                            name="Predicción IA",
                            radius=25, blur=15,
                            gradient={0: 'transparent', 0.5: 'cyan', 1: 'navy'}
                        ).add_to(m)

        # Renderizado de FAVORITOS
        df_spots = db.get_spots_df()
        fg_spots = folium.FeatureGroup(name="Mis Setales")
        
        if not df_spots.empty:
            for _, spot in df_spots.iterrows():
                if fav_species_filter != "Todos" and fav_species_filter not in str(spot['species']):
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

        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.LocateControl().add_to(m)
        plugins.MeasureControl().add_to(m)

        # RENDER MAPA
        map_data = st_folium(
            m, 
            height=600, 
            width="100%",
            key="main_map_widget"
        )
        
        # Actualizar estado
        if map_data:
            if map_data.get('bounds'):
                st.session_state.map_bounds = map_data['bounds']
            if map_data.get('center'):
                st.session_state.map_center = [map_data['center']['lat'], map_data['center']['lng']]
            if map_data.get('zoom'):
                st.session_state.map_zoom = map_data['zoom']

        # GESTIÓN DE CLICS
        if st.session_state.add_spot_mode and map_data and map_data.get("last_clicked"):
            click = map_data["last_clicked"]
            st.session_state.pending_spot_coords = click
            
    # Formulario de nuevo setal (fuera del mapa para evitar reseteos)
    if st.session_state.pending_spot_coords:
        with st.sidebar.form("new_spot_form"):
            st.markdown("### ⭐ Nuevo Setal")
            click = st.session_state.pending_spot_coords
            st.caption(f"Lat: {click['lat']:.5f}, Lon: {click['lng']:.5f}")
            
            s_name = st.text_input("Nombre*", placeholder="Ej: Pinar del Duero")
            s_spec = st.selectbox("Especie", ["Boletus edulis", "Boletus pinophilus", "Níscalos", "Amanita caesarea", "Rebozuelo", "Trompeta", "Otro"])
            s_rate = st.slider("Productividad", 1, 5, 3)
            s_note = st.text_area("Notas", placeholder="Suelo ácido, bajo pinos...")
            
            col_save, col_cancel = st.columns(2)
            
            with col_save:
                if st.form_submit_button("💾 Guardar"):
                    if not s_name or not s_name.strip():
                        st.error("El nombre es obligatorio")
                    else:
                        spot_data = {
                            'name': s_name.strip(), 
                            'species': s_spec,
                            'lat': click['lat'], 
                            'lon': click['lng'],
                            'rating': s_rate, 
                            'notes': s_note.strip()
                        }
                        if db.save_spot(spot_data):
                            st.success("✅ Setal guardado!")
                            st.session_state.pending_spot_coords = None
                            st.session_state.add_spot_mode = False
                            st.rerun()
            
            with col_cancel:
                if st.form_submit_button("❌ Cancelar"):
                    st.session_state.pending_spot_coords = None
                    st.session_state.add_spot_mode = False
                    st.rerun()

# --- TAB 2: ANÁLISIS ---
with tab_analysis:
    st.header("📊 Análisis de Biotopo")
    
    if df_metadata.empty:
        st.info("No hay datos visibles con los filtros actuales.")
    else:
        # KPI Cards
        total_km = df_metadata['total_distance_km'].sum()
        total_ele = df_metadata['elevation_gain_m'].sum()
        total_hours = df_metadata['moving_time_h'].sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distancia Total", f"{total_km:.1f} km")
        c2.metric("Desnivel Positivo", f"{total_ele:.0f} m")
        c3.metric("Tiempo en Movimiento", f"{total_hours:.1f} h")
        c4.metric("Rutas Analizadas", len(df_metadata))
        
        st.divider()
        
        # Gráficos
        col_charts_1, col_charts_2 = st.columns(2)
        
        with col_charts_1:
            st.subheader("🏔️ Perfil Altitudinal")
            fig_ele = px.histogram(
                df_metadata, 
                x="max_elevation_m", 
                nbins=20, 
                title="Cotas Máximas Alcanzadas",
                color_discrete_sequence=['#5c6bc0'],
                labels={'max_elevation_m': 'Elevación (m)'}
            )
            st.plotly_chart(fig_ele, use_container_width=True)
            
        with col_charts_2:
            st.subheader("📅 Patrones Temporales")
            
            # Convertir fechas de forma segura
            df_metadata['date_start_parsed'] = pd.to_datetime(df_metadata['date_start'], errors='coerce')
            
            # Solo procesar filas con fechas válidas
            df_valid_dates = df_metadata[df_metadata['date_start_parsed'].notna()].copy()
            
            if not df_valid_dates.empty:
                df_valid_dates['month'] = df_valid_dates['date_start_parsed'].dt.month_name()
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                               'July', 'August', 'September', 'October', 'November', 'December']
                
                month_counts = df_valid_dates['month'].value_counts().reindex(month_order).dropna()
                
                fig_time = px.bar(
                    x=month_counts.index,
                    y=month_counts.values,
                    title="Salidas por Mes",
                    color_discrete_sequence=['#66bb6a'],
                    labels={'x': 'Mes', 'y': 'Número de Salidas'}
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.warning("No hay fechas válidas para mostrar patrones temporales.")
        
        # Tabla de rutas
        st.subheader("📋 Detalle de Rutas")
        display_df = df_metadata[['filename', 'date_start', 'tags', 'total_distance_km', 'elevation_gain_m', 'avg_speed_kmh']].copy()
        
        # Formatear fecha de forma segura
        display_df['date_start'] = pd.to_datetime(display_df['date_start'], errors='coerce')
        display_df['date_start'] = display_df['date_start'].dt.strftime('%Y-%m-%d')
        display_df['date_start'] = display_df['date_start'].fillna('Sin fecha')
        
        display_df.columns = ['Archivo', 'Fecha', 'Etiquetas', 'Distancia (km)', 'Desnivel (m)', 'Velocidad Media (km/h)']
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# --- TAB 3: PREDICCIÓN METEO ---
with tab_prediction:
    st.header("🔮 Predicción Meteorológica Híbrida")
    st.markdown("""
        Conexión con **Open-Meteo Archive** para calcular 
        **Evapotranspiración** y **Choque Térmico**.
    """)
    
    col_pred_ctrl, col_pred_viz = st.columns([1, 2])
    
    with col_pred_ctrl:
        st.info(f"📍 Analizando: {st.session_state.map_center[0]:.3f}, {st.session_state.map_center[1]:.3f}")
        
        if st.button("🔎 EJECUTAR ANÁLISIS BIO-METEOROLÓGICO", type="primary"):
            with st.spinner("Descargando datos satelitales (últimos 45 días)..."):
                lat, lon = st.session_state.map_center
                
                w_df = meteo_engine.fetch_historical_context(lat, lon, days_back=45)
                
                if w_df is not None:
                    mai, w_df_processed = meteo_engine.calculate_fungi_indices(w_df)
                    st.session_state.last_weather_data = (mai, w_df_processed)
                    st.success("✅ Cálculos finalizados.")
                else:
                    st.error("❌ No se pudo conectar con el servidor meteorológico.")

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
                st.markdown('<div class="success-box">🚀 <b>CONDICIONES EXCELENTES</b>: Alta probabilidad de eclosión en las próximas 48-72h.</div>', unsafe_allow_html=True)
            elif mai > 40:
                st.markdown('<div class="warning-box">⚠️ <b>CONDICIONES MEDIAS</b>: Actividad posible en zonas húmedas (regatos, umbrías).</div>', unsafe_allow_html=True)
            else:
                st.error("🛑 CONDICIONES POBRES: Suelo seco o falta de estímulo térmico.")

            # Gráfico Detallado
            st.subheader("💧 Ciclo Hidrológico del Suelo")
            fig_water = go.Figure()
            
            fig_water.add_trace(go.Bar(
                x=df_w['time'], 
                y=df_w['precipitation_sum'], 
                name='Lluvia (mm)', 
                marker_color='blue', 
                opacity=0.3
            ))
            
            fig_water.add_trace(go.Scatter(
                x=df_w['time'], 
                y=df_w['SMI'], 
                name='Humedad Suelo (SMI)', 
                line=dict(color='blue', width=3),
                fill='tozeroy',
                fillcolor='rgba(0,100,255,0.2)'
            ))
            
            fig_water.add_trace(go.Scatter(
                x=df_w['time'], 
                y=df_w['et0_fao_evapotranspiration'],
                name='Evaporación (ET0)', 
                line=dict(color='red', dash='dot', width=2)
            ))
            
            fig_water.update_layout(
                height=350, 
                title="Lluvia Real vs Agua Retenida vs Evaporación",
                xaxis_title="Fecha",
                yaxis_title="mm",
                hovermode='x unified'
            )
            st.plotly_chart(fig_water, use_container_width=True)
            
            # Gráfico de Temperatura y Triggers
            st.subheader("🌡️ Análisis Térmico")
            fig_temp = go.Figure()
            
            fig_temp.add_trace(go.Scatter(
                x=df_w['time'],
                y=df_w['temperature_2m_max'],
                name='Temp. Máxima',
                line=dict(color='orange', width=2)
            ))
            
            fig_temp.add_trace(go.Scatter(
                x=df_w['time'],
                y=df_w['temperature_2m_min'],
                name='Temp. Mínima',
                line=dict(color='lightblue', width=2)
            ))
            
            fig_temp.add_trace(go.Scatter(
                x=df_w['time'],
                y=df_w['soil_temperature_0_to_7cm_mean'],
                name='Temp. Suelo (0-7cm)',
                line=dict(color='brown', width=2, dash='dash')
            ))
            
            # Marcar triggers
            trigger_dates = df_w[df_w['thermal_trigger'].isin(['POTENTIAL_TRIGGER', 'MILD_TRIGGER'])]['time']
            if not trigger_dates.empty:
                fig_temp.add_trace(go.Scatter(
                    x=trigger_dates,
                    y=[df_w[df_w['time'].isin(trigger_dates)]['temperature_2m_min'].max() + 2] * len(trigger_dates),
                    mode='markers',
                    name='Choque Térmico',
                    marker=dict(size=15, color='red', symbol='star')
                ))
            
            fig_temp.update_layout(
                height=350,
                title="Evolución Térmica y Triggers de Fructificación",
                xaxis_title="Fecha",
                yaxis_title="Temperatura (°C)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Tabla resumen últimos 7 días
            st.subheader("📊 Resumen Últimos 7 Días")
            recent = df_w.tail(7)[['time', 'precipitation_sum', 'SMI', 'temperature_2m_min', 'soil_temperature_0_to_7cm_mean', 'thermal_trigger']].copy()
            recent.columns = ['Fecha', 'Lluvia (mm)', 'Humedad Suelo', 'Temp. Mín (°C)', 'Temp. Suelo (°C)', 'Estado']
            recent['Fecha'] = recent['Fecha'].dt.strftime('%Y-%m-%d')
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Haz clic en 'Ejecutar Análisis' para obtener la predicción meteorológica.")

# --- TAB 4: GESTIÓN DE DATOS ---
with tab_data:
    st.header("💾 Gestión de Base de Datos")
    
    tab_routes, tab_spots = st.tabs(["🗺️ Rutas", "📍 Setales"])
    
    with tab_routes:
        st.subheader("Rutas Indexadas")
        
        routes_df = pd.read_sql(
            "SELECT id, filename, date_start, tags, total_distance_km, elevation_gain_m FROM routes ORDER BY date_start DESC", 
            db.conn
        )
        
        if not routes_df.empty:
            st.dataframe(routes_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            col_del1, col_del2 = st.columns([3, 1])
            with col_del1:
                rid_del = st.number_input("ID de Ruta a Eliminar", min_value=1, step=1, key="route_del_id")
            with col_del2:
                st.write("")  # Espaciador
                st.write("")  # Espaciador
                if st.button("🗑️ Borrar Ruta", type="secondary", key="btn_del_route"):
                    try:
                        db.delete_element('routes', rid_del)
                        st.success(f"✅ Ruta {rid_del} eliminada.")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al borrar: {e}")
        else:
            st.info("No hay rutas en la base de datos.")
    
    with tab_spots:
        st.subheader("Setales Favoritos")
        
        spots_df = db.get_spots_df()
        
        if not spots_df.empty:
            # Formatear para visualización
            display_spots = spots_df[['id', 'name', 'species', 'lat', 'lon', 'rating', 'date_created', 'notes']].copy()
            display_spots.columns = ['ID', 'Nombre', 'Especie', 'Latitud', 'Longitud', 'Rating', 'Fecha Creación', 'Notas']
            
            st.dataframe(display_spots, use_container_width=True, hide_index=True)
            
            st.divider()
            
            col_del1, col_del2 = st.columns([3, 1])
            with col_del1:
                sid_del = st.number_input("ID de Setal a Eliminar", min_value=1, step=1, key="spot_del_id")
            with col_del2:
                st.write("")
                st.write("")
                if st.button("🗑️ Borrar Setal", type="secondary", key="btn_del_spot"):
                    try:
                        db.delete_element('spots', sid_del)
                        st.success(f"✅ Setal {sid_del} eliminado.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al borrar: {e}")
        else:
            st.info("No hay setales guardados. Añádelos desde el mapa.")
    
    st.divider()
    
    # Estadísticas de BD
    st.subheader("📊 Estadísticas de Base de Datos")
    
    try:
        total_routes = pd.read_sql("SELECT COUNT(*) as count FROM routes", db.conn).iloc[0]['count']
        total_spots = pd.read_sql("SELECT COUNT(*) as count FROM spots", db.conn).iloc[0]['count']
        total_points = pd.read_sql("SELECT SUM(json_array_length(points_blob)) as count FROM routes", db.conn).iloc[0]['count']
        db_size = pd.read_sql("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()", db.conn).iloc[0]['size']
        
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        col_stats1.metric("Rutas Totales", total_routes)
        col_stats2.metric("Setales Guardados", total_spots)
        col_stats3.metric("Puntos GPS", f"{int(total_points):,}" if total_points else "0")
        col_stats4.metric("Tamaño BD", f"{db_size / 1024 / 1024:.1f} MB" if db_size else "N/A")
        
    except Exception as e:
        st.warning(f"No se pudieron cargar las estadísticas: {e}")
    
    st.divider()
    
    # Herramientas de mantenimiento
    st.subheader("🔧 Herramientas de Mantenimiento")
    
    col_maint1, col_maint2 = st.columns(2)
    
    with col_maint1:
        if st.button("🧹 Limpiar Caché de Streamlit", type="secondary"):
            st.cache_data.clear()
            st.success("✅ Caché limpiada.")
    
    with col_maint2:
        if st.button("🔄 Optimizar Base de Datos (VACUUM)", type="secondary"):
            try:
                db.conn.execute("VACUUM")
                st.success("✅ Base de datos optimizada.")
            except Exception as e:
                st.error(f"Error al optimizar: {e}")
    
    # Exportar datos
    st.divider()
    st.subheader("📤 Exportar Datos")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        if st.button("💾 Descargar Rutas (CSV)"):
            try:
                export_df = pd.read_sql(
                    "SELECT filename, date_start, tags, total_distance_km, elevation_gain_m, moving_time_h FROM routes", 
                    db.conn
                )
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name="micobrain_rutas.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error al exportar: {e}")
    
    with col_exp2:
        if st.button("💾 Descargar Setales (CSV)"):
            try:
                export_df = pd.read_sql(
                    "SELECT name, species, lat, lon, rating, biotope_type, notes, date_created FROM spots", 
                    db.conn
                )
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name="micobrain_setales.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error al exportar: {e}")

# ==========================================
# FOOTER
# ==========================================

st.divider()

col_footer1, col_footer2, col_footer3 = st.columns([2, 1, 1])

with col_footer1:
    st.caption("🍄 **MicoBrain OMNI v4.0.2** - Professional Foraging Suite")
    st.caption("Desarrollado con ❤️ para la comunidad micológica")

with col_footer2:
    if not df_metadata.empty:
        st.caption(f"📊 {len(df_metadata)} rutas cargadas")
        st.caption(f"📍 {len(db.get_spots_df())} setales guardados")

with col_footer3:
    st.caption("🔗 [Documentación](https://github.com)")
    st.caption("🐛 [Reportar Bug](https://github.com)")

# ==========================================
# INFORMACIÓN DE DEBUG (Solo en desarrollo)
# ==========================================

if st.sidebar.checkbox("🔧 Modo Debug", value=False):
    st.sidebar.divider()
    st.sidebar.subheader("Debug Info")
    st.sidebar.json({
        "map_center": st.session_state.map_center,
        "map_zoom": st.session_state.map_zoom,
        "map_bounds": str(st.session_state.map_bounds)[:100] if st.session_state.map_bounds else None,
        "routes_loaded": len(df_metadata),
        "add_spot_mode": st.session_state.add_spot_mode,
        "pending_coords": st.session_state.pending_spot_coords is not None
    })
