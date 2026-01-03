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
    def get_aspect_category(bearing):
        """
        Clasifica la orientación de una ladera para determinar Solana vs Umbría.
        Fundamental para micología (humedad vs calor).
        """
        if bearing is None or (isinstance(bearing, float) and np.isnan(bearing)): return "Plano"
        
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
        # Aceptar str o bytes
        if isinstance(file_content, bytes):
            b = file_content
        else:
            b = str(file_content).encode('utf-8')
        return hashlib.md5(b).hexdigest()

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
            # Orden: b.max_lat, b.min_lat, b.max_lon, b.min_lon
            params.extend([ne['lat'], sw['lat'], ne['lng'], sw['lng']])
            
        df = pd.read_sql_query(query, self.conn, params=params)
        
        # Deshidratar JSONs bajo demanda (lazy loading si fuera necesario, aqui lo hacemos directo)
        if not df.empty:
            # Los blobs pueden estar en columns con nombres exactos
            if 'points_blob' in df.columns:
                df['points'] = df['points_blob'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else [])
            if 'analysis_blob' in df.columns:
                df['analysis'] = df['analysis_blob'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else {})
            if 'segments_blob' in df.columns:
                df['segments'] = df['segments_blob'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else [])
            if 'date_start' in df.columns:
                df['date_start'] = pd.to_datetime(df['date_start'], errors='coerce')
            
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

            # Convertir a DataFrame para manipulació...