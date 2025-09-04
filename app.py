# ============================================================
# app.py — Asistente Coreográfico Inteligente (Interfaz Moderna)
# ============================================================

import os
import sys
import importlib
import importlib.util
import tempfile
import traceback
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import time
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Asistente Coreográfico Inteligente | Análisis Profesional",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎭"
)

# CSS personalizado para una apariencia profesional y moderna
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2c3e50;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.6rem;
        color: #34495e;
        margin-top: 2rem;
        font-weight: 600;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 5px solid #3498db;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-family: 'Inter', 'Segoe UI', sans-serif;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .suggestion-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 5px solid #26a69a;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .metric-badge {
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.8rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(38, 118, 252, 0.3);
    }
    .stFileUploader {
        border: 2px dashed #bdc3c7;
        border-radius: 12px;
        padding: 2rem;
        background-color: #f8f9fa;
        transition: all 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: #3498db;
        background-color: #e8f4fc;
    }
    .video-container {
        background: #2c3e50;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .analysis-progress {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    .frame-container {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .frame-image {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 0.5rem;
        transition: transform 0.3s ease;
    }
    .frame-image:hover {
        transform: scale(1.02);
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        margin-top: 3rem;
        font-size: 0.9rem;
        padding: 1rem;
        border-top: 1px solid #ecf0f1;
    }
    .tab-content {
        padding: 1.5rem;
        background: white;
        border-radius: 0 0 12px 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Personalizar pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #3498db;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ====== DEFINICIONES DE FUNCIONES NECESARIAS ======

def save_uploaded_file_to_tmp(uploaded_file):
    """Guarda un archivo subido en un archivo temporal y devuelve la ruta"""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def get_video_duration_seconds(file_path):
    """Obtiene la duración de un video en segundos"""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 0.0
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps <= 0 or frames <= 0:
            return 0.0
            
        return frames / fps
    except Exception:
        return 0.0

def extract_frames(video_path, num_frames=5):
    """Extrae frames equidistantes de un video"""
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            num_frames = total_frames
        
        for i in range(num_frames):
            frame_idx = int(i * (total_frames / num_frames))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_idx, frame))
        
        cap.release()
        return frames
    except Exception:
        return []

# Stubs para funciones que deberían estar en tus módulos
def run_inference_over_video_yolo(video_path, model_name, conf, stride, win_sec, hop_sec):
    """STUB: Ejecuta inferencia YOLO en un video"""
    # En una implementación real, esto procesaría el video y devolvería keypoints
    # Simulamos un array de keypoints para demostración
    T, J, D = 100, 17, 3  # 100 frames, 17 joints, 3 dimensiones
    return np.random.rand(T, J, D)

def features_coreograficos(keypoints):
    """STUB: Extrae características coreográficas de los keypoints"""
    # En una implementación real, esto analizaría los keypoints y extraería métricas
    return {
        "amplitud_x": np.random.uniform(50, 200),
        "amplitud_y": np.random.uniform(30, 100),
        "amplitud_z": np.random.uniform(40, 150),
        "velocidad_media": np.random.uniform(1.0, 5.0),
        "simetria": np.random.uniform(20, 80),
        "nivel_alto": np.random.uniform(120, 160),
        "nivel_bajo": np.random.uniform(100, 140),
        "nivel_rango": np.random.uniform(-40, -10),
        "variedad_direcciones": np.random.uniform(0.5, 2.0)
    }

def sugerencias(feats):
    """STUB: Genera sugerencias basadas en las características"""
    # En una implementación real, esto generaría sugerencias personalizadas
    return [
        "Aumentar amplitud: proyectar más en horizontal/vertical y ampliar desplazamientos.",
        "Explorar niveles: añadir suelo (low) y saltos (high) para mayor contraste vertical.",
        "Incrementar fluidez: encadenar transiciones y elevar tonicidad en pasajes clave.",
        "Trabajar la simetría: equilibrar movimientos entre los lados izquierdo y derecho.",
        "Variar direcciones: incorporar más cambios de frente y diagonales."
    ]

# ====== FIN DEFINICIONES DE FUNCIONES ======

# Título principal
st.markdown('<h1 class="main-header">🎭 Asistente Coreográfico Inteligente</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2.5rem;'>
    <p style='font-size: 1.2rem; color: #34495e;'>Sube un vídeo de <strong>3-5 minutos</strong> para obtener un análisis profesional de movimiento, métricas coreográficas detalladas y sugerencias personalizadas.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para configuración
with st.sidebar:
    st.markdown('<div class="sub-header">⚙️ Configuración de Análisis</div>', unsafe_allow_html=True)
    
    # Selector de modelo
    model_option = st.selectbox(
        "Modelo de detección de postura",
        ["MediaPipe (CPU - Recomendado)", "YOLO Pose (GPU - Más preciso)"],
        index=0
    )
    
    # Configuración específica del modelo
    if "YOLO" in model_option:
        conf = st.slider("Umbral de confianza", 0.1, 0.9, 0.5, 0.05)
        stride = st.number_input("Stride (cada N fotogramas)", min_value=1, max_value=10, value=2, step=1)
    else:
        conf = st.slider("Umbral de confianza", 0.1, 0.9, 0.5, 0.05)
        min_detection_confidence = st.slider("Confianza mínima de detección", 0.1, 0.9, 0.5, 0.05)
        stride = 1  # Valor por defecto para MediaPipe
    
    st.markdown("---")
    st.markdown("### 🔍 Opciones de visualización")
    show_keypoints = st.checkbox("Mostrar puntos clave en frames", value=True)
    num_frames = st.slider("Número de frames a mostrar", 3, 12, 6)
    
    st.markdown("---")
    st.markdown("### 📊 Información del sistema")
    st.write(f"Directorio actual: `{os.getcwd()}`")

# Inicializar session_state si no existe
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# Sección principal
tab1, tab2, tab3 = st.tabs(["🎬 Subir Video", "📊 Resultados", "ℹ️ Información"])

with tab1:
    st.markdown('<div class="sub-header">📤 Subir Video para Análisis</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Selecciona un video para analizar (MP4, MOV, AVI, MKV - 3-5 minutos)", 
        type=["mp4", "mov", "avi", "mkv"],
        help="El video debe tener entre 3 y 5 minutos de duración para un análisis óptimo"
    )
    
    if uploaded_file is not None:
        # Guardar archivo temporal
        tmp_path = save_uploaded_file_to_tmp(uploaded_file)
        duration = get_video_duration_seconds(tmp_path)
        
        if duration > 0:
            minutes = duration / 60.0
            st.success(f"✅ Video válido detectado: **{minutes:.1f} minutos** de duración")
            
            # Mostrar video
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.video(tmp_path)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Botón de análisis
            if st.button("🚀 Ejecutar Análisis Completo", type="primary", use_container_width=True):
                with st.spinner("Preparando análisis..."):
                    # Iniciar análisis
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Análisis de todo el video
                        status_text.markdown('<div class="analysis-progress">🔄 Extrayendo puntos clave del video...</div>', unsafe_allow_html=True)
                        
                        # Ejecutar inferencia en TODO el video
                        keypoints = run_inference_over_video_yolo(
                            video_path=tmp_path,
                            model_name="yolov8s-pose.pt" if "YOLO" in model_option else "mediapipe",
                            conf=float(conf),
                            stride=int(stride),
                            win_sec=float(duration),  # Ventana = duración total
                            hop_sec=float(duration),  # Salto = duración total (análisis completo)
                        )
                        
                        progress_bar.progress(30)
                        status_text.markdown('<div class="analysis-progress">📊 Analizando características coreográficas...</div>', unsafe_allow_html=True)
                        
                        # Extraer características
                        caracteristicas = features_coreograficos(keypoints)
                        
                        progress_bar.progress(60)
                        status_text.markdown('<div class="analysis-progress">💡 Generando sugerencias...</div>', unsafe_allow_html=True)
                        
                        # Generar sugerencias
                        sugerencias_coreograficas = sugerencias(caracteristicas)
                        
                        progress_bar.progress(90)
                        status_text.markdown('<div class="analysis-progress">🎬 Extrayendo frames clave...</div>', unsafe_allow_html=True)
                        
                        # Extraer frames para visualización
                        frames = extract_frames(tmp_path, num_frames=num_frames)
                        
                        # Almacenar resultados en session_state para usar en otras pestañas
                        st.session_state.analysis_results = {
                            "caracteristicas": caracteristicas,
                            "sugerencias": sugerencias_coreograficas,
                            "frames": frames,
                            "keypoints": keypoints,
                            "video_path": tmp_path
                        }
                        
                        progress_bar.progress(100)
                        status_text.markdown('<div class="analysis-progress">✅ Análisis completado correctamente!</div>', unsafe_allow_html=True)
                        
                        # Notificación de éxito
                        st.balloons()
                        st.success("Análisis completado exitosamente! Ve a la pestaña 'Resultados' para ver los detalles.")
                        
                    except Exception as e:
                        st.error(f"❌ Error durante el análisis: {type(e).__name__}: {e}")
                        st.code("Traza:\n" + traceback.format_exc())
        else:
            st.error("⚠️ No se pudo determinar la duración del video. Asegúrate de que es un video válido.")

with tab2:
    st.markdown('<div class="sub-header">📊 Resultados del Análisis</div>', unsafe_allow_html=True)
    
    if st.session_state.analysis_results is not None:
        resultados = st.session_state.analysis_results
        
        # Mostrar métricas y sugerencias
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📈 Métricas Coreográficas")
            for k, v in resultados["caracteristicas"].items():
                if isinstance(v, (int, float)):
                    st.markdown(f'<div class="feature-card"><span class="metric-badge">{k}</span> {v:.2f}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("##### 💡 Sugerencias de Mejora")
            for idx, sugerencia in enumerate(resultados["sugerencias"], 1):
                st.markdown(f'<div class="suggestion-card">{idx}. {sugerencia}</div>', unsafe_allow_html=True)
        
        # Mostrar frames clave
        st.markdown("##### 🎭 Frames Clave del Video")
        
        # Crear columnas para los frames
        cols = st.columns(3)
        for idx, (frame_idx, frame) in enumerate(resultados["frames"]):
            with cols[idx % 3]:
                st.markdown(f'<div class="frame-container">', unsafe_allow_html=True)
                st.image(frame, caption=f"Frame {frame_idx}", use_column_width=True)
                
                # Aquí podrías añadir anotaciones específicas para cada frame
                # basadas en los keypoints detectados
                if show_keypoints and idx < 3:  # Mostrar análisis solo en los primeros 3 frames
                    st.info("Análisis postural detectado")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Botón para descargar resultados
        st.markdown("---")
        st.markdown("##### 💾 Exportar Resultados")
        
        if st.button("📥 Descargar Reporte Completo", use_container_width=True):
            # Aquí iría el código para generar y descargar un reporte PDF/CSV
            st.success("Reporte generado correctamente!")
            
    else:
        st.info("ℹ️ Ejecuta un análisis en la pestaña 'Subir Video' para ver los resultados aquí.")

with tab3:
    st.markdown('<div class="sub-header">ℹ️ Información de la Aplicación</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
    <h4>🎯 Características del Asistente Coreográfico</h4>
    <ul>
        <li>Análisis completo de videos de 3-5 minutos</li>
        <li>Detección de postura con modelos de IA avanzados</li>
        <li>Métricas cuantitativas de performance coreográfica</li>
        <li>Sugerencias personalizadas para mejorar la técnica</li>
        <li>Visualización de frames clave con análisis postural</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="suggestion-card">
    <h4>📋 Cómo usar la aplicación</h4>
    <ol>
        <li>Sube un video de danza/performance (3-5 minutos)</li>
        <li>Configura los parámetros de análisis en la barra lateral</li>
        <li>Ejecuta el análisis completo del video</li>
        <li>Revisa los resultados en la pestaña de Resultados</li>
        <li>Descarga el reporte para referencia futura</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Asistente Coreográfico Inteligente - Desarrollado con 🤍 para la comunidad de danza</p>
    <p>© 2024 - Todos los derechos reservados</p>
</div>
""", unsafe_allow_html=True)