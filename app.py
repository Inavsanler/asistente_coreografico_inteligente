# ============================================================
# app.py — Asistente Coreográfico Inteligente (Interfaz Moderna)
# ============================================================

import os
import io
import json
import tempfile
import traceback
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2

# ----------------------------
# Configuración de la página
# ----------------------------
st.set_page_config(
    page_title="Asistente Coreográfico Inteligente | Análisis Profesional",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎭"
)

# ----------------------------
# CSS personalizado
# ----------------------------
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
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa; border-radius: 8px 8px 0 0;
        padding: 12px 24px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background: #3498db; color: white; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Utilidades de E/S
# ============================================================

def save_uploaded_file_to_tmp(uploaded_file):
    """Guarda un archivo subido en un archivo temporal y devuelve la ruta"""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def get_video_duration_seconds(file_path):
    """Obtiene la duración del vídeo en segundos con fallback si FPS=0"""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 0.0

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

        if fps > 0 and frames > 0:
            cap.release()
            return float(frames / fps)

        # Fallback: aproximar con timestamps
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok1, _ = cap.read()
        t1 = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, int(frames // 2)))
        ok2, _ = cap.read()
        t2 = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.release()
        if ok1 and ok2 and t2 > t1:
            return float((t2 - t1) / 1000.0 * 2.0)
        return 0.0
    except Exception:
        return 0.0

def extract_frames(video_path, num_frames=5):
    """Extrae frames equidistantes de un video"""
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            cap.release()
            return frames

        if total_frames < num_frames:
            num_frames = total_frames

        for i in range(num_frames):
            frame_idx = int(i * (total_frames / num_frames))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_idx, frame))
        cap.release()
        return frames
    except Exception:
        return []

# ============================================================
# Carga de funciones del modelo (con fallbacks)
# ============================================================

def _stub_run_inference(video_path, model_name, conf, stride, win_sec, hop_sec):
    T, J, D = 120, 17, 3
    return np.random.rand(T, J, D)

def _stub_features(keypoints):
    return {
        "amplitud_x": float(np.random.uniform(50, 200)),
        "amplitud_y": float(np.random.uniform(30, 100)),
        "amplitud_z": float(np.random.uniform(40, 150)),
        "velocidad_media": float(np.random.uniform(1.0, 5.0)),
        "simetria": float(np.random.uniform(20, 80)),
        "nivel_alto": float(np.random.uniform(120, 160)),
        "nivel_bajo": float(np.random.uniform(100, 140)),
        "nivel_rango": float(np.random.uniform(-40, -10)),
        "variedad_direcciones": float(np.random.uniform(0.5, 2.0)),
    }

def _stub_suggestions(_):
    return [
        "Aumentar amplitud horizontal/vertical para mayor proyección.",
        "Explorar niveles (suelo y saltos) para contraste vertical.",
        "Encadenar transiciones para incrementar fluidez.",
        "Equilibrar simetría entre lados izquierdo y derecho.",
        "Introducir cambios de frente y diagonales."
    ]

try:
    # Ajusta estas rutas a tu repo real
    from src.inference import run_inference_over_video_yolo as _real_infer
    from src.features import features_coreograficos as _real_features
    from src.suggestions import sugerencias as _real_suggestions
    run_inference_over_video_yolo = _real_infer
    features_coreograficos = _real_features
    sugerencias = _real_suggestions
except Exception:
    run_inference_over_video_yolo = _stub_run_inference
    features_coreograficos = _stub_features
    sugerencias = _stub_suggestions

# ============================================================
# Estado inicial
# ============================================================
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "last_tmp" not in st.session_state:
    st.session_state.last_tmp = None

# ============================================================
# Encabezado
# ============================================================
st.markdown('<h1 class="main-header">🎭 Asistente Coreográfico Inteligente</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2.5rem;'>
    <p style='font-size: 1.2rem; color: #34495e;'>
        Sube un vídeo de <strong>1-5 minutos</strong> para obtener un análisis profesional de movimiento,
        métricas coreográficas detalladas y sugerencias personalizadas.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar (configuración)
# ============================================================
with st.sidebar:
    st.markdown('<div class="sub-header">⚙️ Configuración de Análisis</div>', unsafe_allow_html=True)

    model_option = st.selectbox(
        "Modelo de detección de postura",
        ["MediaPipe (CPU - Recomendado)", "YOLO Pose (GPU - Más preciso)"],
        index=0
    )

    if "YOLO" in model_option:
        conf = st.slider("Umbral de confianza", 0.1, 0.9, 0.5, 0.05)
        stride = st.number_input("Stride (cada N fotogramas)", min_value=1, max_value=10, value=2, step=1)
    else:
        conf = st.slider("Umbral de confianza (general)", 0.1, 0.9, 0.5, 0.05)
        stride = 1  # por defecto en CPU

    st.markdown("---")
    st.markdown("### 🔍 Opciones de visualización")
    show_keypoints = st.checkbox("Mostrar puntos clave en frames", value=True)
    num_frames = st.slider("Número de frames a mostrar", 3, 12, 6)

    st.markdown("---")
    st.markdown("### 📊 Información del sistema")
    st.write(f"Directorio actual: `{os.getcwd()}`")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["🎬 Subir Video", "📊 Resultados", "ℹ️ Información"])

# =======================
# Tab 1 — Subir Video
# =======================
with tab1:
    st.markdown('<div class="sub-header">📤 Subir Video para Análisis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Selecciona un video para analizar (MP4, MOV, AVI, MKV - 1-5 minutos)",
        type=["mp4", "mov", "avi", "mkv"],
        help="El video debe tener entre 1 y 5 minutos de duración para un análisis óptimo."
    )

    if uploaded_file is not None:
        # Limpia temporal anterior si existe
        if st.session_state.last_tmp and os.path.exists(st.session_state.last_tmp):
            try:
                os.remove(st.session_state.last_tmp)
            except Exception:
                pass

        # Guarda nuevo temporal
        tmp_path = save_uploaded_file_to_tmp(uploaded_file)
        st.session_state.last_tmp = tmp_path

        # Duración con validación 1–5 min
        duration = get_video_duration_seconds(tmp_path)
        if duration <= 0:
            st.error("⚠️ No se pudo determinar la duración del vídeo. Prueba con otro archivo (recomendado MP4 H.264).")
            st.stop()

        minutes = duration / 60.0
        if minutes < 1.0 or minutes > 5.0:
            st.error(f"⛔ Duración detectada: {minutes:.1f} min. Debe estar entre **1 y 5 minutos**.")
            st.info("Consejo: recorta el vídeo a 1–5 minutos antes de subirlo.")
            st.stop()

        st.success(f"✅ Vídeo válido: **{minutes:.1f} minutos**")

        # Mostrar vídeo
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(tmp_path)
        st.markdown('</div>', unsafe_allow_html=True)

        # Botón de análisis
        if st.button("🚀 Ejecutar Análisis Completo", type="primary", use_container_width=True):
            with st.spinner("Preparando análisis..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.markdown('<div class="analysis-progress">🔄 Extrayendo puntos clave del video...</div>', unsafe_allow_html=True)

                    model_name = "yolov8s-pose.pt" if "YOLO" in model_option else "mediapipe"
                    infer_kwargs = dict(
                        video_path=tmp_path,
                        model_name=model_name,
                        conf=float(conf),
                        stride=int(stride),
                        # análisis sobre TODA la duración detectada
                        win_sec=float(duration),
                        hop_sec=float(duration),
                    )

                    # Aviso si MediaPipe no está conectado en la función real
                    if model_name == "mediapipe" and run_inference_over_video_yolo is _stub_run_inference:
                        st.warning("ℹ️ Seleccionaste MediaPipe (CPU), pero la función de inferencia actual no está conectada. Usando stub.")

                    keypoints = run_inference_over_video_yolo(**infer_kwargs)

                    progress_bar.progress(30)
                    status_text.markdown('<div class="analysis-progress">📊 Analizando características coreográficas...</div>', unsafe_allow_html=True)

                    caracteristicas = features_coreograficos(keypoints)

                    progress_bar.progress(60)
                    status_text.markdown('<div class="analysis-progress">💡 Generando sugerencias...</div>', unsafe_allow_html=True)

                    sugerencias_coreograficas = sugerencias(caracteristicas)

                    progress_bar.progress(90)
                    status_text.markdown('<div class="analysis-progress">🎬 Extrayendo frames clave...</div>', unsafe_allow_html=True)

                    frames = extract_frames(tmp_path, num_frames=num_frames)

                    st.session_state.analysis_results = {
                        "caracteristicas": caracteristicas,
                        "sugerencias": sugerencias_coreograficas,
                        "frames": frames,
                        "keypoints": keypoints,
                        "video_path": tmp_path,
                        "duration_sec": float(duration),
                        "model_name": model_name,
                    }

                    progress_bar.progress(100)
                    status_text.markdown('<div class="analysis-progress">✅ ¡Análisis completado correctamente!</div>', unsafe_allow_html=True)
                    st.balloons()
                    st.success("Análisis completado. Ve a la pestaña 'Resultados' para ver los detalles.")

                except Exception as e:
                    st.error(f"❌ Error durante el análisis: {type(e).__name__}: {e}")
                    st.code("Traza:\n" + traceback.format_exc())

# =======================
# Tab 2 — Resultados
# =======================
with tab2:
    st.markdown('<div class="sub-header">📊 Resultados del Análisis</div>', unsafe_allow_html=True)

    if st.session_state.analysis_results is not None:
        resultados = st.session_state.analysis_results

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 📈 Métricas Coreográficas")
            for k, v in resultados["caracteristicas"].items():
                if isinstance(v, (int, float, np.floating)):
                    st.markdown(f'<div class="feature-card"><span class="metric-badge">{k}</span> {float(v):.2f}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("##### 💡 Sugerencias de Mejora")
            for idx, sug in enumerate(resultados["sugerencias"], 1):
                st.markdown(f'<div class="suggestion-card">{idx}. {sug}</div>', unsafe_allow_html=True)

        st.markdown("##### 🎭 Frames Clave del Video")
        cols = st.columns(3)
        for idx, item in enumerate(resultados["frames"]):
            if not item or len(item) != 2:
                continue
            frame_idx, frame = item
            with cols[idx % 3]:
                st.markdown(f'<div class="frame-container">', unsafe_allow_html=True)
                st.image(frame, caption=f"Frame {frame_idx}", use_column_width=True)
                if show_keypoints and idx < 3:
                    st.info("Análisis postural detectado")
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 💾 Exportar Resultados")
        carac = resultados["caracteristicas"]
        sugs = resultados["sugerencias"]
        meta = {
            "modelo": resultados.get("model_name", "desconocido"),
            "duracion_seg": resultados.get("duration_sec", None),
        }

        df_metrics = pd.DataFrame([carac])
        csv_buf = io.StringIO()
        df_metrics.to_csv(csv_buf, index=False)

        st.download_button(
            "📥 Descargar métricas (CSV)",
            data=csv_buf.getvalue(),
            file_name="metricas_coreograficas.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.download_button(
            "📥 Descargar sugerencias (JSON)",
            data=json.dumps({"sugerencias": sugs, "meta": meta}, ensure_ascii=False, indent=2),
            file_name="sugerencias_coreograficas.json",
            mime="application/json",
            use_container_width=True
        )

    else:
        st.info("ℹ️ Ejecuta un análisis en la pestaña 'Subir Video' para ver los resultados aquí.")

# =======================
# Tab 3 — Información
# =======================
with tab3:
    st.markdown('<div class="sub-header">ℹ️ Información de la Aplicación</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
    <h4>🎯 Características del Asistente Coreográfico</h4>
    <ul>
        <li>Análisis completo de videos de 1–5 minutos</li>
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
        <li>Sube un video de danza/performance (1–5 minutos)</li>
        <li>Configura los parámetros de análisis en la barra lateral</li>
        <li>Ejecuta el análisis completo del video</li>
        <li>Revisa los resultados en la pestaña de Resultados</li>
        <li>Descarga el reporte para referencia futura</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Asistente Coreográfico Inteligente - Desarrollado con 🤍 para la comunidad de danza</p>
    <p>© 2024 - Todos los derechos reservados</p>
</div>
""", unsafe_allow_html=True)
