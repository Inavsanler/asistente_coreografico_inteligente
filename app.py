# ============================================================
# app.py — Asistente Coreográfico (versión robusta final)
# ============================================================

# ====== BLOQUE DE IMPORTACIÓN ROBUSTA (inicio de app.py) ======
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

# Configuración de la página
st.set_page_config(
    page_title="Asistente Coreográfico Inteligente | Análisis Profesional de Movimiento",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎭"
)

# CSS personalizado mejorado para apariencia profesional
st.markdown("""
<style>
    .main-header { 
        font-size: 2.8rem; 
        color: #2c3e50; 
        text-align: center; 
        font-weight: 600;
        margin-bottom: 1.5rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sub-header { 
        font-size: 1.6rem; 
        color: #34495e; 
        margin-top: 2.5rem; 
        font-weight: 500;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .feature-card { 
        background-color: #f8f9fa; 
        padding: 1.2rem; 
        border-radius: 8px; 
        margin: 0.8rem 0; 
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .suggestion-card { 
        background-color: #e8f4fc; 
        padding: 1.2rem; 
        border-radius: 8px; 
        margin: 0.8rem 0; 
        border-left: 4px solid #2ecc71;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-badge { 
        background-color: #3498db; 
        color: white; 
        padding: 0.3rem 0.7rem; 
        border-radius: 4px; 
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3498db;
        color: white;
    }
    .stFileUploader {
        border: 2px dashed #bdc3c7;
        border-radius: 8px;
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        margin-top: 3rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Mensaje principal actualizado
st.markdown("""
<div style='text-align: center; margin-bottom: 2.5rem;'>
    <p style='font-size: 1.2rem; color: #34495e;'>Sube un vídeo de <strong>3-5 minutos</strong> para obtener un análisis profesional de movimiento, métricas coreográficas detalladas y sugerencias personalizadas.</p>
</div>
""", unsafe_allow_html=True)

# 1) Localiza posibles rutas de src/
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_CANDIDATES = [
    os.path.join(_THIS_DIR, "src"),
    os.path.join(os.path.abspath(os.path.join(_THIS_DIR, "..")), "src"),
]

SRC_DIR = None
for cand in SRC_CANDIDATES:
    if os.path.isdir(cand):
        SRC_DIR = cand
        if cand not in sys.path:
            sys.path.insert(0, cand)
        break

# 2) Intento A: importar desde paquete (src/api.py -> 'api')
predict_labels = None
run_inference_over_video_yolo = None
import_error_msg = ""

if SRC_DIR is not None:
    try:
        from api import predict_labels, run_inference_over_video_yolo, features_coreograficos, sugerencias, center_of_mass
    except Exception as e:
        import_error_msg = f"Import paquete falló: {type(e).__name__}: {e}"

# 3) Intento B: carga directa por ruta de src/api.py
if (predict_labels is None or run_inference_over_video_yolo is None or features_coreograficos is None) and SRC_DIR is not None:
    api_path = os.path.join(SRC_DIR, "api.py")
    if os.path.isfile(api_path):
        try:
            spec = importlib.util.spec_from_file_location("api_dyn", api_path)
            api_dyn = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(api_dyn)
            predict_labels = getattr(api_dyn, "predict_labels", None)
            run_inference_over_video_yolo = getattr(api_dyn, "run_inference_over_video_yolo", None)
            features_coreograficos = getattr(api_dyn, "features_coreograficos", None)
            sugerencias = getattr(api_dyn, "sugerencias", None)
            center_of_mass = getattr(api_dyn, "center_of_mass", None)
        except Exception as e:
            import_error_msg = f"Carga por ruta falló: {type(e).__name__}: {e}"

# 4) Fallback final: stubs seguros
if run_inference_over_video_yolo is None:
    st.sidebar.warning("No se encontró `run_inference_over_video_yolo`; usando STUB.")
    def run_inference_over_video_yolo(*args, **kwargs):
        # Simular keypoints de ejemplo
        T, J, D = 100, 17, 3
        return np.random.rand(T, J, D)

if predict_labels is None:
    st.sidebar.warning("No se encontró `predict_labels`; usando STUB.")
    def predict_labels(x, artifacts_dir="artifacts", threshold=None):
        return ["alineacion_brazos", "mirada_foco", "landings_stiff"], [0.82, 0.65, 0.73]

if features_coreograficos is None:
    st.sidebar.warning("No se encontró `features_coreograficos`; usando STUB.")
    def features_coreograficos(K):
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

if sugerencias is None:
    st.sidebar.warning("No se encontró `sugerencias`; usando STUB.")
    def sugerencias(feats):
        return [
            "Aumentar amplitud: proyectar más en horizontal/vertical y ampliar desplazamientos.",
            "Explorar niveles: añadir suelo (low) y saltos (high) para mayor contraste vertical.",
            "Incrementar fluidez: encadenar transiciones y elevar tonicidad en pasajes clave."
        ]

if center_of_mass is None:
    def center_of_mass(K):
        if K.shape[1] >= 13:
            return (K[:, 11, :] + K[:, 12, :]) / 2
        return K.mean(axis=1)

# 5) Diagnóstico en barra lateral
with st.sidebar.expander("🔍 Diagnóstico imports", expanded=False):
    st.write("Directorio de la app:", _THIS_DIR)
    st.write("SRC_DIR detectado:", SRC_DIR or "No encontrado")
    st.write("sys.path[0:5]:", sys.path[:5])
    if import_error_msg:
        st.error(import_error_msg)
# ====== FIN BLOQUE IMPORTACIÓN ROBUSTA ======

# ------------------------------------------------------------
# Helpers de vídeo
# ------------------------------------------------------------
def try_import_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None

def get_video_duration_seconds(file_path: str) -> float:
    cv2 = try_import_cv2()
    if cv2 is None:
        st.error("OpenCV no está instalado. No se puede procesar el video.")
        return 0.0

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        st.error("No se pudo abrir el video.")
        return 0.0
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps <= 0 or frames <= 0:
        st.error("No se pudo obtener la duración del video.")
        return 0.0
        
    return frames / fps

def save_uploaded_file_to_tmp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def extract_frames(video_path, num_frames=5):
    """Extrae frames equidistantes de un video"""
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
            # Convertir BGR a RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame_idx, frame))
    
    cap.release()
    return frames

def to_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]

def safe_convert_scores(scores):
    """
    Convierte una variedad de formatos de scores a una lista de floats.
    Args:
        scores: Puede ser una lista de números, una lista de diccionarios, o un solo diccionario.
    Returns:
        Lista de floats.
    """
    scores_list = to_list(scores)
    processed_scores = []
    for sc in scores_list:
        if isinstance(sc, dict):
            # Intentar extraer un valor numérico del diccionario
            if 'score' in sc:
                processed_scores.append(float(sc['score']))
            elif 'confidence' in sc:
                processed_scores.append(float(sc['confidence']))
            elif 'probability' in sc:
                processed_scores.append(float(sc['probability']))
            else:
                # Buscar el primer valor que sea número
                found = False
                for v in sc.values():
                    if isinstance(v, (int, float)):
                        processed_scores.append(float(v))
                        found = True
                        break
                if not found:
                    processed_scores.append(0.0)
        else:
            try:
                processed_scores.append(float(sc))
            except (TypeError, ValueError):
                processed_scores.append(0.0)
    return processed_scores

# ------------------------------------------------------------
# Verificación del dataset de Colab
# ------------------------------------------------------------
def check_colab_dataset():
    """Verifica si se está utilizando el dataset de Google Colab"""
    dataset_paths = [
        "/content/drive/MyDrive/asistente_coreografico",
        "/content/drive/MyDrive/Colab Notebooks/asistente_coreografico",
        "./asistente_coreografico"
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            return True, path
    
    return False, "No se encontró el dataset de Google Colab"

# ------------------------------------------------------------
# Verificación del directorio de artifacts
# ------------------------------------------------------------
def check_artifacts_dir(artifacts_dir="artifacts"):
    """Verifica si el directorio de artifacts existe y contiene archivos esenciales"""
    if not os.path.exists(artifacts_dir):
        return False, f"El directorio {artifacts_dir} no existe."
    
    essential_files = [
        "complete_model_thresholded_bundle.joblib",
        "complete_label_names.csv",
        "complete_feature_cols.csv",
        "complete_thresholds.json"
    ]
    
    missing_files = []
    for file in essential_files:
        if not os.path.isfile(os.path.join(artifacts_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        return False, f"Faltan archivos en {artifacts_dir}: {', '.join(missing_files)}"
    
    return True, "Artifacts completos."

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown('<h1 class="main-header">💃 Asistente Coreográfico Inteligente</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p>Sube un vídeo de <strong>3–5 minutos</strong> para analizar posturas, dinámicas y obtener <strong>sugerencias coreográficas personalizadas</strong>.</p>
</div>
""", unsafe_allow_html=True)

# Verificar dataset de Colab
dataset_found, dataset_path = check_colab_dataset()
if dataset_found:
    st.sidebar.success(f"✅ Dataset encontrado en: {dataset_path}")
else:
    st.sidebar.warning(f"⚠️ {dataset_path}")

# Verificar directorio de artifacts
artifacts_dir = "artifacts"
artifacts_ok, artifacts_msg = check_artifacts_dir(artifacts_dir)
if not artifacts_ok:
    st.sidebar.error(f"❌ {artifacts_msg}")
else:
    st.sidebar.success(f"✅ {artifacts_msg}")

with st.sidebar:
    st.markdown('<div class="sub-header">⚙️ Preferencias de Análisis</div>', unsafe_allow_html=True)
    chunk_mode = st.radio("Modo de análisis", ["Clip completo", "Segmentos temporales"], index=0)
    conf = st.slider("Confianza detección (YOLO Pose)", 0.05, 0.95, 0.25, 0.05)
    stride = st.number_input("Stride (cada N fotogramas)", min_value=1, max_value=10, value=1, step=1)
    thr_mode = st.radio("Umbral de decisión", ["Auto (por clase desde artifacts)", "Global (único valor)"], index=0)
    global_thr = None
    if "Global" in thr_mode:
        global_thr = st.slider("Umbral global", 0.0, 1.0, 0.50, 0.01)

uploaded_file = st.file_uploader("Sube tu vídeo (MP4/MOV/AVI/MKV, 3–5 minutos)", type=["mp4","mov","avi","mkv"])

# ------------------------------------------------------------
# Procesamiento
# ------------------------------------------------------------
if uploaded_file is not None:
    tmp_path = save_uploaded_file_to_tmp(uploaded_file)
    duration = get_video_duration_seconds(tmp_path)
    
    if duration <= 0.0:
        st.error("No pude leer la duración del video. Asegúrate de que OpenCV esté instalado y el video sea válido.")
    else:
        minutes = duration / 60.0
        st.info(f"📏 Duración detectada: **{minutes:.1f} min**")
        
        if duration < 180 or duration > 300:
            st.error("⚠️ El vídeo debe durar entre 3 y 5 minutos (180–300 s).")
        else:
            st.success("✅ Vídeo válido (3–5 min).")
            
            # Extraer frames para mostrar
            frames = extract_frames(tmp_path, num_frames=5)
            
            if st.button("▶️ Analizar vídeo", type="primary"):
                with st.spinner("Procesando el vídeo. Esto puede tomar varios minutos..."):
                    # Ventanas
                    if chunk_mode == "Clip completo":
                        win_sec, hop_sec = float(duration), float(duration)
                    else:
                        win_sec, hop_sec = 60.0, 60.0

                    # Inferencia
                    try:
                        keypoints = run_inference_over_video_yolo(
                            video_path=tmp_path,
                            model_name="yolov8s-pose.pt",
                            conf=float(conf),
                            stride=int(stride),
                            win_sec=win_sec,
                            hop_sec=hop_sec,
                        )
                    except Exception as e:
                        st.error(f"❌ Error en inferencia: {type(e).__name__}: {e}")
                        st.code("Traza:\n" + traceback.format_exc())
                        st.stop()

                    # Clasificación - Manejo robusto de errores
                    labels, scores = [], []
                    try:
                        thr_arg = None if global_thr is None else float(global_thr)
                        labels, scores = predict_labels(keypoints, artifacts_dir=artifacts_dir, threshold=thr_arg)
                        scores = safe_convert_scores(scores)
                    except Exception as e:
                        st.warning(f"⚠️ El modelo avanzado falló: {type(e).__name__}: {e}. Usando sugerencias por reglas.")
                        # Limpiar labels y scores para usar solo reglas
                        labels = []
                        scores = []

                # Extraer características coreográficas
                try:
                    caracteristicas = features_coreograficos(keypoints)
                    sugerencias_coreograficas = sugerencias(caracteristicas)
                except Exception as e:
                    st.error(f"❌ Error en análisis coreográfico: {type(e).__name__}: {e}")
                    st.code("Traza:\n" + traceback.format_exc())
                    st.stop()

                # Mostrar resultados
                st.markdown('<div class="sub-header">🎬 Muestras del Vídeo</div>', unsafe_allow_html=True)
                
                # Mostrar frames del video
                cols = st.columns(len(frames))
                for idx, (frame_idx, frame) in enumerate(frames):
                    with cols[idx]:
                        st.image(frame, caption=f"Frame {frame_idx}", use_column_width=True)
                
                # Mostrar características y sugerencias en dos columnas
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="sub-header">📊 Métricas Coreográficas</div>', unsafe_allow_html=True)
                    for k, v in caracteristicas.items():
                        st.markdown(f'<div class="feature-card"><span class="metric-badge">{k}</span> {v:.2f}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="sub-header">💡 Sugerencias Coreográficas</div>', unsafe_allow_html=True)
                    for idx, sugerencia in enumerate(sugerencias_coreograficas, 1):
                        st.markdown(f'<div class="suggestion-card">{idx}. {sugerencia}</div>', unsafe_allow_html=True)
                
                # Mostrar resultados del modelo avanzado si está disponible
                if labels and scores:
                    st.markdown('<div class="sub-header">🤖 Sugerencias del Modelo Avanzado</div>', unsafe_allow_html=True)
                    for label, score in zip(labels, scores):
                        st.markdown(f'<div class="model-card">{label} (confianza: {score:.2f})</div>', unsafe_allow_html=True)
                else:
                    st.info("ℹ️ No se pudieron generar sugerencias del modelo avanzado. Se muestran solo sugerencias por reglas.")
                
                # Exportar CSV
                if st.checkbox("💾 Exportar resultados"):
                    import pandas as pd
                    from datetime import datetime
                    
                    # Crear DataFrames
                    features_df = pd.DataFrame.from_dict(caracteristicas, orient='index', columns=['Valor'])
                    suggestions_df = pd.DataFrame(sugerencias_coreograficas, columns=['Sugerencias'])
                    
                    # Exportar
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    features_csv = features_df.to_csv().encode('utf-8')
                    suggestions_csv = suggestions_df.to_csv().encode('utf-8')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Descargar métricas (CSV)",
                            data=features_csv,
                            file_name=f"metricas_coreograficas_{timestamp}.csv",
                            mime="text/csv"
                        )
                    with col2:
                        st.download_button(
                            "Descargar sugerencias (CSV)",
                            data=suggestions_csv,
                            file_name=f"sugerencias_coreograficas_{timestamp}.csv",
                            mime="text/csv"
                        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Asistente Coreográfico Inteligente - Desarrollado con 🤍 para la comunidad de danza</p>
</div>
""", unsafe_allow_html=True)