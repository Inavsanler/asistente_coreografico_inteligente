# ============================================================
# app.py — Asistente Coreográfico Inteligente (Robusto)
# ============================================================

import os, io, json, glob, tempfile, traceback
import numpy as np, pandas as pd, cv2, streamlit as st

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
# Estilos (resumidos)
# ----------------------------
st.markdown("""
<style>
    .main-header { font-size:2.6rem; text-align:center; font-weight:700;
        background:linear-gradient(135deg,#6a11cb,#2575fc);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .sub-header { font-size:1.3rem; font-weight:700; border-bottom:3px solid #3498db; margin:1rem 0 .8rem 0; }
    .feature-card { background:#f7f9fc; border-left:5px solid #3498db; padding:.9rem; border-radius:12px; margin:.5rem 0; }
    .suggestion-card { background:#eefbf7; border-left:5px solid #10b981; padding:.9rem; border-radius:12px; margin:.5rem 0; }
    .metric-badge { background:#2c3e50; color:#fff; padding:.15rem .55rem; border-radius:999px; margin-right:.4rem; font-weight:700; }
    .video-container { background:#1f2937; border-radius:12px; padding:1rem; margin-bottom:1rem; }
    .analysis-progress { background:linear-gradient(135deg,#2c3e50,#3498db); color:#fff; padding:1rem; border-radius:12px; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Utilidades de vídeo
# ============================================================

def save_uploaded_file_to_tmp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def get_video_duration_seconds(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0 and frames > 0:
            cap.release()
            return float(frames / fps)
        # Fallback con timestamps
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok1, _ = cap.read(); t1 = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, int(frames // 2)))
        ok2, _ = cap.read(); t2 = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.release()
        if ok1 and ok2 and t2 > t1:
            return float((t2 - t1) / 1000.0 * 2.0)
        return 0.0
    except Exception:
        return 0.0

def extract_frames(video_path, num_frames=6):
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            cap.release(); return frames
        if total_frames < num_frames: num_frames = total_frames
        for i in range(num_frames):
            frame_idx = int(i * (total_frames / num_frames))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_idx, frame))
        cap.release()
    except Exception:
        pass
    return frames

# ============================================================
# Diagnóstico de archivos & carga de bundle
# ============================================================

ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")

with st.sidebar.expander("🧪 Diagnóstico de archivos", expanded=False):
    st.write("CWD:", os.getcwd())
    try:
        st.write("Contenido raíz:", os.listdir("."))
    except Exception:
        st.write("No se pudo listar raíz.")
    st.write("Contenido /artifacts:", os.listdir("artifacts") if os.path.exists("artifacts") else "NO existe")
    st.write("Joblib encontrados:", glob.glob("**/*.joblib", recursive=True))

from src.model_io import load_bundle, ensure_feature_frame
from src.inference import run_inference_over_video
from src.features import features_coreograficos
from src.suggestions import map_labels_to_suggestions

try:
    BUNDLE = load_bundle(ART_DIR)
    PIPE = BUNDLE["pipeline"]
    FEATURE_COLS = BUNDLE["feature_cols"]
    LABEL_NAMES = BUNDLE["label_names"]
except Exception as e:
    st.error(f"No se pudo cargar el bundle del modelo desde '{ART_DIR}/'.\nDetalles: {e}")
    st.stop()

# ============================================================
# Estado
# ============================================================
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "last_tmp" not in st.session_state:
    st.session_state.last_tmp = None

# ============================================================
# Encabezado
# ============================================================
st.markdown('<h1 class="main-header">🎭 Asistente Coreográfico Inteligente</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#34495e;'>Sube un vídeo (recomendado 10 s–10 min). Analizaremos toda su duración con tu modelo entrenado.</p>", unsafe_allow_html=True)

# ============================================================
# Sidebar (configuración)
# ============================================================
with st.sidebar:
    st.markdown('<div class="sub-header">⚙️ Configuración de Análisis</div>', unsafe_allow_html=True)
    model_option = st.selectbox("Backbone de keypoints", ["MediaPipe (CPU)", "YOLOv8-Pose (GPU)"], index=0)
    conf = st.slider("Umbral de confianza (si aplica)", 0.1, 0.9, 0.5, 0.05)
    stride = st.number_input("Stride (1 procesa todo, mayor = muestrea)", min_value=1, max_value=10, value=2, step=1)
    normalizar_xy_sidebar = st.checkbox("Normalizar XY a [0,1] (si entrenaste normalizado)", value=True)

    st.markdown("---")
    show_keypoints = st.checkbox("Mostrar anotación de puntos en algunos frames", value=True)
    num_frames = st.slider("Frames a mostrar", 3, 12, 6)

    st.markdown("---")
    st.info("💡 Consejo:\n\n• Si usas **YOLO** con *stride* alto y el bailarín sale pequeño o hay poco contraste, puede quedarse sin detecciones. "
            "Prueba **stride = 1–2** y **conf ≈ 0.3–0.5**.\n• Con **MediaPipe**, asegúrate de **cuerpo completo** y **buena luz**.")

# ============================================================
# Función: Auto-inferencia robusta (multi-pasada)
# ============================================================

def smart_infer(video_path, prefer_backbone, conf_hint=0.5, stride_hint=2):
    """
    Devuelve (keypoints, meta, used_backbone, used_params)
    Estrategia: cambia backbone y barre conf/stride hasta lograr suficientes frames con keypoints.
    """
    order = [prefer_backbone, ("mediapipe" if prefer_backbone == "yolo" else "yolo")]
    conf_grid   = [conf_hint, 0.4, 0.35, 0.3, 0.25]
    stride_grid = [max(1, stride_hint), 2, 1, 3]

    tried = []
    for bb in order:
        for c in conf_grid:
            for s in stride_grid:
                try:
                    kp, meta = run_inference_over_video(
                        video_path=video_path, backbone=bb, conf=float(c), stride=int(s), return_frame_size=True
                    )
                except Exception as e:
                    tried.append((bb, c, s, f"error:{type(e).__name__}"))
                    continue
                T = 0 if kp is None or not hasattr(kp, "shape") else kp.shape[0]
                tried.append((bb, c, s, f"T={T}"))
                if T >= 30 or (T >= 10 and meta and meta.get("fps", 0) < 20):
                    return kp, meta, bb, {"conf": c, "stride": s, "tried": tried}
    # última oportunidad: acepta cualquier T>0
    for bb, c, s, _ in tried:
        try:
            kp, meta = run_inference_over_video(
                video_path=video_path, backbone=bb, conf=float(c), stride=int(s), return_frame_size=True
            )
            if kp is not None and hasattr(kp, "shape") and kp.shape[0] > 0:
                return kp, meta, bb, {"conf": c, "stride": s, "tried": tried}
        except:
            pass
    return None, {"backend": prefer_backbone}, prefer_backbone, {"conf": conf_hint, "stride": stride_hint, "tried": tried}

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["🎬 Subir Video", "📊 Resultados", "ℹ️ Info"])

# =======================
# Tab 1 — Subir Video
# =======================
with tab1:
    st.markdown('<div class="sub-header">📤 Subir Video para Análisis</div>', unsafe_allow_html=True)
    up = st.file_uploader("Selecciona un video (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"])
    if up is not None:
        # Limpia temporal anterior
        if st.session_state.last_tmp and os.path.exists(st.session_state.last_tmp):
            try: os.remove(st.session_state.last_tmp)
            except: pass

        tmp_path = save_uploaded_file_to_tmp(up)
        st.session_state.last_tmp = tmp_path

        # Duración: advertir fuera de rango, pero procesar siempre
        duration = get_video_duration_seconds(tmp_path)
        if duration <= 0:
            st.error("⚠️ No se pudo determinar la duración. Prueba MP4 (H.264).")
            st.stop()
        minutes = duration / 60.0
        if minutes < (10/60) or minutes > 10.0:
            st.warning(f"⏱️ Duración detectada {minutes:.2f} min (recomendado 0.17–10.0). Se procesará igualmente.")
        else:
            st.success(f"✅ Vídeo válido: {minutes:.2f} min")

        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(tmp_path)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Ejecutar análisis completo", use_container_width=True, type="primary"):
            with st.spinner("Analizando…"):
                pbar = st.progress(0)
                msg = st.empty()
                try:
                    # 1) Auto-inferencia robusta
                    prefer = "yolo" if "YOLO" in model_option else "mediapipe"
                    msg.markdown('<div class="analysis-progress">🔄 Extrayendo keypoints (multi-pasada)…</div>', unsafe_allow_html=True)
                    keypoints, meta, used_bb, used_params = smart_infer(
                        tmp_path, prefer_backbone=prefer, conf_hint=float(conf), stride_hint=int(stride)
                    )
                    pbar.progress(35)

                    with st.sidebar.expander("🔎 Diagnóstico runtime", expanded=False):
                        st.write("Backbone preferido:", prefer)
                        st.write("Backbone usado:", used_bb)
                        st.write("Parámetros usados:", {k:v for k,v in used_params.items() if k!="tried"})
                        st.write("Intentos (primeros 10):", used_params.get("tried", [])[:10])
                        st.write("Keypoints shape:", None if keypoints is None else getattr(keypoints, "shape", None))
                        st.write("Meta:", meta)

                    # 2) Features (siempre dict) — normalización como en Colab
                    msg.markdown('<div class="analysis-progress">📊 Calculando métricas coreográficas…</div>', unsafe_allow_html=True)

                    def _safe_features(kp, meta):
                        try:
                            return features_coreograficos(kp, meta=meta, normalizar_xy=normalizar_xy_sidebar)
                        except Exception:
                            return {"amplitud_x":0.0,"amplitud_y":0.0,"amplitud_z":0.0,
                                    "velocidad_media":0.0,"simetria":0.0,"nivel_rango":0.0,"variedad_direcciones":0.0}

                    if keypoints is None or not hasattr(keypoints, "shape") or keypoints.shape[0] == 0:
                        st.warning("No se detectaron keypoints. Se usarán métricas nulas (fallback).")
                        feats = _safe_features(np.zeros((0,17,3), dtype=float), meta=meta)
                    else:
                        feats = _safe_features(keypoints, meta=meta)
                    pbar.progress(60)

                    with st.sidebar.expander("🔎 Métricas (preview)", expanded=False):
                        st.write({k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                  for k, v in list(feats.items())[:10]})

                    # 3) Inferencia del bundle + Fallback de etiquetas por probabilidad
                    msg.markdown('<div class="analysis-progress">🧠 Inferencia del modelo entrenado…</div>', unsafe_allow_html=True)
                    X = ensure_feature_frame(feats, FEATURE_COLS)
                    yhat = PIPE.predict(X)[0]
                    proba = None; topk = []

                    try:
                        proba = PIPE.predict_proba(X)[0]
                        top_idx = np.argsort(proba)[::-1]
                        topk = [(LABEL_NAMES[i], float(proba[i])) for i in top_idx[:5]]
                    except Exception:
                        pass

                    labels_on = [lbl for lbl, z in zip(LABEL_NAMES, yhat) if int(z) == 1]

                    # Fallback suave: activa etiquetas por proba si ninguna pasó umbral
                    if len(labels_on) == 0 and proba is not None:
                        MIN_SOFT_PROB = 0.35
                        MAX_SOFT_LABELS = 3
                        soft = [(LABEL_NAMES[i], float(proba[i])) for i in np.argsort(proba)[::-1]
                                if float(proba[i]) >= MIN_SOFT_PROB]
                        if len(soft) == 0:
                            soft = [(LABEL_NAMES[int(np.argmax(proba))], float(np.max(proba)))]
                        labels_on = [s[0] for s in soft[:MAX_SOFT_LABELS]]

                    pbar.progress(80)

                    # 4) Sugerencias (siempre alguna)
                    msg.markdown('<div class="analysis-progress">💡 Generando sugerencias…</div>', unsafe_allow_html=True)
                    suggestions = map_labels_to_suggestions(labels_on)
                    if len(suggestions) == 0:
                        suggestions = [
                            "Mejora la proyección espacial aumentando amplitud en desplazamientos.",
                            "Trabaja transiciones para mantener fluidez y continuidad.",
                            "Explora niveles (alto/medio/bajo) y cambios de dirección."
                        ]
                    pbar.progress(95)

                    # 5) Frames muestra
                    frames = extract_frames(tmp_path, num_frames=num_frames)

                    st.session_state.analysis_results = {
                        "feats": feats,
                        "labels_on": labels_on,
                        "proba": (proba.tolist() if proba is not
