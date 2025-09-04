# ============================================================
# app.py — Asistente Coreográfico Inteligente (Robusto)
# ============================================================

import os
import io
import json
import glob
import tempfile
import traceback
import numpy as np
import pandas as pd
import streamlit as st
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
# CSS (resumido)
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
# Carga de modelo y módulos de proyecto
# ============================================================
# Estructura esperada:
# artifacts/
#   complete_model_thresholded_bundle.joblib
#   complete_feature_cols.csv
#   complete_label_names.csv
# src/
#   model_io.py, inference.py, features.py, suggestions.py

ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")

# Diagnóstico de archivos en sidebar
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

# Carga bundle entrenado
try:
    BUNDLE = load_bundle(ART_DIR)
    PIPE = BUNDLE["pipeline"]
    FEATURE_COLS = BUNDLE["feature_cols"]
    LABEL_NAMES = BUNDLE["label_names"]
except Exception as e:
    st.error(f"No se pudo cargar el bundle del modelo entrenado desde '{ART_DIR}/'.\n\nDetalles: {e}")
    st.stop()

# Estado
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "last_tmp" not in st.session_state:
    st.session_state.last_tmp = None

# ============================================================
# Encabezado
# ============================================================
st.markdown('<h1 class="main-header">🎭 Asistente Coreográfico Inteligente</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#34495e;'>Sube un vídeo de <b>1–5 minutos</b>. "
    "Analizaremos toda su duración para darte métricas y sugerencias basadas en tu modelo entrenado.</p>",
    unsafe_allow_html=True
)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown('<div class="sub-header">⚙️ Configuración de Análisis</div>', unsafe_allow_html=True)
    model_option = st.selectbox(
        "Backbone de keypoints",
        ["MediaPipe (CPU)", "YOLOv8-Pose (GPU)"],
        index=0
    )
    conf = st.slider("Umbral de confianza (si aplica)", 0.1, 0.9, 0.5, 0.05)
    stride = st.number_input("Stride (procesar 1 de cada N frames)", min_value=1, max_value=10, value=2, step=1)

    st.markdown("---")
    show_keypoints = st.checkbox("Mostrar anotación de puntos en algunos frames", value=True)
    num_frames = st.slider("Frames a mostrar", 3, 12, 6)

    st.markdown("---")
    # Consejo solicitado (visible en la UI)
    st.info(
        "💡 Consejo:\n\n"
        "• Si usas **YOLO** con *stride* alto y el bailarín sale pequeño o hay poco contraste, "
        "puede quedarse sin detecciones. Prueba **stride = 1–2** y **conf ≈ 0.3–0.5**.\n"
        "• Con **MediaPipe**, asegúrate de un vídeo con **cuerpo completo** y **buena luz**."
    )

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["🎬 Subir Video", "📊 Resultados", "ℹ️ Info"])

# =======================
# Tab 1 — Subir Video
# =======================
with tab1:
    st.markdown('<div class="sub-header">📤 Subir Video para Análisis</div>', unsafe_allow_html=True)
    up = st.file_uploader(
        "Selecciona un video (MP4, MOV, AVI, MKV - 1–5 min)",
        type=["mp4", "mov", "avi", "mkv"]
    )
    if up is not None:
        # Limpia temporal anterior
        if st.session_state.last_tmp and os.path.exists(st.session_state.last_tmp):
            try: os.remove(st.session_state.last_tmp)
            except: pass

        tmp_path = save_uploaded_file_to_tmp(up)
        st.session_state.last_tmp = tmp_path

        duration = get_video_duration_seconds(tmp_path)
        if duration <= 0:
            st.error("⚠️ No se pudo determinar la duración. Prueba con MP4 (H.264).")
            st.stop()
        minutes = duration / 60.0
        if not (1.0 <= minutes <= 5.0):
            st.error(f"⛔ Duración: {minutes:.1f} min. Debe estar entre 1 y 5 minutos.")
            st.stop()

        st.success(f"✅ Vídeo válido: {minutes:.1f} min")
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(tmp_path)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Ejecutar análisis completo", use_container_width=True, type="primary"):
            with st.spinner("Analizando…"):
                pbar = st.progress(0)
                msg = st.empty()
                try:
                    # 1) Keypoints de TODO el vídeo
                    msg.markdown('<div class="analysis-progress">🔄 Extrayendo keypoints…</div>', unsafe_allow_html=True)
                    backbone = "yolo" if "YOLO" in model_option else "mediapipe"
                    keypoints, meta = run_inference_over_video(
                        video_path=tmp_path,
                        backbone=backbone,
                        conf=float(conf),
                        stride=int(stride)
                    )
                    pbar.progress(35)

                    # Diagnóstico runtime
                    with st.sidebar.expander("🔎 Diagnóstico runtime", expanded=False):
                        st.write("Backbone:", backbone)
                        st.write("Stride:", int(stride), "Conf:", float(conf))
                        st.write("Keypoints shape:", None if keypoints is None else getattr(keypoints, "shape", None))

                    # 2) Features (siempre dict)
                    msg.markdown('<div class="analysis-progress">📊 Calculando métricas coreográficas…</div>', unsafe_allow_html=True)

                    # Wrapper seguro por si el módulo features devuelve None
                    def _safe_features(kp, meta):
                        try:
                            feats_local = features_coreograficos(kp, meta=meta)
                            if not isinstance(feats_local, dict):
                                return {
                                    "amplitud_x": 0.0, "amplitud_y": 0.0, "amplitud_z": 0.0,
                                    "velocidad_media": 0.0, "simetria": 0.0,
                                    "nivel_rango": 0.0, "variedad_direcciones": 0.0
                                }
                            return feats_local
                        except Exception:
                            return {
                                "amplitud_x": 0.0, "amplitud_y": 0.0, "amplitud_z": 0.0,
                                "velocidad_media": 0.0, "simetria": 0.0,
                                "nivel_rango": 0.0, "variedad_direcciones": 0.0
                            }

                    if keypoints is None or not hasattr(keypoints, "shape") or keypoints.size == 0 or keypoints.shape[0] == 0:
                        st.warning("No se detectaron keypoints en el vídeo (iluminación, plano, stride alto o conf alta).")
                        feats = _safe_features(np.zeros((0, 17, 3), dtype=float), meta=meta)
                    else:
                        feats = _safe_features(keypoints, meta=meta)

                    with st.sidebar.expander("🔎 Métricas (preview)", expanded=False):
                        st.write({k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in list(feats.items())[:6]})

                    pbar.progress(60)

                    # 3) Inferencia con el pipeline entrenado (umbrales ya embebidos)
                    msg.markdown('<div class="analysis-progress">🧠 Inferencia del modelo entrenado…</div>', unsafe_allow_html=True)
                    X = ensure_feature_frame(feats, FEATURE_COLS)
                    yhat = PIPE.predict(X)[0]
                    try:
                        proba = PIPE.predict_proba(X)[0]
                        proba_list = proba.tolist()
                    except Exception:
                        proba_list = None
                    labels_on = [lbl for lbl, z in zip(LABEL_NAMES, yhat) if int(z) == 1]
                    pbar.progress(80)

                    # 4) Sugerencias
                    msg.markdown('<div class="analysis-progress">💡 Generando sugerencias…</div>', unsafe_allow_html=True)
                    suggestions = map_labels_to_suggestions(labels_on)
                    pbar.progress(95)

                    # 5) Frames para visualización
                    frames = extract_frames(tmp_path, num_frames=num_frames)

                    st.session_state.analysis_results = {
                        "feats": feats,
                        "labels_on": labels_on,
                        "proba": proba_list,
                        "suggestions": suggestions,
                        "frames": frames,
                        "video_path": tmp_path,
                        "duration_sec": float(duration),
                        "backbone": backbone,
                        "meta": meta
                    }
                    pbar.progress(100)
                    msg.markdown('<div class="analysis-progress">✅ ¡Análisis completado!</div>', unsafe_allow_html=True)
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error: {type(e).__name__}: {e}")
                    st.code(traceback.format_exc())

# =======================
# Tab 2 — Resultados
# =======================
with tab2:
    st.markdown('<div class="sub-header">📊 Resultados del Análisis</div>', unsafe_allow_html=True)
    R = st.session_state.analysis_results
    if R is None:
        st.info("Sube un vídeo y ejecuta el análisis en la pestaña anterior.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📈 Métricas Coreográficas")
            for k, v in R["feats"].items():
                if isinstance(v, (int, float, np.floating)):
                    st.markdown(f'<div class="feature-card"><span class="metric-badge">{k}</span> {float(v):.2f}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("##### 🏷️ Etiquetas activas")
            if len(R["labels_on"]) == 0:
                st.info("Sin etiquetas activas según el modelo y umbrales.")
            else:
                for t in R["labels_on"]:
                    st.markdown(f'<div class="feature-card">✅ {t}</div>', unsafe_allow_html=True)
            st.markdown("##### 💡 Sugerencias")
            if len(R["suggestions"]) == 0:
                st.warning("El modelo no generó sugerencias para este caso.")
            else:
                for i, s in enumerate(R["suggestions"], 1):
                    st.markdown(f'<div class="suggestion-card">{i}. {s}</div>', unsafe_allow_html=True)

        st.markdown("##### 🎭 Frames clave (muestra)")
        cols = st.columns(3)
        for idx, (fidx, fr) in enumerate(R["frames"]):
            with cols[idx % 3]:
                st.image(fr, caption=f"Frame {fidx}", use_column_width=True)
                if show_keypoints and idx < 3 and isinstance(R["meta"], dict) and "draw" in R["meta"]:
                    st.info("Keypoints disponibles (dibujado no implementado en esta vista).")

        st.markdown("---")
        st.markdown("##### 💾 Exportar")
        df = pd.DataFrame([R["feats"]])
        csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False)
        st.download_button("📥 Métricas (CSV)", csv_buf.getvalue(), "metricas_coreograficas.csv", "text/csv", use_container_width=True)
        st.download_button(
            "📥 Sugerencias (JSON)",
            json.dumps({"labels_on": R["labels_on"], "suggestions": R["suggestions"]}, ensure_ascii=False, indent=2),
            "sugerencias.json", "application/json", use_container_width=True
        )

# =======================
# Tab 3 — Info
# =======================
with tab3:
    st.markdown('<div class="sub-header">ℹ️ Información</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-card">
    <b>Flujo:</b> Vídeo ▶ keypoints ▶ features ▶ modelo (bundle) ▶ etiquetas ▶ sugerencias.
    <br/>El bundle incluye pipeline + umbrales + nombres de etiquetas y columnas de features.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2024 – Asistente Coreográfico Inteligente")
