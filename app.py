# ============================================================
# app.py — Asistente Coreográfico Inteligente (versión estable)
# ============================================================

from __future__ import annotations

import os
import io
import sys
import json
import math
import time
import traceback
import tempfile
from typing import Dict, Any, Optional

import numpy as np
import streamlit as st
import cv2
from PIL import Image

# -------------------------------
# Configuración de la página
# -------------------------------
st.set_page_config(
    page_title="Asistente Coreográfico Inteligente | Análisis Profesional de Movimiento",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎭",
)

# -------------------------------
# CSS minimalista
# -------------------------------
st.markdown(
    """
<style>
:root{ --pri:#111827; --sec:#374151; --acc:#2563eb; --ok:#059669; --err:#dc2626; }
.main-header{font-size:2.4rem;color:var(--pri);text-align:center;font-weight:700;margin:0.5rem 0 1rem 0}
.subtle{color:#6b7280}
.card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin:8px 0;background:#ffffffaa}
.badge{display:inline-block;padding:.25rem .5rem;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:.8rem;margin-left:.5rem}
.kpi{font-size:1.2rem;font-weight:600}
hr{border:none;border-top:1px solid #e5e7eb;margin:1rem 0}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# Bloque de import robusto
# -------------------------------
try:
    from src.inference import run_inference_over_video
except SyntaxError as e:
    st.error("Hay un **error de sintaxis** en `src/inference.py`. Revísalo. Detalle en logs.")
    st.code("".join(traceback.format_exception_only(type(e), e)))
    raise
except Exception as e:
    st.error("No se pudo importar `run_inference_over_video` desde `src/inference.py`.")
    st.code("".join(traceback.format_exception_only(type(e), e)))
    raise


# ===============================
# Utilidades locales
# ===============================
def _save_uploaded_video_to_tmp(upload) -> str:
    """Guarda un archivo subido/capturado en un temporal y devuelve la ruta."""
    suffix = ".mp4"
    if hasattr(upload, "name") and isinstance(upload.name, str):
        name = upload.name.lower()
        for ext in (".mp4", ".mov", ".avi", ".mkv"):
            if name.endswith(ext):
                suffix = ext
                break
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if hasattr(upload, "read"):
        tmp.write(upload.read())
    else:
        # st.camera_input -> getvalue()
        tmp.write(upload.getvalue())
    tmp.flush()
    return tmp.name


def _probe_video(path: str) -> Dict[str, Any]:
    """Lee metadatos simples del vídeo con OpenCV."""
    cap = cv2.VideoCapture(path)
    if not cap or not cap.isOpened():
        return {"ok": False, "reason": f"No se pudo abrir el vídeo: {path}"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return {
        "ok": True,
        "fps": float(fps),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_s": duration_s,
    }


def _nice_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _estimate_frames_for_minutes(fps: float, minutes: float) -> int:
    """Devuelve los frames aproximados para una duración objetivo (minutos)."""
    if fps <= 0:
        return int(60 * minutes * 25)  # fallback a 25fps
    return int(round(minutes * 60.0 * fps))


def _draw_quick_overlay(frame: np.ndarray, result_data: Dict[str, Any]) -> np.ndarray:
    """
    Dibuja una superposición muy simple si hay keypoints de pose (mediapipe o yolo-pose).
    - Para mediapipe: espera result_data["data"]["keypoints"][0]["pose"] = [(x_norm,y_norm,score), ...]
    - Para yolo: espera result_data["data"]["detections"][0][...]["keypoints"] = [[x,y], ...]
    """
    out = frame.copy()
    h, w = out.shape[:2]

    backend = (result_data or {}).get("backend", "")
    data = (result_data or {}).get("data", {})

    try:
        if backend == "mediapipe":
            kps0 = (data.get("keypoints") or [])
            if kps0:
                kps = kps0[0].get("pose") or []
                for (xn, yn, sc) in kps:
                    x = int(xn * w)
                    y = int(yn * h)
                    cv2.circle(out, (x, y), 3, (0, 200, 0), -1)

        elif backend == "yolo":
            det0 = (data.get("detections") or [])
            if det0:
                first = det0[0]
                # Pueden ser varias detecciones en el primer frame:
                for obj in first:
                    if "boxes" in obj:
                        for bx in [obj]:
                            bxs = bx.get("boxes")
                            if isinstance(bxs, list) and len(bxs) > 0:
                                # bxs es lista de [x1,y1,x2,y2]; dibujamos cada una
                                for rect in bxs:
                                    if len(rect) >= 4:
                                        x1, y1, x2, y2 = map(int, rect[:4])
                                        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 60, 240), 2)
                    if "keypoints" in obj:
                        pts = obj.get("keypoints") or []
                        # si keypoints es lista de pares [x,y]
                        for pt in pts:
                            if isinstance(pt, list) and len(pt) >= 2:
                                x, y = int(pt[0]), int(pt[1])
                                cv2.circle(out, (x, y), 3, (0, 200, 255), -1)
    except Exception:
        # Fallo silencioso del overlay (no crítico para la app)
        pass

    return out


# ===============================
# UI — Encabezado
# ===============================
st.markdown("<div class='main-header'>🎭 Asistente Coreográfico Inteligente</div>", unsafe_allow_html=True)
st.write("Analiza tus vídeos de danza con técnicas de *Computer Vision* (MediaPipe / YOLO-Pose) y obtén métricas básicas para sugerencias coreográficas.")

# ===============================
# Sidebar — Configuración
# ===============================
st.sidebar.header("⚙️ Configuración de análisis")

backend = st.sidebar.selectbox("Backend de visión", ["mediapipe", "yolo"], index=0, help="Elige el motor de inferencia.")
yolo_model_path = st.sidebar.text_input("Ruta del modelo YOLO (si aplica)", "artifacts/yolo.pt")

st.sidebar.markdown("**Duración objetivo del análisis** (solo se procesa el inicio del vídeo):")
target_minutes = st.sidebar.slider("Minutos", min_value=0.5, max_value=5.0, value=3.0, step=0.5)
st.sidebar.caption("Consejo: 3–5 minutos recomendado. El análisis toma los primeros N minutos del vídeo.")

advanced = st.sidebar.expander("Opciones avanzadas")
with advanced:
    manual_max_frames = st.checkbox("Fijar manualmente máx. de frames", value=False)
    max_frames_manual_value = st.number_input("Máx. frames", min_value=10, max_value=20000, value=600, step=10)
    st.caption("Si está activo, esta cifra anula la duración objetivo.")

st.sidebar.markdown("---")
st.sidebar.info("Asegúrate de que existe `src/__init__.py` para que el paquete `src` sea importable.")

# ===============================
# Carga de vídeo
# ===============================
col_up1, col_up2 = st.columns(2)
with col_up1:
    up_video = st.file_uploader("📤 Sube un vídeo (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"])
with col_up2:
    cam_video = st.camera_input("🎥 O graba con tu cámara")

video_path: Optional[str] = None
if up_video or cam_video:
    video_path = _save_uploaded_video_to_tmp(up_video or cam_video)
    st.video(video_path)

# ===============================
# Metadatos y preparación
# ===============================
if video_path:
    meta = _probe_video(video_path)
    if not meta.get("ok"):
        st.error(f"No se pudo leer el vídeo: {meta.get('reason')}")
        st.stop()

    fps = meta["fps"]
    total_frames = meta["total_frames"]
    dur = meta["duration_s"]
    width, height = meta["width"], meta["height"]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].markdown(f"<div class='kpi'>FPS</div><div class='subtle'>{fps:.2f}</div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='kpi'>Frames totales</div><div class='subtle'>{total_frames}</div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='kpi'>Duración</div><div class='subtle'>{_nice_time(dur)} ({dur:.1f}s)</div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='kpi'>Resolución</div><div class='subtle'>{width}×{height}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Calcular frames a procesar
    frames_by_minutes = _estimate_frames_for_minutes(fps, target_minutes)
    if manual_max_frames:
        max_frames_to_process = int(max_frames_manual_value)
    else:
        max_frames_to_process = int(min(frames_by_minutes, total_frames))

    st.markdown(
        f"**Se analizarán ~{max_frames_to_process} frames** "
        f"(equiv. a ~{_nice_time(max_frames_to_process / (fps or 25))})."
        " El análisis empieza desde el inicio del vídeo."
    )

    # ===============================
    # Botón de ejecución
    # ===============================
    run = st.button("🚀 Ejecutar análisis")
    if run:
        try:
            with st.spinner("Procesando vídeo..."):
                results: Dict[str, Any] = run_inference_over_video(
                    video_path,
                    backend=backend,
                    max_frames=max_frames_to_process,
                    yolo_model_path=yolo_model_path,
                )

            ok = bool(results.get("available", True))
            if not ok:
                st.warning(
                    f"El backend `{results.get('backend')}` no está disponible en este entorno. "
                    f"Detalle: {results.get('data', {}).get('reason', '—')}"
                )

            st.success(f"✅ Análisis completado con backend: **{results.get('backend')}**")
            st.caption(
                f"Frames analizados: {results.get('n_frames')} · Vídeo: `{os.path.basename(results.get('video_path',''))}`"
            )

            # Vista previa con overlay rápido (si hay datos)
            try:
                cap = cv2.VideoCapture(video_path)
                ret, frame0 = cap.read()
                cap.release()
                if ret and frame0 is not None:
                    over = _draw_quick_overlay(frame0, results)
                    over_rgb = cv2.cvtColor(over, cv2.COLOR_BGR2RGB)
                    st.image(over_rgb, caption="Vista previa (frame 0 con overlay rápido)", use_container_width=True)
            except Exception:
                pass

            # Resultados en bruto
            with st.expander("📊 Ver JSON de resultados"):
                st.json(results, expanded=False)

        except Exception as e:
            st.error("❌ Ocurrió un error durante el análisis.")
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
else:
    st.info("📌 Sube un vídeo o utiliza la cámara para comenzar el análisis.")