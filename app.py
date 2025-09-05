# ============================================================
# app.py — Asistente Coreográfico Inteligente (Progresión + Sugerencias)
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
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import streamlit as st
import cv2
from PIL import Image

# -------------------------------
# Configuración de la página
# -------------------------------
st.set_page_config(
    page_title="Asistente Coreográfico Inteligente | Análisis y Sugerencias",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎭",
)

# -------------------------------
# CSS
# -------------------------------
st.markdown(
    """
<style>
:root{ --pri:#111827; --sec:#374151; --acc:#2563eb; --ok:#059669; --warn:#b45309; --err:#dc2626; }
.main-header{font-size:2.2rem;color:var(--pri);text-align:center;font-weight:700;margin:0.5rem 0 1rem 0}
.card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin:8px 0;background:#ffffff}
.kpi{font-size:1.05rem;font-weight:600}
.small{font-size:.9rem;color:#6b7280}
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:.75rem;margin-left:.5rem}
.sugg{border-left:4px solid var(--acc);padding:.5rem .75rem;margin:.35rem 0;border-radius:8px;background:#f8fafc}
.sugg h4{margin:.2rem 0 .15rem 0;font-size:1rem}
.sugg .why{color:#6b7280;font-size:.9rem}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# IMPORT ROBUSTO DEL MOTOR DE INFERENCIA
# ============================================================
try:
    from src.inference import run_inference_over_video  # ← nuestro orquestador de CV
except SyntaxError as e:
    st.error("Hay un **error de sintaxis** en `src/inference.py`. Revísalo. Detalle en logs.")
    st.code("".join(traceback.format_exception_only(type(e), e)))
    raise
except Exception as e:
    st.error("No se pudo importar `run_inference_over_video` desde `src/inference.py`.")
    st.code("".join(traceback.format_exception_only(type(e), e)))
    raise

# ============================================================
# IMPORTS OPCIONALES (del proyecto Colab / repo original)
# Si existen, se usarán; si no, hay fallbacks internos
# ============================================================
_features_fn = None
_predict_fn = None
_suggestions_fn = None

try:
    # Ejemplos esperados por el proyecto original (ajústalo si tus nombres difieren)
    # 1) ingeniería de características
    try:
        from src.features import compute_features_from_inference as _features_fn  # noqa
    except Exception:
        from src.features import extract_features as _features_fn  # noqa
    # 2) predicción ML
    try:
        from src.model import predict_labels as _predict_fn  # noqa
    except Exception:
        pass
    # 3) generador de sugerencias
    try:
        from src.suggestions import generate_suggestions as _suggestions_fn  # noqa
    except Exception:
        from src.rules import generate_suggestions as _suggestions_fn  # noqa
except Exception:
    # Si algo falla, tiramos de fallbacks internos más abajo
    pass


# ============================================================
# UTILIDADES LOCALES
# ============================================================
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
        tmp.write(upload.getvalue())  # st.camera_input
    tmp.flush()
    return tmp.name


def _probe_video(path: str) -> Dict[str, Any]:
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
    if fps <= 0:
        return int(60 * minutes * 25)  # fallback 25fps
    return int(round(minutes * 60.0 * fps))


# ============================================================
# FALLBACKS INTERNOS (por si no existen tus módulos)
# ============================================================
def _fallback_compute_features(inf: Dict[str, Any]) -> Dict[str, float]:
    """
    Extrae métricas simples de movimiento a partir del diccionario devuelto por run_inference_over_video.
    Funciona con mediapipe (norm coords) o yolo (px).
    Devuelve:
      - motion_intensity
      - vertical_amplitude
      - lateral_drift
      - tempo_irregularity
    """
    backend = (inf or {}).get("backend", "")
    data = (inf or {}).get("data", {})
    n_frames = max(1, int(inf.get("n_frames", 0)))

    # Recolectar lista de keypoints por frame como np.array NxKx2 (x,y)
    keypoints_per_frame: List[np.ndarray] = []

    if backend == "mediapipe":
        frames = data.get("keypoints") or []  # lista de dicts: {"pose": [(xnorm,ynorm,score), ...]}
        for fr in frames:
            pts = fr.get("pose") or []
            if not pts:
                keypoints_per_frame.append(np.zeros((0, 2), dtype=float))
                continue
            arr = np.array([(p[0], p[1]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2], dtype=float)
            keypoints_per_frame.append(arr)

    elif backend == "yolo":
        frames = data.get("detections") or []  # lista por frame: [ { "keypoints": [[x,y],...], ...}, ... ]
        for fr in frames:
            # si hay varias detecciones, cogemos la primera con keypoints
            pts = None
            for obj in fr:
                if "keypoints" in obj and obj["keypoints"]:
                    pts = obj["keypoints"]
                    break
            if pts is None:
                keypoints_per_frame.append(np.zeros((0, 2), dtype=float))
            else:
                arr = np.array([[p[0], p[1]] for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2], dtype=float)
                keypoints_per_frame.append(arr)

    # Asegurar longitud consistente
    if not keypoints_per_frame:
        keypoints_per_frame = [np.zeros((0, 2), dtype=float) for _ in range(n_frames)]

    # Métricas
    # 1) Intensidad de movimiento: desplazamiento promedio por landmark y frame
    disps = []
    for i in range(1, len(keypoints_per_frame)):
        a = keypoints_per_frame[i - 1]
        b = keypoints_per_frame[i]
        if a.shape == b.shape and a.size > 0:
            d = np.linalg.norm(b - a, axis=1).mean()
            disps.append(float(d))
    motion_intensity = float(np.mean(disps)) if disps else 0.0

    # 2) Amplitud vertical: rango vertical medio (por frame)
    ranges = []
    for a in keypoints_per_frame:
        if a.size > 0:
            rng = float((a[:, 1].max() - a[:, 1].min()))
            ranges.append(rng)
    vertical_amplitude = float(np.mean(ranges)) if ranges else 0.0

    # 3) Deriva lateral: varianza del centro x a lo largo del tiempo
    centers_x = []
    for a in keypoints_per_frame:
        if a.size > 0:
            centers_x.append(float(a[:, 0].mean()))
    lateral_drift = float(np.std(centers_x)) if len(centers_x) >= 2 else 0.0

    # 4) Irregularidad de tempo: desviación sobre la diferencia entre desplazamientos consecutivos
    tempo_irreg = 0.0
    if len(disps) >= 3:
        deltas = np.diff(disps)
        tempo_irreg = float(np.std(deltas))

    return dict(
        motion_intensity=motion_intensity,
        vertical_amplitude=vertical_amplitude,
        lateral_drift=lateral_drift,
        tempo_irregularity=tempo_irreg,
    )


def _fallback_predict_labels(features: Dict[str, float], artifacts_dir: str = "artifacts") -> Tuple[List[str], List[float]]:
    """
    Predicción mínima si no hay modelo: clasifica heurísticamente tres etiquetas.
    """
    mi = features.get("motion_intensity", 0.0)
    va = features.get("vertical_amplitude", 0.0)
    ld = features.get("lateral_drift", 0.0)
    ti = features.get("tempo_irregularity", 0.0)

    labels, scores = [], []

    # Ejemplo heurístico: "Energia Alta/Media/Baja"
    if mi > 8e-2:       labels.append("Energía Alta");  scores.append(min(0.99, mi))
    elif mi > 3e-2:     labels.append("Energía Media"); scores.append(min(0.8, mi + 0.1))
    else:               labels.append("Energía Baja");   scores.append(min(0.7, 0.5 - mi))

    if va > 6e-2:       labels.append("Amplitud Elevada"); scores.append(min(0.95, va + 0.1))
    elif va > 3e-2:     labels.append("Amplitud Media");   scores.append(min(0.85, va + 0.05))
    else:               labels.append("Amplitud Reducida");scores.append(min(0.75, 0.5 - va))

    if ld > 6e-2:       labels.append("Deriva Lateral Alta"); scores.append(min(0.9, ld + 0.1))
    else:               labels.append("Deriva Lateral Controlada"); scores.append(min(0.8, 0.6 - ld))

    if ti > 2e-2:       labels.append("Tempo Irregular"); scores.append(min(0.9, ti + 0.1))
    else:               labels.append("Tempo Estable");   scores.append(min(0.85, 0.7 - ti))

    return labels, scores


def _fallback_generate_suggestions(
    features: Dict[str, float],
    labels: List[str],
    scores: List[float]
) -> List[Dict[str, Any]]:
    """
    Genera sugerencias coreográficas en lenguaje natural a partir de rasgos simples.
    """
    s = []

    mi = features.get("motion_intensity", 0.0)
    va = features.get("vertical_amplitude", 0.0)
    ld = features.get("lateral_drift", 0.0)
    ti = features.get("tempo_irregularity", 0.0)

    # Amplitud / Energía
    if "Energía Baja" in labels or mi < 3e-2:
        s.append({
            "title": "Incrementa la proyección y la amplitud de brazos",
            "severity": "media",
            "why": f"Intensidad de movimiento estimada {mi:.3f} — inferior al umbral recomendado.",
            "how": "Amplía el rango de hombros y caderas en las frases de transición; marca un acento claro en la salida y cierre de port de bras.",
        })
    elif "Energía Alta" in labels:
        s.append({
            "title": "Controla la inercia en los cambios de dirección",
            "severity": "baja",
            "why": f"Intensidad de movimiento estimada {mi:.3f} — buena proyección.",
            "how": "Añade medio tiempo de sostén tras diagonales rápidas para afianzar el equilibrio en el eje.",
        })

    # Verticalidad / Centro
    if va < 3e-2:
        s.append({
            "title": "Mayor elasticidad vertical",
            "severity": "media",
            "why": f"Amplitud vertical {va:.3f} — limitada.",
            "how": "Introduce variaciones de altura: plié–demi–gran en transiciones; alterna planos alto/medio/bajo en ocho tiempos.",
        })

    # Eje y deriva
    if ld > 6e-2:
        s.append({
            "title": "Reafirma el eje en giros y desplazamientos",
            "severity": "alta",
            "why": f"Deriva lateral {ld:.3f} — indica oscilación del centro.",
            "how": "Ensaya diagonales con foco frontal fijo y contrapeso en escápulas; usa marcas de suelo para trazar líneas limpias.",
        })

    # Musicalidad
    if ti > 2e-2:
        s.append({
            "title": "Regular el pulso entre frases",
            "severity": "media",
            "why": f"Irregularidad de tempo {ti:.3f}.",
            "how": "Trabaja con metrónomo en 8+8, manteniendo homogeneidad en entradas y cierres; sincroniza respiración con acentos musicales.",
        })

    # Siempre añadimos una sugerencia “de calidad”
    s.append({
        "title": "Clarifica intenciones en los remates",
        "severity": "baja",
        "why": "La legibilidad gestual mejora la recepción del público.",
        "how": "Define el foco y la dirección de mirada en los compases finales; reserva 1/4 de tiempo para 'presentar' el gesto.",
    })

    return s


# Wrappers que priorizan tus módulos si existen
def compute_features(inf: Dict[str, Any]) -> Dict[str, float]:
    if _features_fn is not None:
        try:
            return _features_fn(inf)
        except Exception:
            pass
    return _fallback_compute_features(inf)


def predict_labels(features: Dict[str, float], artifacts_dir: str = "artifacts") -> Tuple[List[str], List[float]]:
    if _predict_fn is not None:
        try:
            return _predict_fn(features, artifacts_dir=artifacts_dir)
        except Exception:
            pass
    return _fallback_predict_labels(features, artifacts_dir=artifacts_dir)


def generate_suggestions(features: Dict[str, float], labels: List[str], scores: List[float]) -> List[Dict[str, Any]]:
    if _suggestions_fn is not None:
        try:
            return _suggestions_fn(features, labels, scores)
        except Exception:
            pass
    return _fallback_generate_suggestions(features, labels, scores)


def _draw_quick_overlay(frame: np.ndarray, result_data: Dict[str, Any]) -> np.ndarray:
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
                for obj in first:
                    if "boxes" in obj:
                        bxs = obj.get("boxes")
                        if isinstance(bxs, list) and len(bxs) > 0:
                            for rect in bxs:
                                if len(rect) >= 4:
                                    x1, y1, x2, y2 = map(int, rect[:4])
                                    cv2.rectangle(out, (x1, y1), (x2, y2), (60, 60, 240), 2)
                    if "keypoints" in obj:
                        pts = obj.get("keypoints") or []
                        for pt in pts:
                            if isinstance(pt, list) and len(pt) >= 2:
                                x, y = int(pt[0]), int(pt[1])
                                cv2.circle(out, (x, y), 3, (0, 200, 255), -1)
    except Exception:
        pass
    return out


# ============================================================
# UI — Encabezado
# ============================================================
st.markdown("<div class='main-header'>🎭 Asistente Coreográfico Inteligente</div>", unsafe_allow_html=True)
st.write("Análisis con progresión visible y **sugerencias coreográficas**. El sistema usa tus módulos del proyecto si están disponibles; si no, aplica un motor de reglas interno para no dejarte sin feedback.")

# ============================================================
# Sidebar — Configuración
# ============================================================
st.sidebar.header("⚙️ Configuración")

backend = st.sidebar.selectbox("Backend de visión", ["mediapipe", "yolo"], index=0)
yolo_model_path = st.sidebar.text_input("Ruta del modelo YOLO (si aplica)", "artifacts/yolo.pt")

with st.sidebar.expander("⏱️ Duración a analizar"):
    target_minutes = st.slider("Minutos (inicio del vídeo)", 0.5, 5.0, 3.0, 0.5)
    st.caption("Consejo: 3–5 minutos. El sistema recorta del inicio.")

with st.sidebar.expander("⚙️ Avanzado"):
    manual_max_frames = st.checkbox("Fijar manualmente máx. de frames", value=False)
    max_frames_manual_value = st.number_input("Máx. frames", min_value=10, max_value=20000, value=600, step=10)
    artifacts_dir = st.text_input("Carpeta de artifacts (bundle, CSV, modelos)", "artifacts")

st.sidebar.markdown("---")
st.sidebar.info("Asegúrate de que `src/__init__.py` existe. Si tienes `src/features.py`, `src/model.py` y `src/suggestions.py`, se usarán automáticamente.")

# ============================================================
# Carga de vídeo
# ============================================================
col_up1, col_up2 = st.columns(2)
with col_up1:
    up_video = st.file_uploader("📤 Sube un vídeo (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"])
with col_up2:
    cam_video = st.camera_input("🎥 O graba con tu cámara")

video_path: Optional[str] = None
if up_video or cam_video:
    video_path = _save_uploaded_video_to_tmp(up_video or cam_video)
    st.video(video_path)

# ============================================================
# Metadatos y preparación
# ============================================================
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
    cols[0].markdown(f"<div class='kpi'>FPS</div><div class='small'>{fps:.2f}</div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='kpi'>Frames</div><div class='small'>{total_frames}</div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='kpi'>Duración</div><div class='small'>{_nice_time(dur)} ({dur:.1f}s)</div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='kpi'>Resolución</div><div class='small'>{width}×{height}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    frames_by_minutes = _estimate_frames_for_minutes(fps, target_minutes)
    if manual_max_frames:
        max_frames_to_process = int(max_frames_manual_value)
    else:
        max_frames_to_process = int(min(frames_by_minutes, total_frames))

    st.markdown(
        f"**Se analizarán ~{max_frames_to_process} frames** "
        f"(≈ {_nice_time(max_frames_to_process / (fps or 25))})."
    )

    # ========================================================
    # EJECUCIÓN CON PROGRESIÓN VISIBLE
    # ========================================================
    run = st.button("🚀 Ejecutar análisis")
    if run:
        progress = st.progress(0)
        status_box = st.empty()

        try:
            # 1) INFERENCIA CV
            status_box.info("① Extrayendo frames y ejecutando inferencia de pose/detecciones…")
            progress.progress(10)
            results: Dict[str, Any] = run_inference_over_video(
                video_path,
                backend=backend,
                max_frames=max_frames_to_process,
                yolo_model_path=yolo_model_path,
            )
            if not results.get("available", True):
                st.warning(
                    f"El backend `{results.get('backend')}` no está disponible. "
                    f"Detalle: {results.get('data', {}).get('reason', '—')}"
                )
            progress.progress(40)

            # Vista previa
            try:
                cap = cv2.VideoCapture(video_path)
                ret, frame0 = cap.read()
                cap.release()
                if ret and frame0 is not None:
                    over = _draw_quick_overlay(frame0, results)
                    over_rgb = cv2.cvtColor(over, cv2.COLOR_BGR2RGB)
                    st.image(over_rgb, caption="Vista previa (frame 0 con overlay)", use_container_width=True)
            except Exception:
                pass

            # 2) FEATURES
            status_box.info("② Calculando rasgos de movimiento (feature engineering)…")
            feats: Dict[str, float] = compute_features(results)
            progress.progress(65)

            # 3) PREDICCIÓN (ML si existe, si no heurístico)
            status_box.info("③ Predicción de etiquetas coreográficas…")
            labels, scores = predict_labels(feats, artifacts_dir=artifacts_dir)
            progress.progress(80)

            # 4) SUGERENCIAS (NLP)
            status_box.info("④ Generando **sugerencias coreográficas**…")
            suggestions = generate_suggestions(feats, labels, scores)
            progress.progress(95)

            # 5) SALIDA + EXPORTS
            status_box.success("⑤ ¡Listo! Análisis finalizado.")
            progress.progress(100)
            st.success(f"✅ Backend: **{results.get('backend')}** · Frames analizados: **{results.get('n_frames')}**")

            # Bloque de resultados
            colA, colB = st.columns([1, 1])
            with colA:
                st.subheader("Etiquetas y puntuaciones")
                if labels:
                    lbltbl = [{"label": l, "score": round(float(s), 3)} for l, s in zip(labels, scores)]
                    st.table(lbltbl)
                else:
                    st.write("—")

                st.subheader("Rasgos (features)")
                st.json({k: float(v) for k, v in feats.items()}, expanded=False)

            with colB:
                st.subheader("💡 Sugerencias coreográficas")
                if not suggestions:
                    st.info("No se generaron sugerencias (comprueba que hay landmarks detectados).")
                else:
                    for sug in suggestions:
                        st.markdown(
                            f"<div class='sugg'>"
                            f"<h4>• {sug.get('title','Sugerencia')}</h4>"
                            f"<div class='why'><b>Motivo:</b> {sug.get('why','')}</div>"
                            f"<div><b>Cómo aplicarlo:</b> {sug.get('how','')}</div>"
                            f"<div class='small'>Severidad: <span class='badge'>{sug.get('severity','—')}</span></div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            # Exportables
            export = {
                "video": os.path.basename(results.get("video_path", "")),
                "backend": results.get("backend"),
                "n_frames": results.get("n_frames"),
                "features": feats,
                "labels": labels,
                "scores": [float(s) for s in scores],
                "suggestions": suggestions,
            }
            st.download_button(
                "⬇️ Descargar reporte (JSON)",
                data=json.dumps(export, ensure_ascii=False, indent=2),
                file_name="reporte_coreografico.json",
                mime="application/json",
            )

            with st.expander("📦 Ver JSON bruto de inferencia"):
                st.json(results, expanded=False)

        except Exception as e:
            progress.progress(0)
            status_box.error("❌ Ocurrió un error durante el análisis.")
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
else:
    st.info("📌 Sube un vídeo o usa la cámara para comenzar.")
