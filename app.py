# ============================================================
# app.py — Asistente Coreográfico Inteligente (modelo auto-detectado)
#   • Upload + Cámara HTML5 (sin deps problemáticas)
#   • Modelo ML: auto-detección (incluye complete_model_thresholded_bundle.joblib)
#   • Progreso + Sugerencias
# ============================================================

from __future__ import annotations

import os
import json
import base64
import pickle
import traceback
import tempfile
import importlib.util
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import cv2

# -------------------------------
# Configuración
# -------------------------------
st.set_page_config(
    page_title="Asistente Coreográfico Inteligente | Análisis y Sugerencias",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎭",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
:root{ --pri:#111827; --sec:#374151; --acc:#2563eb; --ok:#059669; --warn:#b45309; --err:#dc2626; }
.main-header{font-size:2.2rem;color:var(--pri);text-align:center;font-weight:700;margin:0.5rem 0 1rem 0}
.card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin:8px 0;background:#ffffff}
.kpi{font-size:1.05rem;font-weight:600}
.small{font-size:.9rem;color:#6b7280}
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:.75rem;margin-left:.5rem}
.sugg{border-left:4px solid var(--acc);padding:.5rem .75rem;margin:.35rem 0;border-radius:8px;background:#f8fafc}
.sugg h4{margin:.2rem 0 .15rem 0;font-size:1rem}
.sugg .why{color:#6b7280;font-size:.9rem}
.note{font-size:.9rem;color:#6b7280}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Import del motor de inferencia del proyecto
# ============================================================
try:
    from src.inference import run_inference_over_video
except SyntaxError as e:
    st.error("Hay un **error de sintaxis** en `src/inference.py`. Revísalo.")
    st.code("".join(traceback.format_exception_only(type(e), e)))
    st.stop()
except Exception as e:
    st.error("No se pudo importar `run_inference_over_video` desde `src/inference.py`.")
    st.code("".join(traceback.format_exception_only(type(e), e)))
    st.stop()

# ============================================================
# Features opcionales (si existen en tu repo del Colab)
# ============================================================
_features_fn: Optional[callable] = None
if importlib.util.find_spec("src.features") is not None:
    try:
        from src.features import compute_features_from_inference as _features_fn  # type: ignore
    except Exception:
        try:
            from src.features import extract_features as _features_fn  # type: ignore
        except Exception:
            _features_fn = None

# ============================================================
# Utilidades de vídeo / UI
# ============================================================
def _save_uploaded_video_to_tmp(upload) -> str:
    suffix = ".mp4"
    if hasattr(upload, "name") and isinstance(upload.name, str):
        name = upload.name.lower()
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            if name.endswith(ext): suffix = ext; break
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if hasattr(upload, "read"): tmp.write(upload.read())
    else: tmp.write(upload.getvalue())
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
    return {"ok": True, "fps": float(fps), "total_frames": total_frames,
            "width": width, "height": height, "duration_s": duration_s}

def _nice_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

def _estimate_frames_for_minutes(fps: float, minutes: float) -> int:
    if fps <= 0: return int(60 * minutes * 25)
    return int(round(minutes * 60.0 * fps))

def _draw_quick_overlay(frame: np.ndarray, result_data: Dict[str, Any]) -> np.ndarray:
    out = frame.copy(); h, w = out.shape[:2]
    backend = (result_data or {}).get("backend", ""); data = (result_data or {}).get("data", {})
    try:
        if backend == "mediapipe":
            kps0 = (data.get("keypoints") or [])
            if kps0:
                for (xn, yn, sc) in kps0[0].get("pose") or []:
                    x = int(xn * w); y = int(yn * h)
                    cv2.circle(out, (x, y), 3, (0, 200, 0), -1)
        elif backend == "yolo":
            det0 = (data.get("detections") or [])
            if det0:
                for obj in det0[0]:
                    if "boxes" in obj:
                        for rect in obj.get("boxes") or []:
                            if len(rect) >= 4:
                                x1, y1, x2, y2 = map(int, rect[:4])
                                cv2.rectangle(out, (x1, y1), (x2, y2), (60, 60, 240), 2)
                    if "keypoints" in obj:
                        for pt in obj.get("keypoints") or []:
                            if isinstance(pt, list) and len(pt) >= 2:
                                x, y = int(pt[0]), int(pt[1])
                                cv2.circle(out, (x, y), 3, (0, 200, 255), -1)
    except Exception: pass
    return out

# ============================================================
# Features fallback (si no hay src.features)
# ============================================================
def _fallback_compute_features(inf: Dict[str, Any]) -> Dict[str, float]:
    backend = (inf or {}).get("backend", ""); data = (inf or {}).get("data", {})
    n_frames = max(1, int(inf.get("n_frames", 0))); kpf: List[np.ndarray] = []

    if backend == "mediapipe":
        for fr in (data.get("keypoints") or []):
            pts = fr.get("pose") or []
            arr = np.array([(p[0], p[1]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2], dtype=float)
            kpf.append(arr)
    elif backend == "yolo":
        for fr in (data.get("detections") or []):
            pts = None
            for obj in fr:
                if "keypoints" in obj and obj["keypoints"]:
                    pts = obj["keypoints"]; break
            arr = np.array([[p[0], p[1]] for p in (pts or []) if isinstance(p, (list, tuple)) and len(p) >= 2], dtype=float)
            kpf.append(arr)
    if not kpf: kpf = [np.zeros((0,2), dtype=float) for _ in range(n_frames)]

    disps = []
    for i in range(1, len(kpf)):
        a, b = kpf[i-1], kpf[i]
        if a.shape == b.shape and a.size > 0:
            disps.append(float(np.linalg.norm(b - a, axis=1).mean()))
    motion_intensity = float(np.mean(disps)) if disps else 0.0

    ranges = [float((a[:,1].max() - a[:,1].min())) for a in kpf if a.size>0]
    vertical_amplitude = float(np.mean(ranges)) if ranges else 0.0

    centers_x = [float(a[:,0].mean()) for a in kpf if a.size>0]
    lateral_drift = float(np.std(centers_x)) if len(centers_x)>=2 else 0.0

    tempo_irreg = float(np.std(np.diff(disps))) if len(disps) >= 3 else 0.0

    return dict(
        motion_intensity=motion_intensity,
        vertical_amplitude=vertical_amplitude,
        lateral_drift=lateral_drift,
        tempo_irregularity=tempo_irreg,
    )

def compute_features(inf: Dict[str, Any]) -> Dict[str, float]:
    if _features_fn is not None:
        try: return _features_fn(inf)
        except Exception: pass
    return _fallback_compute_features(inf)

# ============================================================
# Modelo — AUTO-DETECCIÓN DE ARCHIVO (incluye thresholded_bundle)
# ============================================================
def _load_json_if_exists(path: str) -> Optional[Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _find_model_file(artifacts_dir: str) -> Optional[str]:
    """
    Busca el archivo del modelo probando nombres comunes del Colab:
    """
    candidates = [
        # NOMBRE DEL COLAB (tu caso)
        "complete_model_thresholded_bundle.joblib",
        "complete_model_thresholded_bundle.pkl",
        # Variantes frecuentes
        "complete_model_thresholded.joblib",
        "complete_model_thresholded.pkl",
        "complete_model_threshold.joblib",
        "complete_model_threshold.pkl",
        # Nombre con typo anterior (compat)
        "complete_model_thersholder.joblib",
        "complete_model_thersholder.pkl",
        # Genéricos
        "model.joblib",
        "model.pkl",
    ]
    for name in candidates:
        p = os.path.join(artifacts_dir, name)
        if os.path.exists(p):
            return p
    # Último recurso: primer .joblib/.pkl en la carpeta
    for fname in os.listdir(artifacts_dir or "."):
        if fname.lower().endswith((".joblib", ".pkl")):
            return os.path.join(artifacts_dir, fname)
    return None

def _load_model_and_meta(artifacts_dir: str):
    path = _find_model_file(artifacts_dir)
    model = None
    if path and path.lower().endswith(".joblib"):
        try:
            import joblib
            model = joblib.load(path)
        except Exception as e:
            st.error(f"No se pudo cargar `{os.path.basename(path)}` (joblib).")
            st.code("".join(traceback.format_exception_only(type(e), e)))
    elif path and path.lower().endswith(".pkl"):
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            st.error(f"No se pudo cargar `{os.path.basename(path)}` (pickle).")
            st.code("".join(traceback.format_exception_only(type(e), e)))

    meta: Dict[str, Any] = {}
    fo = _load_json_if_exists(os.path.join(artifacts_dir, "feature_order.json"))
    if isinstance(fo, list): meta["feature_order"] = [str(x) for x in fo]
    cn = _load_json_if_exists(os.path.join(artifacts_dir, "class_names.json"))
    if isinstance(cn, list): meta["classes"] = [str(x) for x in cn]
    th = _load_json_if_exists(os.path.join(artifacts_dir, "thresholds.json"))
    if isinstance(th, dict): meta["thresholds"] = {str(k): float(v) for k, v in th.items() if isinstance(v, (int,float))}

    return model, meta, path

def _vectorize_features(features: Dict[str, float], feature_order: Optional[List[str]] = None) -> np.ndarray:
    keys = list(feature_order) if feature_order else sorted(features.keys())
    return np.array([[float(features.get(k, 0.0)) for k in keys]], dtype=np.float64)

def _get_model_classes(model, meta: Dict[str, Any]) -> List[str]:
    if "classes" in meta and isinstance(meta["classes"], list) and meta["classes"]:
        return [str(c) for c in meta["classes"]]
    classes = getattr(getattr(model, "named_steps", model), "classes_", None)
    if classes is None: classes = getattr(model, "classes_", None)
    if classes is not None and len(classes): return [str(c) for c in list(classes)]
    return ["Clase_0", "Clase_1"]

def _get_thresholds(classes: List[str], meta: Dict[str, Any], default: float = 0.5) -> Dict[str, float]:
    th = {c: default for c in classes}
    meta_th = meta.get("thresholds", {})
    if isinstance(meta_th, dict):
        for c, v in meta_th.items():
            try: th[str(c)] = float(v)
            except Exception: pass
    return th

def predict_with_model(
    features: Dict[str, float],
    artifacts_dir: str = "artifacts"
) -> Tuple[List[str], List[float], Dict[str, float], Optional[str]]:
    model, meta, model_path = _load_model_and_meta(artifacts_dir)
    if model is None:
        raise RuntimeError(
            "No se encontró un modelo en artifacts/ "
            "(p.ej., complete_model_thresholded_bundle.joblib)."
        )

    X = _vectorize_features(features, meta.get("feature_order"))
    probs: Optional[np.ndarray] = None

    try:
        est = model
        if hasattr(est, "predict_proba"):
            probs = est.predict_proba(X)
        elif hasattr(est, "decision_function"):
            df = est.decision_function(X)
            if isinstance(df, np.ndarray):
                probs = 1 / (1 + np.exp(-df))
        elif hasattr(est, "predict"):
            y = est.predict(X)
            classes = _get_model_classes(model, meta)
            probs = np.zeros((1, len(classes)), dtype=float)
            try:
                idx = int(y[0])
                if 0 <= idx < probs.shape[1]: probs[0, idx] = 1.0
            except Exception:
                for i, c in enumerate(classes):
                    if str(y[0]) == str(c):
                        probs[0, i] = 1.0
                        break
    except Exception as e:
        st.error("No se pudo obtener puntuaciones del modelo. Revisa compatibilidad de versiones.")
        st.code("".join(traceback.format_exception_only(type(e), e)))
        probs = None

    classes = _get_model_classes(model, meta)
    if probs is not None and probs.ndim == 1:
        probs = probs.reshape(1, -1)

    prob_map: Dict[str, float] = {}
    if probs is not None and probs.shape[1] == len(classes):
        for i, c in enumerate(classes):
            prob_map[c] = float(probs[0, i])

    thresholds = _get_thresholds(classes, meta, default=0.5)
    labels_out: List[str] = []
    scores_out: List[float] = []
    if prob_map:
        for c in classes:
            p = prob_map.get(c, 0.0)
            if p >= thresholds.get(c, 0.5):
                labels_out.append(c); scores_out.append(p)
        if not labels_out:
            best_idx = int(np.argmax([prob_map.get(c, 0.0) for c in classes]))
            labels_out = [classes[best_idx]]
            scores_out = [prob_map[classes[best_idx]]]

    return labels_out, scores_out, prob_map, model_path

# ============================================================
# Sugerencias (reglas simples)
# ============================================================
def _generate_suggestions(features: Dict[str, float], labels: List[str], scores: List[float]) -> List[Dict[str, Any]]:
    s = []
    mi = features.get("motion_intensity", 0.0)
    va = features.get("vertical_amplitude", 0.0)
    ld = features.get("lateral_drift", 0.0)
    ti = features.get("tempo_irregularity", 0.0)

    if "Energía Baja" in labels or mi < 3e-2:
        s.append({"title":"Incrementa la proyección y la amplitud de brazos","severity":"media",
                  "why":f"Intensidad {mi:.3f} — baja.","how":"Amplía rango en port de bras; acentúa salidas/cierres."})
    elif "Energía Alta" in labels:
        s.append({"title":"Controla la inercia en cambios de dirección","severity":"baja",
                  "why":f"Intensidad {mi:.3f} — buena proyección.","how":"Añade medio tiempo de sostén tras diagonales."})
    if va < 3e-2:
        s.append({"title":"Mayor elasticidad vertical","severity":"media",
                  "why":f"Amplitud vertical {va:.3f} — limitada.","how":"Introduce variaciones plié–demi–gran y planos alto/medio/bajo."})
    if ld > 6e-2:
        s.append({"title":"Reafirma el eje en giros y desplazamientos","severity":"alta",
                  "why":f"Deriva lateral {ld:.3f}.","how":"Foco frontal fijo; contrapeso en escápulas; marcas de suelo."})
    if ti > 2e-2:
        s.append({"title":"Regular el pulso entre frases","severity":"media",
                  "why":f"Irregularidad de tempo {ti:.3f}.","how":"Metrónomo en 8+8; sincroniza respiración con acentos."})
    s.append({"title":"Clarifica intenciones en remates","severity":"baja",
              "why":"Mejora la legibilidad gestual.","how":"Define foco y mirada en compases finales; 1/4 de tiempo para presentar el gesto."})
    return s

# ============================================================
# Cámara HTML5 (MediaRecorder) — sin dependencias extra
# ============================================================
def _camera_recorder_html_ui() -> Optional[str]:
    html = """
    <div style="font-family:system-ui,Segoe UI,Roboto,Arial">
      <video id="preview" autoplay playsinline style="width:100%;max-height:260px;background:#000;border-radius:12px"></video>
      <div style="margin:.5rem 0;display:flex;gap:.5rem;flex-wrap:wrap">
        <button id="startBtn">⏺️ Comenzar</button>
        <button id="stopBtn" disabled>⏹️ Detener</button>
        <button id="useBtn" disabled>💾 Usar clip</button>
        <label style="margin-left:auto">Duración máx. (s):
          <input id="maxSec" type="number" min="5" max="60" value="15" style="width:4rem">
        </label>
        <label style="margin-left:.5rem">FPS:
          <input id="fps" type="number" min="10" max="30" value="24" style="width:4rem">
        </label>
      </div>
      <div id="note" style="color:#6b7280;font-size:.9rem">Si el navegador pide permisos, acéptalos. El clip se grabará en <b>WebM</b>.</div>
      <script>
        const video = document.getElementById('preview');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const useBtn = document.getElementById('useBtn');
        const maxSec = document.getElementById('maxSec');
        const fps = document.getElementById('fps');
        let mediaStream = null, mediaRecorder = null, chunks = [], timer = null;

        function postValue(val){ window.parent.postMessage({type:'streamlit:setComponentValue', value: val}, '*'); }
        function componentReady(){ window.parent.postMessage({type:'streamlit:componentReady', value:true}, '*'); }
        componentReady();

        async function initCamera(){
          try{
            mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            video.srcObject = mediaStream;
          }catch(e){ document.getElementById('note').innerText = 'No se pudo acceder a la cámara: ' + e; }
        }
        initCamera();

        startBtn.onclick = () => {
          if (!mediaStream) return;
          chunks = [];
          let mime = 'video/webm;codecs=vp9';
          if (!MediaRecorder.isTypeSupported(mime)) mime = 'video/webm;codecs=vp8';
          if (!MediaRecorder.isTypeSupported(mime)) mime = 'video/webm';
          try{ mediaRecorder = new MediaRecorder(mediaStream, { mimeType: mime }); }
          catch(e){ document.getElementById('note').innerText = 'MediaRecorder no soportado en este navegador.'; return; }

          mediaRecorder.ondataavailable = ev => { if (ev.data && ev.data.size > 0) chunks.push(ev.data); };
          mediaRecorder.onstop = () => { stopBtn.disabled = true; useBtn.disabled = chunks.length === 0; };
          mediaRecorder.start(Math.max(1000 / parseInt(fps.value||'24'), 200));
          startBtn.disabled = true; stopBtn.disabled = false; useBtn.disabled = true;

          const limit = parseInt(maxSec.value || '15'); if (timer) clearTimeout(timer);
          timer = setTimeout(()=>{ try{ mediaRecorder.stop(); }catch{} }, limit*1000);
        };

        stopBtn.onclick = () => {
          try{ mediaRecorder && mediaRecorder.stop(); }catch{}
          startBtn.disabled = false; stopBtn.disabled = true;
        };

        useBtn.onclick = async () => {
          if (!chunks.length) return;
          const blob = new Blob(chunks, { type: 'video/webm' });
          const reader = new FileReader();
          reader.onloadend = () => { postValue(reader.result); };
          reader.readAsDataURL(blob);
        };
      </script>
    </div>
    """
    data_url = components.html(html, height=420, scrolling=False)
    if isinstance(data_url, str) and data_url.startswith("data:video/"):
        header, b64 = data_url.split(",", 1)
        ext = ".webm" if "webm" in header else ".mp4"
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=ext).name
        with open(tmp_path, "wb") as f: f.write(base64.b64decode(b64))
        st.success(f"✅ Clip guardado: {os.path.basename(tmp_path)}")
        st.video(tmp_path)
        return tmp_path
    return None

# ============================================================
# UI
# ============================================================
st.markdown("<div class='main-header'>🎭 Asistente Coreográfico Inteligente</div>", unsafe_allow_html=True)
st.write("Analiza vídeos (upload o **cámara HTML5**) y ejecuta tu **modelo ML** igual que en Colab. Progreso y sugerencias incluidos.")

st.sidebar.header("⚙️ Configuración")
# Cambiamos el default a YOLO (índice 1) para evitar el aviso de Mediapipe.
backend = st.sidebar.selectbox("Backend de visión", ["mediapipe", "yolo"], index=1)
yolo_model_path = st.sidebar.text_input("Ruta modelo YOLO (si aplica)", "artifacts/yolo.pt")

with st.sidebar.expander("⏱️ Duración a analizar (del inicio)"):
    target_minutes = st.slider("Minutos", 0.5, 5.0, 3.0, 0.5)
    st.caption("Consejo: 3–5 minutos. Se recorta desde el inicio del vídeo.")

with st.sidebar.expander("⚙️ Avanzado"):
    manual_max_frames = st.checkbox("Fijar manualmente máx. de frames", value=False)
    max_frames_manual_value = st.number_input("Máx. frames", min_value=10, max_value=20000, value=600, step=10)
    artifacts_dir = st.text_input("Carpeta de artifacts", "artifacts")

st.sidebar.markdown("---")
st.sidebar.info("Sitúa el modelo en `artifacts/` (p.ej., **complete_model_thresholded_bundle.joblib**).\n"
                "Opcional: feature_order.json, class_names.json, thresholds.json.")

tab_upload, tab_camera = st.tabs(["📤 Subir vídeo", "🎥 Grabar con cámara (HTML5)"])
video_path: Optional[str] = None

with tab_upload:
    up_video = st.file_uploader("Sube un vídeo (mp4/mov/avi/mkv/webm)", type=["mp4", "mov", "avi", "mkv", "webm"])
    if up_video:
        video_path = _save_uploaded_video_to_tmp(up_video)
        st.video(video_path)

with tab_camera:
    st.markdown("Graba un **clip de vídeo** desde tu cámara (HTML5) y úsalo directamente en el análisis.")
    cam_saved = _camera_recorder_html_ui()
    if cam_saved:
        video_path = cam_saved

# ============================================================
# Pipeline — INFERENCIA → FEATURES → ML → SUGERENCIAS
# ============================================================
def _run_pipeline(video_path: str):
    meta = _probe_video(video_path)
    if not meta.get("ok"):
        st.error(f"No se pudo leer el vídeo: {meta.get('reason')}"); st.stop()

    fps = meta["fps"]; total_frames = meta["total_frames"]
    dur = meta["duration_s"]; width, height = meta["width"], meta["height"]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].markdown(f"<div class='kpi'>FPS</div><div class='small'>{fps:.2f}</div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='kpi'>Frames</div><div class='small'>{total_frames}</div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='kpi'>Duración</div><div class='small'>{_nice_time(dur)} ({dur:.1f}s)</div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='kpi'>Resolución</div><div class='small'>{width}×{height}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    frames_by_minutes = _estimate_frames_for_minutes(fps, target_minutes)
    max_frames_to_process = int(max_frames_manual_value) if manual_max_frames else int(min(frames_by_minutes, total_frames))
    st.markdown(f"**Se analizarán ~{max_frames_to_process} frames** (≈ {_nice_time(max_frames_to_process / (fps or 25))}).")

    progress = st.progress(0); status_box = st.empty()

    # ① INFERENCIA
    status_box.info("① Extrayendo frames y ejecutando inferencia (pose/detecciones)…")
    progress.progress(10)
    results: Dict[str, Any] = run_inference_over_video(
        video_path, backend=backend, max_frames=max_frames_to_process, yolo_model_path=yolo_model_path
    )
    if not results.get("available", True):
        st.warning(
            f"El backend `{results.get('backend')}` no está disponible. "
            f"Detalle: {results.get('data', {}).get('reason', '—')}"
        )
    progress.progress(40)

    # Vista previa
    try:
        cap = cv2.VideoCapture(video_path); ret, frame0 = cap.read(); cap.release()
        if ret and frame0 is not None:
            over = _draw_quick_overlay(frame0, results)
            st.image(cv2.cvtColor(over, cv2.COLOR_BGR2RGB), caption="Vista previa (frame 0 con overlay)", use_container_width=True)
    except Exception: pass

    # ② FEATURES
    status_box.info("② Calculando rasgos de movimiento (feature engineering)…")
    feats: Dict[str, float] = compute_features(results)
    progress.progress(65)

    # ③ MODELO
    status_box.info("③ Ejecutando modelo ML (umbralado por clase)…")
    labels, scores, prob_map, model_path = [], [], {}, None
    model_ok = True
    try:
        labels, scores, prob_map, model_path = predict_with_model(feats, artifacts_dir=artifacts_dir)
    except Exception as e:
        model_ok = False
        st.error("No fue posible ejecutar el modelo (¿archivo no encontrado o incompatible?).")
        st.code("".join(traceback.format_exception_only(type(e), e)))
    progress.progress(80)

    # ④ SUGERENCIAS
    status_box.info("④ Generando **sugerencias coreográficas**…")
    suggestions = _generate_suggestions(feats, labels, scores)
    progress.progress(95)

    # ⑤ SALIDA
    status_box.success("⑤ ¡Listo! Análisis finalizado.")
    progress.progress(100)
    if model_path:
        st.success(f"✅ Modelo cargado: **{os.path.basename(model_path)}**")
    st.success(f"✅ Backend: **{results.get('backend')}** · Frames analizados: **{results.get('n_frames')}**")

    colA, colB = st.columns([1, 1], gap="large")

    with colA:
        st.subheader("Etiquetas y puntuaciones (modelo)")
        if model_ok and labels:
            lbltbl = [{"label": l, "score": round(float(s), 3)} for l, s in zip(labels, scores)]
            st.table(lbltbl)
        elif not model_ok:
            st.info("Se omitieron etiquetas del modelo por incidencia. Revisa artifacts y versiones.")
        else:
            st.write("—")

        st.subheader("Probabilidades por clase")
        st.json({k: round(float(v), 4) for k, v in (prob_map or {}).items()} if prob_map else {}, expanded=False)

        st.subheader("Rasgos (features)")
        st.json({k: float(v) for k, v in feats.items()}, expanded=False)

    with colB:
        st.subheader("💡 Sugerencias coreográficas")
        if not suggestions:
            st.info("No se generaron sugerencias (verifica que hay landmarks detectados).")
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

    export = {
        "video": os.path.basename(results.get("video_path", "")),
        "backend": results.get("backend"),
        "n_frames": results.get("n_frames"),
        "features": feats,
        "labels": labels,
        "scores": [float(s) for s in scores],
        "probs": prob_map,
        "model_file": os.path.basename(model_path) if model_path else None,
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

# ============================================================
# Lanzador
# ============================================================
if video_path:
    if st.button("🚀 Ejecutar análisis"):
        _run_pipeline(video_path)
else:
    st.info("📌 Sube un vídeo o graba un clip para comenzar.")
