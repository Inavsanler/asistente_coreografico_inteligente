# ============================================================
# app.py — Asistente Coreográfico (Colab-compat: YOLO Pose + features)
# ============================================================

from __future__ import annotations

import os, json, base64, pickle, traceback, tempfile, importlib.util
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import cv2

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
# Estilos
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
:root{ --pri:#111827; --sec:#374151; --acc:#2563eb; --ok:#059669; --warn:#b45309; --err:#dc2626; }
.main-header{font-size:2.1rem;color:var(--pri);text-align:center;font-weight:700;margin:0.5rem 0 1rem 0}
.card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin:8px 0;background:#ffffff}
.kpi{font-size:1.05rem;font-weight:600}
.small{font-size:.9rem;color:#6b7280}
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:.75rem;margin-left:.5rem}
.sugg{border-left:4px solid var(--acc);padding:.5rem .75rem;margin:.35rem 0;border-radius:8px;background:#f8fafc}
.sugg h4{margin:.2rem 0 .15rem 0;font-size:1rem}
.sugg .why{color:#6b7280;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

# ============================================================
# IMPORT opcional del motor de tu repo (si existe)
# ============================================================
_run_inference_src = None
try:
    from src.inference import run_inference_over_video as _run_inference_src  # type: ignore
except Exception:
    _run_inference_src = None  # usaremos ruta local YOLO si falla

# ============================================================
# UTILIDADES: vídeo
# ============================================================
def _save_uploaded_video_to_tmp(upload) -> str:
    suffix = ".mp4"
    if hasattr(upload, "name") and isinstance(upload.name, str):
        name = upload.name.lower()
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            if name.endswith(ext):
                suffix = ext; break
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

# ============================================================
# COLAB-COMPAT: YOLO Pose -> K (T,17,2) en píxeles
# ============================================================
def _video_to_keypoints_yolo(video_path: str, weights: str = "yolov8n-pose.pt",
                             conf: float = 0.25, stride: int = 1) -> Tuple[np.ndarray, float]:
    """
    Devuelve K: (T,17,2) en píxeles (x,y) con NaNs si no hay detección.
    Selecciona la persona con mayor caja por frame (como en el Colab).
    """
    try:
        from ultralytics import YOLO  # requiere ultralytics instalado
    except Exception as e:
        raise RuntimeError("Ultralytics no está instalado. Instala 'ultralytics torch torchvision'.") from e

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"No se pudo abrir el vídeo: {video_path}"
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if T <= 0:
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok: break
            frames.append(fr)
        T = len(frames)
        cap.release()
        cap = cv2.VideoCapture(video_path)

    model = YOLO(weights)  # autodescarga si pones nombre (p.ej. "yolov8n-pose.pt")
    K = np.full((T, 17, 2), np.nan, dtype=np.float32)
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if t % max(1, stride) == 0:
            res = model.predict(source=frame, conf=conf, iou=0.5, verbose=False)
            if len(res):
                res = res[0]
                if hasattr(res, "keypoints") and res.keypoints is not None and res.boxes is not None and len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                    idx = int(np.argmax(areas))
                    kps = res.keypoints.xy[idx].cpu().numpy()  # (17,2)
                    kps[:,0] = np.clip(kps[:,0], 0, W-1)
                    kps[:,1] = np.clip(kps[:,1], 0, H-1)
                    K[t] = kps
        t += 1
    cap.release()
    return K, FPS

# ============================================================
# Limpieza/interpolación (como en Colab, simplificada)
# ============================================================
def _interpolate_nan_1d(y: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    T = len(y); idx = np.arange(T)
    mask = np.isfinite(y)
    if mask.sum() == 0: return y
    if mask.sum() == 1:
        y[~mask] = y[mask][0]; return y
    y[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
    return y

def _clean_nan_interpolate(K: np.ndarray, min_valid_ratio: float = 0.10) -> Tuple[np.ndarray, List[int]]:
    """
    K: (T,J,2). Mantiene solo articulaciones con >= min_valid_ratio de frames válidos
    e interpola en el tiempo.
    """
    if K is None or len(K) == 0:
        return K, []
    Kc = K.copy()
    T, J, D = Kc.shape
    used = []
    for j in range(J):
        valid = np.isfinite(Kc[:, j, :]).all(axis=1)
        if valid.mean() < min_valid_ratio:
            Kc[:, j, :] = np.nan
            continue
        for d in range(D):
            Kc[:, j, d] = _interpolate_nan_1d(Kc[:, j, d])
        used.append(j)
    if not used:
        return Kc, []
    return Kc[:, used, :], used

# ============================================================
# Features del Colab
# ============================================================
def _center_of_mass(K: np.ndarray) -> np.ndarray:
    # caderas 11 y 12 (COCO); fallback promedio
    if K.shape[1] >= 13 and np.all(np.isfinite(K[:,11,:])) and np.all(np.isfinite(K[:,12,:])):
        return (K[:,11,:] + K[:,12,:]) / 2.0
    return np.nanmean(K, axis=1)

def features_coreograficos(K: np.ndarray) -> Dict[str, float]:
    """
    Replica las métricas base usadas en tu Colab.
    """
    T, J, D = K.shape
    feats: Dict[str, float] = {}
    # Amplitud global por eje
    amp = (np.nanmax(K, axis=0) - np.nanmin(K, axis=0))
    amp = np.nanmean(amp, axis=0)  # media sobre articulaciones
    feats["amplitud_x"], feats["amplitud_y"] = float(amp[0]), float(amp[1])
    if D == 3 and K.shape[2] > 2 and np.isfinite(amp[2]): feats["amplitud_z"] = float(amp[2])

    # Velocidad media del COM
    com = _center_of_mass(K)
    disp = np.diff(com, axis=0)
    vel = float(np.nanmean(np.linalg.norm(disp, axis=1))) if disp.size else 0.0
    feats["velocidad_media"] = vel

    # Simetría: pares (5,6),(7,8),(9,10),(11,12),(13,14),(15,16)
    pairs = [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    pdists = []
    for a,b in pairs:
        if a < J and b < J:
            d = np.linalg.norm(K[:,a,:]-K[:,b,:], axis=1)
            pdists.append(np.nanmean(d))
    feats["simetria"] = float(np.nanmean(pdists)) if pdists else 0.0

    # Nivel (p10 - p90 de Y del COM) — ojo, Y hacia abajo en imagen
    y = com[:,1]
    if np.isfinite(y).any():
        p10 = float(np.nanpercentile(y, 10))
        p90 = float(np.nanpercentile(y, 90))
        feats["nivel_rango"] = float(p10 - p90)
    else:
        feats["nivel_rango"] = 0.0

    # Variedad direccional: media de |Δθ|
    if disp.size:
        dirs = np.arctan2(disp[:,1], disp[:,0])
        cambios = np.abs(np.diff(dirs))
        feats["variedad_direcciones"] = float(np.nanmean(cambios)) if cambios.size else 0.0
    else:
        feats["variedad_direcciones"] = 0.0

    feats["frames"] = float(T)
    return feats

# ============================================================
# Fallback pseudo-CV (sin pose): genera features compatibles
# ============================================================
def pseudo_features_from_video(video_path: str, max_frames: Optional[int] = None) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return dict(amplitud_x=0.0, amplitud_y=0.0, velocidad_media=0.0, simetria=0.0,
                    nivel_rango=0.0, variedad_direcciones=0.0, frames=0.0)
    mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
    centers = []
    T = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        T += 1
        if max_frames and T > max_frames: break
        fg = mog.apply(fr)
        fg = cv2.medianBlur(fg, 5)
        ys, xs = np.where(fg > 0)
        if len(xs) > 50:
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            centers.append([cx, cy])
    cap.release()
    if len(centers) < 2:
        return dict(amplitud_x=0.0, amplitud_y=0.0, velocidad_media=0.0, simetria=0.0,
                    nivel_rango=0.0, variedad_direcciones=0.0, frames=float(T))
    C = np.array(centers)
    disp = np.diff(C, axis=0)
    vel = float(np.mean(np.linalg.norm(disp, axis=1)))
    ax = float(np.max(C[:,0]) - np.min(C[:,0]))
    ay = float(np.max(C[:,1]) - np.min(C[:,1]))
    dirs = np.arctan2(disp[:,1], disp[:,0])
    cambios = np.abs(np.diff(dirs))
    variedad = float(np.mean(cambios)) if cambios.size else 0.0
    p10 = float(np.percentile(C[:,1], 10)); p90 = float(np.percentile(C[:,1], 90))
    nivel_rango = float(p10 - p90)
    return dict(amplitud_x=ax, amplitud_y=ay, velocidad_media=vel, simetria=0.0,
                nivel_rango=nivel_rango, variedad_direcciones=variedad, frames=float(T))

# ============================================================
# Modelo — auto-detección (incluye thresholded_bundle)
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
    candidates = [
        "complete_model_thresholded_bundle.joblib",
        "complete_model_thresholded_bundle.pkl",
        "complete_model_thresholded.joblib",
        "complete_model_thresholded.pkl",
        "complete_model_threshold.joblib",
        "complete_model_threshold.pkl",
        "complete_model_thersholder.joblib",
        "complete_model_thersholder.pkl",
        "model.joblib", "model.pkl",
    ]
    for name in candidates:
        p = os.path.join(artifacts_dir, name)
        if os.path.exists(p): return p
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

def _vectorize_features_dict(features: Dict[str, float], feature_order: Optional[List[str]] = None) -> np.ndarray:
    if feature_order is None:
        # Orden por nombres clave usados en el Colab si están presentes:
        prefer = ['amplitud_x','amplitud_y','amplitud_z','velocidad_media','simetria','nivel_rango','variedad_direcciones','frames']
        keys = [k for k in prefer if k in features] + [k for k in sorted(features.keys()) if k not in prefer]
    else:
        keys = list(feature_order)
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

def predict_with_model(features: Dict[str, float], artifacts_dir: str = "artifacts"
) -> Tuple[List[str], List[float], Dict[str, float], Optional[str]]:
    model, meta, model_path = _load_model_and_meta(artifacts_dir)
    if model is None:
        raise RuntimeError("No se encontró un modelo en artifacts/ (p.ej., complete_model_thresholded_bundle.joblib).")
    X = _vectorize_features_dict(features, meta.get("feature_order"))
    probs = None
    try:
        est = model
        if hasattr(est, "predict_proba"):
            probs = est.predict_proba(X)
        elif hasattr(est, "decision_function"):
            df = est.decision_function(X)
            if isinstance(df, np.ndarray): probs = 1/(1+np.exp(-df))
        elif hasattr(est, "predict"):
            y = est.predict(X)
            classes = _get_model_classes(model, meta)
            probs = np.zeros((1, len(classes)), dtype=float)
            try:
                idx = int(y[0]); 
                if 0 <= idx < probs.shape[1]: probs[0, idx] = 1.0
            except Exception:
                for i, c in enumerate(classes):
                    if str(y[0]) == str(c): probs[0, i] = 1.0; break
    except Exception as e:
        st.error("No se pudieron obtener puntuaciones del modelo (comprueba versiones sklearn/joblib).")
        st.code("".join(traceback.format_exception_only(type(e), e)))
        probs = None

    classes = _get_model_classes(model, meta)
    if probs is not None and probs.ndim == 1: probs = probs.reshape(1, -1)
    prob_map: Dict[str, float] = {}
    if probs is not None and probs.shape[1] == len(classes):
        for i, c in enumerate(classes):
            prob_map[c] = float(probs[0, i])

    thresholds = _get_thresholds(classes, meta, default=0.5)
    labels_out: List[str] = []; scores_out: List[float] = []
    if prob_map:
        for c in classes:
            p = prob_map.get(c, 0.0)
            if p >= thresholds.get(c, 0.5): labels_out.append(c); scores_out.append(p)
        if not labels_out:
            # top-1 para no dejar vacío
            best_idx = int(np.argmax([prob_map.get(c, 0.0) for c in classes]))
            labels_out = [classes[best_idx]]; scores_out = [prob_map[classes[best_idx]]]
    return labels_out, scores_out, prob_map, model_path

# ============================================================
# Sugerencias simples (puedes sustituir por las tuyas del repo)
# ============================================================
def generate_suggestions(features: Dict[str, float], labels: List[str], scores: List[float]) -> List[Dict[str, Any]]:
    s = []
    ax = features.get("amplitud_x", 0.0)
    ay = features.get("amplitud_y", 0.0)
    vel = features.get("velocidad_media", 0.0)
    sim = features.get("simetria", 0.0)
    nivel = features.get("nivel_rango", 0.0)
    varD = features.get("variedad_direcciones", 0.0)

    if max(ax, ay) < 60:
        s.append({"title":"Aumentar amplitud espacial","severity":"media",
                  "why":f"Amplitud baja (x≈{ax:.1f}, y≈{ay:.1f}).",
                  "how":"Incorpora diagonales y niveles alto/medio/bajo con transiciones más amplias."})
    if vel < 1.5:
        s.append({"title":"Mayor proyección dinámica","severity":"media",
                  "why":f"Velocidad media baja ({vel:.2f}).",
                  "how":"Añade acentos y aceleraciones puntuales en 8+8 para crear contraste."})
    if varD < 0.25:
        s.append({"title":"Explora direcciones","severity":"baja",
                  "why":f"Variedad direccional limitada ({varD:.2f}).",
                  "how":"Secuencia de giros/fintas en planos frontal y diagonal posterior."})
    if nivel > -20:  # recuerda: y crece hacia abajo, p10-p90 suele ser negativo en ‘subidas’
        s.append({"title":"Trabaja niveles","severity":"baja",
                  "why":f"Rango de nivel escaso ({nivel:.1f}).",
                  "how":"Introduce plié y transiciones al suelo para ampliar el rango vertical."})
    s.append({"title":"Clarifica los remates","severity":"baja",
              "why":"Mejora la legibilidad de las frases.",
              "how":"Define foco y pausa breve (¼ tiempo) al final de cada frase."})
    return s

# ============================================================
# Cámara HTML5 (MediaRecorder) — sin deps extra
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
st.write("Analiza vídeos (subida o **cámara HTML5**) y ejecuta **YOLO Pose + features** como en tu Colab. Carga tu modelo y recibe **etiquetas** y **sugerencias**.")

st.sidebar.header("⚙️ Configuración")
backend = st.sidebar.selectbox("Backend de visión", ["yolo", "mediapipe", "auto"], index=0)
yolo_weights = st.sidebar.text_input("Pesos YOLO Pose", "yolov8n-pose.pt")
with st.sidebar.expander("⏱️ Duración a analizar (del inicio)"):
    target_minutes = st.slider("Minutos", 0.5, 5.0, 3.0, 0.5)
with st.sidebar.expander("⚙️ Avanzado"):
    manual_max_frames = st.checkbox("Fijar manualmente máx. de frames", value=False)
    max_frames_manual_value = st.number_input("Máx. frames", min_value=10, max_value=20000, value=600, step=10)
    artifacts_dir = st.text_input("Carpeta de artifacts", "artifacts")

st.sidebar.info("Modelo esperado en `artifacts/` (p.ej., **complete_model_thresholded_bundle.joblib**). "
                "Opcional: `feature_order.json`, `class_names.json`, `thresholds.json`.")

tab_upload, tab_camera = st.tabs(["📤 Subir vídeo", "🎥 Grabar con cámara (HTML5)"])
video_path: Optional[str] = None
with tab_upload:
    up_video = st.file_uploader("Sube un vídeo (mp4/mov/avi/mkv/webm)", type=["mp4","mov","avi","mkv","webm"])
    if up_video:
        video_path = _save_uploaded_video_to_tmp(up_video)
        st.video(video_path)
with tab_camera:
    st.caption("Graba un **clip** desde tu cámara y úsalo directamente.")
    cam_saved = _camera_recorder_html_ui()
    if cam_saved: video_path = cam_saved

# ============================================================
# PIPELINE
# ============================================================
def _run_pipeline(video_path: str):
    meta = _probe_video(video_path)
    if not meta.get("ok"):
        st.error(f"No se pudo leer el vídeo: {meta.get('reason')}"); st.stop()
    fps, total_frames = meta["fps"], meta["total_frames"]
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

    progress = st.progress(0); status = st.empty()

    # ① INFERENCIA — preferimos YOLO Pose local como en Colab
    status.info("① Ejecutando inferencia de pose…")
    progress.progress(10)
    K: Optional[np.ndarray] = None
    results: Dict[str, Any] = {"backend": backend, "n_frames": max_frames_to_process, "video_path": video_path}
    used_backend = None

    try:
        if backend == "yolo":
            K, _ = _video_to_keypoints_yolo(video_path, weights=yolo_weights, conf=0.25, stride=1)
            used_backend = "yolo"
        elif backend == "mediapipe":
            # si tienes src.inference con mediapipe, úsalo
            if _run_inference_src:
                res = _run_inference_src(video_path, backend="mediapipe", max_frames=max_frames_to_process)
                # intentar convertir a K si vienen keypoints normalizados:
                klist = (res.get("data", {}).get("keypoints") or [])
                if klist:
                    # asumimos (x,y,score) normalizado -> escalar a píxeles aprox
                    T = min(len(klist), max_frames_to_process)
                    K = np.full((T, 17, 2), np.nan, dtype=np.float32)
                    for t in range(T):
                        pose = klist[t].get("pose") or []
                        for j, (xn, yn, *_rest) in enumerate(pose[:17]):
                            K[t, j, 0] = float(xn * width)
                            K[t, j, 1] = float(yn * height)
                used_backend = "mediapipe(src)"
            else:
                raise RuntimeError("Mediapipe no disponible en este entorno.")
        else:  # auto
            try:
                K, _ = _video_to_keypoints_yolo(video_path, weights=yolo_weights, conf=0.25, stride=1)
                used_backend = "yolo"
            except Exception:
                if _run_inference_src:
                    res = _run_inference_src(video_path, backend="mediapipe", max_frames=max_frames_to_process)
                    klist = (res.get("data", {}).get("keypoints") or [])
                    if klist:
                        T = min(len(klist), max_frames_to_process)
                        K = np.full((T, 17, 2), np.nan, dtype=np.float32)
                        for t in range(T):
                            pose = klist[t].get("pose") or []
                            for j, (xn, yn, *_rest) in enumerate(pose[:17]):
                                K[t, j, 0] = float(xn * width)
                                K[t, j, 1] = float(yn * height)
                        used_backend = "mediapipe(src)"
    except Exception as e:
        st.warning(f"El backend seleccionado no está disponible: {e}")

    # Fallback pseudo-CV si no hay K
    if K is None or not np.isfinite(K).any():
        used_backend = f"{backend}→pseudo"
        st.info("No se obtuvo pose. Usaré un fallback rápido sin pose para estimar features.")
    progress.progress(40)

    # Vista previa (si hay frame 0)
    try:
        cap = cv2.VideoCapture(video_path); ret, frame0 = cap.read(); cap.release()
        if ret and frame0 is not None:
            if K is not None and np.isfinite(K).any():
                # dibuja esqueleto simple con algunos pares COCO
                pairs = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,6),(11,12)]
                P = K[min(0, K.shape[0]-1)]
                fr = frame0.copy()
                for a,b in pairs:
                    if a<P.shape[0] and b<P.shape[0] and np.all(np.isfinite(P[a,:2])) and np.all(np.isfinite(P[b,:2])):
                        cv2.line(fr, (int(P[a,0]),int(P[a,1])), (int(P[b,0]),int(P[b,1])), (60,60,240), 2)
                for j in range(min(P.shape[0],17)):
                    x,y = P[j,:2]
                    if np.isfinite(x) and np.isfinite(y):
                        cv2.circle(fr, (int(x),int(y)), 2, (0,200,255), -1)
                st.image(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), caption=f"Vista previa (backend: {used_backend})", use_container_width=True)
            else:
                st.image(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB), caption=f"Vista previa (sin pose · backend: {used_backend})", use_container_width=True)
    except Exception:
        pass

    # ② FEATURES (Colab)
    status.info("② Calculando features (Colab)…")
    if K is not None and np.isfinite(K).any():
        Kc, used = _clean_nan_interpolate(K, min_valid_ratio=0.10)
        feats = features_coreograficos(Kc if used else K)
    else:
        feats = pseudo_features_from_video(video_path, max_frames=max_frames_to_process)
    progress.progress(65)

    # ③ MODELO
    status.info("③ Ejecutando modelo ML (umbralado por clase)…")
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
    status.info("④ Generando sugerencias…")
    suggestions = generate_suggestions(feats, labels, scores)
    progress.progress(95)

    # ⑤ SALIDA
    status.success("⑤ ¡Listo! Análisis finalizado.")
    progress.progress(100)

    if model_path: st.success(f"✅ Modelo cargado: **{os.path.basename(model_path)}**")
    st.success(f"✅ Backend usado: **{used_backend or backend}** · Frames analizados: **{max_frames_to_process}**")

    colA, colB = st.columns([1,1], gap="large")
    with colA:
        st.subheader("Etiquetas y puntuaciones (modelo)")
        if model_ok and labels:
            st.table([{"label": l, "score": round(float(s), 3)} for l, s in zip(labels, scores)])
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
            st.info("No se generaron sugerencias.")
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
        "video": os.path.basename(video_path),
        "backend": used_backend or backend,
        "n_frames": max_frames_to_process,
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

# ============================================================
# Lanzador
# ============================================================
if video_path:
    if st.button("🚀 Ejecutar análisis"):
        _run_pipeline(video_path)
else:
    st.info("📌 Sube un vídeo o graba un clip para comenzar.")
