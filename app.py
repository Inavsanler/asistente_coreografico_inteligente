# ============================================================
# app.py — Asistente Coreográfico (réplica Colab: YOLO Pose → Features → Modelo)
# ============================================================

from __future__ import annotations
import os, json, base64, pickle, traceback, tempfile, importlib.util
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import cv2

st.set_page_config(page_title="Asistente Coreográfico Inteligente", layout="wide", page_icon="🎭")

# ---- Estilos
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
:root{ --pri:#111827; --sec:#374151; --acc:#2563eb; }
.main-header{font-size:2.0rem;color:var(--pri);text-align:center;font-weight:700;margin:.5rem 0 1rem}
.card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin:8px 0;background:#fff}
.kpi{font-size:1.05rem;font-weight:600}
.small{font-size:.9rem;color:#6b7280}
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:.75rem;margin-left:.5rem}
.sugg{border-left:4px solid var(--acc);padding:.5rem .75rem;margin:.35rem 0;border-radius:8px;background:#f8fafc}
.sugg h4{margin:.2rem 0 .15rem 0;font-size:1rem}
.sugg .why{color:#6b7280;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Utilidades vídeo
# ============================================================
def _save_uploaded_video_to_tmp(upload) -> str:
    suffix = ".mp4"
    if getattr(upload, "name", None):
        name = upload.name.lower()
        for ext in (".mp4",".mov",".avi",".mkv",".webm"):
            if name.endswith(ext): suffix = ext; break
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.read() if hasattr(upload,"read") else upload.getvalue())
    tmp.flush()
    return tmp.name

def _probe_video(path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap or not cap.isOpened():
        return {"ok": False, "reason": f"No se pudo abrir el vídeo: {path}"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    dur = total / fps if fps > 0 else 0.0
    cap.release()
    return {"ok": True, "fps": float(fps), "total_frames": total, "width": W, "height": H, "duration_s": dur}

def _nice_time(s: float) -> str:
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h>0 else f"{m:02d}:{s:02d}"

def _estimate_frames_for_minutes(fps: float, minutes: float) -> int:
    return int(round((fps if fps>0 else 25.0) * 60.0 * minutes))

# ============================================================
# YOLO Pose (igual que en Colab)
# ============================================================
def _video_to_keypoints_yolo(video_path: str, weights_path: str, conf: float = 0.25, stride: int = 1
) -> Tuple[np.ndarray, float]:
    """
    Devuelve K: (T,17,2) en píxeles (x,y) con NaNs si no hay detección.
    Selecciona la persona con mayor bbox por frame.
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics no está instalado. Añade 'ultralytics torch torchvision' a requirements.") from e

    # Prioriza pesos locales (entornos sin internet)
    if os.path.exists(weights_path):
        model = YOLO(weights_path)
    else:
        # Nombre "yolov8n-pose.pt" → autodescarga (requiere internet)
        model = YOLO(weights_path)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"No se pudo abrir el vídeo: {video_path}"
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
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

    K = np.full((T, 17, 2), np.nan, dtype=np.float32)
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if t % max(1, stride) == 0:
            res = model.predict(source=frame, conf=conf, iou=0.5, verbose=False)
            if len(res):
                r = res[0]
                if getattr(r, "keypoints", None) is not None and getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    idx = int(np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])))
                    kps = r.keypoints.xy[idx].cpu().numpy()  # (17,2)
                    kps[:,0] = np.clip(kps[:,0], 0, W-1); kps[:,1] = np.clip(kps[:,1], 0, H-1)
                    K[t] = kps
        t += 1
    cap.release()
    return K, FPS

# ============================================================
# Limpieza/interpolación + Features (como en el Colab)
# ============================================================
def _interpolate_nan_1d(y: np.ndarray) -> np.ndarray:
    y = y.astype(float); T = len(y); idx = np.arange(T)
    mask = np.isfinite(y)
    if mask.sum() == 0: return y
    if mask.sum() == 1: y[~mask] = y[mask][0]; return y
    y[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
    return y

def _clean_nan_interpolate(K: np.ndarray, min_valid_ratio: float = 0.10) -> Tuple[np.ndarray, List[int]]:
    if K is None or len(K) == 0: return K, []
    Kc = K.copy(); T, J, D = Kc.shape; used=[]
    for j in range(J):
        valid = np.isfinite(Kc[:, j, :]).all(axis=1)
        if valid.mean() < min_valid_ratio:
            Kc[:, j, :] = np.nan; continue
        for d in range(D):
            Kc[:, j, d] = _interpolate_nan_1d(Kc[:, j, d])
        used.append(j)
    return (Kc[:, used, :], used) if used else (Kc, [])

def _center_of_mass(K: np.ndarray) -> np.ndarray:
    if K.shape[1] >= 13 and np.isfinite(K[:,11,:]).all() and np.isfinite(K[:,12,:]).all():
        return (K[:,11,:] + K[:,12,:]) / 2.0
    return np.nanmean(K, axis=1)

def features_coreograficos(K: np.ndarray) -> Dict[str, float]:
    T, J, D = K.shape; feats: Dict[str,float] = {}
    amp = np.nanmean(np.nanmax(K, axis=0) - np.nanmin(K, axis=0), axis=0)
    feats["amplitud_x"] = float(amp[0]); feats["amplitud_y"] = float(amp[1])
    if D>2 and np.isfinite(amp[2]): feats["amplitud_z"] = float(amp[2])
    COM = _center_of_mass(K)
    disp = np.diff(COM, axis=0)
    feats["velocidad_media"] = float(np.nanmean(np.linalg.norm(disp, axis=1))) if disp.size else 0.0
    pairs = [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    pd = []
    for a,b in pairs:
        if a<J and b<J:
            d = np.linalg.norm(K[:,a,:]-K[:,b,:], axis=1)
            pd.append(np.nanmean(d))
    feats["simetria"] = float(np.nanmean(pd)) if pd else 0.0
    y = COM[:,1]
    if np.isfinite(y).any():
        p10 = float(np.nanpercentile(y, 10)); p90 = float(np.nanpercentile(y, 90))
        feats["nivel_rango"] = float(p10 - p90)
    else:
        feats["nivel_rango"] = 0.0
    if disp.size:
        dirs = np.arctan2(disp[:,1], disp[:,0]); cambios = np.abs(np.diff(dirs))
        feats["variedad_direcciones"] = float(np.nanmean(cambios)) if cambios.size else 0.0
    else:
        feats["variedad_direcciones"] = 0.0
    feats["frames"] = float(T)
    return feats

# ============================================================
# Modelo (auto-detección; prioriza thresholded_bundle)
# ============================================================
def _load_json_if_exists(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        pass
    return None

def _find_model_file(artifacts_dir: str) -> Optional[str]:
    pref = [
        "complete_model_thresholded_bundle.joblib",
        "complete_model_thresholded_bundle.pkl",
        "complete_model_thresholded.joblib",
        "complete_model_thresholded.pkl",
        "complete_model_threshold.joblib",
        "complete_model_threshold.pkl",
    ]
    for n in pref:
        p = os.path.join(artifacts_dir, n)
        if os.path.exists(p): return p
    # fallback: primer joblib/pkl
    for f in os.listdir(artifacts_dir or "."):
        if f.lower().endswith((".joblib",".pkl")): return os.path.join(artifacts_dir, f)
    return None

def _load_model_and_meta(artifacts_dir: str):
    path = _find_model_file(artifacts_dir)
    model = None
    if path and path.lower().endswith(".joblib"):
        try:
            import joblib; model = joblib.load(path)
        except Exception as e:
            st.error(f"No se pudo cargar `{os.path.basename(path)}` (joblib).")
            st.code("".join(traceback.format_exception_only(type(e), e)))
    elif path and path.lower().endswith(".pkl"):
        try:
            with open(path, "rb") as f: model = pickle.load(f)
        except Exception as e:
            st.error(f"No se pudo cargar `{os.path.basename(path)}` (pickle).")
            st.code("".join(traceback.format_exception_only(type(e), e)))
    meta: Dict[str, Any] = {}
    fo = _load_json_if_exists(os.path.join(artifacts_dir, "feature_order.json"))
    if isinstance(fo, list): meta["feature_order"] = [str(x) for x in fo]
    cn = _load_json_if_exists(os.path.join(artifacts_dir, "class_names.json"))
    if isinstance(cn, list): meta["classes"] = [str(x) for x in cn]
    th = _load_json_if_exists(os.path.join(artifacts_dir, "thresholds.json"))
    if isinstance(th, dict): meta["thresholds"] = {str(k): float(v) for k, v in th.items() if isinstance(v,(int,float))}
    return model, meta, path

def _vectorize(features: Dict[str,float], feature_order: Optional[List[str]]) -> np.ndarray:
    if feature_order:
        keys = list(feature_order)
    else:
        prefer = ['amplitud_x','amplitud_y','amplitud_z','velocidad_media','simetria','nivel_rango','variedad_direcciones','frames']
        keys = [k for k in prefer if k in features] + [k for k in sorted(features.keys()) if k not in prefer]
    return np.array([[float(features.get(k,0.0)) for k in keys]], dtype=np.float64)

def _get_model_classes(model, meta: Dict[str,Any]) -> List[str]:
    if isinstance(meta.get("classes"), list) and meta["classes"]:
        return [str(c) for c in meta["classes"]]
    classes = getattr(getattr(model,"named_steps",model), "classes_", None) or getattr(model,"classes_", None)
    if classes is not None and len(classes): return [str(c) for c in list(classes)]
    return ["Clase_0","Clase_1"]

def _get_thresholds(classes: List[str], meta: Dict[str,Any], default: float=0.5) -> Dict[str,float]:
    out = {c: default for c in classes}
    if isinstance(meta.get("thresholds"), dict):
        for c,v in meta["thresholds"].items():
            try: out[str(c)] = float(v)
            except Exception: pass
    return out

def predict_with_model(features: Dict[str,float], artifacts_dir: str="artifacts"
) -> Tuple[List[str], List[float], Dict[str,float], Optional[str]]:
    model, meta, model_path = _load_model_and_meta(artifacts_dir)
    if model is None:
        raise RuntimeError("No se encontró un modelo en artifacts/ (p.ej., complete_model_thresholded_bundle.joblib).")
    X = _vectorize(features, meta.get("feature_order"))
    probs = None
    try:
        est = model
        if hasattr(est,"predict_proba"):
            probs = est.predict_proba(X)
        elif hasattr(est,"decision_function"):
            df = est.decision_function(X)
            if isinstance(df,np.ndarray): probs = 1/(1+np.exp(-df))
        elif hasattr(est,"predict"):
            y = est.predict(X)
            classes = _get_model_classes(model,meta)
            probs = np.zeros((1,len(classes)),dtype=float)
            try:
                idx = int(y[0]); 
                if 0<=idx<probs.shape[1]: probs[0,idx]=1.0
            except Exception:
                for i,c in enumerate(classes):
                    if str(y[0])==str(c): probs[0,i]=1.0; break
    except Exception as e:
        st.error("No se pudieron obtener puntuaciones del modelo (comprueba versiones sklearn/joblib).")
        st.code("".join(traceback.format_exception_only(type(e), e)))
        probs = None

    classes = _get_model_classes(model, meta)
    if probs is not None and probs.ndim==1: probs = probs.reshape(1,-1)
    prob_map: Dict[str,float] = {}
    if probs is not None and probs.shape[1]==len(classes):
        for i,c in enumerate(classes): prob_map[c] = float(probs[0,i])

    th = _get_thresholds(classes, meta, default=0.5)
    labels, scores = [], []
    if prob_map:
        for c in classes:
            p = prob_map.get(c,0.0)
            if p >= th.get(c,0.5): labels.append(c); scores.append(p)
        if not labels:
            best = int(np.argmax([prob_map.get(c,0.0) for c in classes]))
            labels, scores = [classes[best]], [prob_map[classes[best]]]
    return labels, scores, prob_map, model_path

# ============================================================
# Sugerencias simples (puedes reemplazarlas por las tuyas)
# ============================================================
def generate_suggestions(features: Dict[str,float], labels: List[str], scores: List[float]) -> List[Dict[str,Any]]:
    s = []
    ax, ay = features.get("amplitud_x",0.0), features.get("amplitud_y",0.0)
    vel = features.get("velocidad_media",0.0)
    varD = features.get("variedad_direcciones",0.0)
    nivel = features.get("nivel_rango",0.0)
    if max(ax,ay) < 60:
        s.append({"title":"Aumenta amplitud espacial","severity":"media",
                  "why":f"Amplitud baja (x≈{ax:.1f}, y≈{ay:.1f}).",
                  "how":"Añade diagonales y niveles alto/medio/bajo con transiciones amplias."})
    if vel < 1.5:
        s.append({"title":"Proyección dinámica","severity":"media",
                  "why":f"Velocidad media baja ({vel:.2f}).",
                  "how":"Introduce acentos y aceleraciones en 8+8 para contraste."})
    if varD < 0.25:
        s.append({"title":"Explora direcciones","severity":"baja",
                  "why":f"Variedad direccional limitada ({varD:.2f}).",
                  "how":"Secuencia de giros/fintas en frontal y diagonal posterior."})
    if nivel > -20:
        s.append({"title":"Trabaja niveles","severity":"baja",
                  "why":f"Rango vertical escaso ({nivel:.1f}).",
                  "how":"Incluye plié y transiciones al suelo para ampliar rango."})
    s.append({"title":"Clarifica remates","severity":"baja",
              "why":"Mejora la legibilidad de las frases.",
              "how":"Pausa de ¼ tiempo y foco final en cada frase."})
    return s

# ============================================================
# Cámara HTML5 (sin dependencias)
# ============================================================
def _camera_recorder_html_ui() -> Optional[str]:
    html = """
    <div style="font-family:system-ui,Segoe UI,Roboto,Arial">
      <video id="preview" autoplay playsinline style="width:100%;max-height:260px;background:#000;border-radius:12px"></video>
      <div style="margin:.5rem 0;display:flex;gap:.5rem;flex-wrap:wrap">
        <button id="startBtn">⏺️ Comenzar</button>
        <button id="stopBtn" disabled>⏹️ Detener</button>
        <button id="useBtn" disabled>💾 Usar clip</button>
        <label style="margin-left:auto">Duración máx. (s): <input id="maxSec" type="number" min="5" max="60" value="15" style="width:4rem"></label>
        <label style="margin-left:.5rem">FPS: <input id="fps" type="number" min="10" max="30" value="24" style="width:4rem"></label>
      </div>
      <div id="note" style="color:#6b7280;font-size:.9rem">Permite la cámara. Se grabará en <b>WebM</b>.</div>
      <script>
        const video = document.getElementById('preview');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const useBtn = document.getElementById('useBtn');
        const maxSec = document.getElementById('maxSec');
        const fps = document.getElementById('fps');
        let mediaStream=null, mediaRecorder=null, chunks=[], timer=null;
        function postValue(val){ window.parent.postMessage({type:'streamlit:setComponentValue', value: val}, '*'); }
        function componentReady(){ window.parent.postMessage({type:'streamlit:componentReady', value:true}, '*'); }
        componentReady();
        async function init(){ try{ mediaStream = await navigator.mediaDevices.getUserMedia({video:true,audio:false}); video.srcObject = mediaStream; }catch(e){ document.getElementById('note').innerText='No se pudo acceder a la cámara: '+e; } }
        init();
        startBtn.onclick = () => {
          if (!mediaStream) return;
          chunks = [];
          let mime='video/webm;codecs=vp9'; if (!MediaRecorder.isTypeSupported(mime)) mime='video/webm;codecs=vp8'; if (!MediaRecorder.isTypeSupported(mime)) mime='video/webm';
          try{ mediaRecorder = new MediaRecorder(mediaStream, { mimeType: mime }); }catch(e){ document.getElementById('note').innerText='MediaRecorder no soportado.'; return; }
          mediaRecorder.ondataavailable = ev => { if (ev.data && ev.data.size) chunks.push(ev.data); };
          mediaRecorder.onstop = () => { stopBtn.disabled = true; useBtn.disabled = chunks.length === 0; };
          mediaRecorder.start(Math.max(1000/parseInt(fps.value||'24'), 200));
          startBtn.disabled = true; stopBtn.disabled = false; useBtn.disabled = true;
          const limit = parseInt(maxSec.value || '15'); if (timer) clearTimeout(timer);
          timer = setTimeout(()=>{ try{ mediaRecorder.stop(); }catch{} }, limit*1000);
        };
        stopBtn.onclick = () => { try{ mediaRecorder && mediaRecorder.stop(); }catch{}; startBtn.disabled=false; stopBtn.disabled=true; };
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
        with open(tmp_path,"wb") as f: f.write(base64.b64decode(b64))
        st.success(f"✅ Clip guardado: {os.path.basename(tmp_path)}"); st.video(tmp_path)
        return tmp_path
    return None

# ============================================================
# UI
# ============================================================
st.markdown("<div class='main-header'>🎭 Asistente Coreográfico Inteligente</div>", unsafe_allow_html=True)
st.write("Flujo **Colab**: YOLO Pose → *features* del notebook → **modelo** (thresholded bundle) → etiquetas y sugerencias.")

st.sidebar.header("⚙️ Configuración")
colab_strict = st.sidebar.checkbox("Modo Colab estricto (sin fallbacks)", value=True)
weights_default = "artifacts/yolov8n-pose.pt" if os.path.exists("artifacts/yolov8n-pose.pt") else "yolov8n-pose.pt"
yolo_weights = st.sidebar.text_input("Pesos YOLO Pose", weights_default)
target_minutes = st.sidebar.slider("Minutos a analizar (desde el inicio)", 0.5, 5.0, 3.0, 0.5)
manual_max_frames = st.sidebar.checkbox("Fijar máx. frames manual", value=False)
max_frames_manual_value = st.sidebar.number_input("Máx. frames", min_value=10, max_value=20000, value=600, step=10)
artifacts_dir = st.sidebar.text_input("Carpeta de artifacts", "artifacts")
st.sidebar.info("Modelo esperado: **complete_model_thresholded_bundle.joblib** (o .pkl) en `artifacts/`.\n"
                "Opcional: `feature_order.json`, `class_names.json`, `thresholds.json`.")

tab_upload, tab_camera = st.tabs(["📤 Subir vídeo", "🎥 Grabar con cámara (HTML5)"])
video_path: Optional[str] = None
with tab_upload:
    upv = st.file_uploader("Vídeo (mp4/mov/avi/mkv/webm)", type=["mp4","mov","avi","mkv","webm"])
    if upv:
        video_path = _save_uploaded_video_to_tmp(upv)
        st.video(video_path)
with tab_camera:
    st.caption("Graba un clip y úsalo directamente.")
    cam_saved = _camera_recorder_html_ui()
    if cam_saved: video_path = cam_saved

# ============================================================
# Pipeline
# ============================================================
def _run_pipeline(video_path: str):
    meta = _probe_video(video_path)
    if not meta.get("ok"):
        st.error(f"No se pudo leer el vídeo: {meta.get('reason')}"); st.stop()

    fps, total = meta["fps"], meta["total_frames"]
    dur = meta["duration_s"]; W, H = meta["width"], meta["height"]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c = st.columns(4)
    c[0].markdown(f"<div class='kpi'>FPS</div><div class='small'>{fps:.2f}</div>", unsafe_allow_html=True)
    c[1].markdown(f"<div class='kpi'>Frames</div><div class='small'>{total}</div>", unsafe_allow_html=True)
    c[2].markdown(f"<div class='kpi'>Duración</div><div class='small'>{_nice_time(dur)} ({dur:.1f}s)</div>", unsafe_allow_html=True)
    c[3].markdown(f"<div class='kpi'>Resolución</div><div class='small'>{W}×{H}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    frames_by_minutes = _estimate_frames_for_minutes(fps, target_minutes)
    max_frames = int(max_frames_manual_value) if manual_max_frames else int(min(frames_by_minutes, total))
    st.markdown(f"**Se analizarán ~{max_frames} frames** (≈ {_nice_time(max_frames / (fps or 25))}).")

    progress = st.progress(0); status = st.empty()

    # ① YOLO Pose (obligatorio en modo estricto)
    status.info("① Ejecutando YOLO Pose…")
    progress.progress(15)
    K = None; used_backend = "yolo"
    try:
        K, _ = _video_to_keypoints_yolo(video_path, weights_path=yolo_weights, conf=0.25, stride=1)
    except Exception as e:
        if colab_strict:
            st.error("❌ El backend YOLO Pose no está disponible. "
                     "Instala las dependencias del Colab o coloca `artifacts/yolov8n-pose.pt`.")
            st.code(str(e))
            st.stop()
        else:
            st.warning(f"YOLO no disponible, continuar (no estricto): {e}")
            K = None; used_backend = "none"

    progress.progress(45)

    # Vista previa (frame 0)
    try:
        cap = cv2.VideoCapture(video_path); ret, f0 = cap.read(); cap.release()
        if ret and f0 is not None and K is not None and np.isfinite(K).any():
            pairs = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,6),(11,12)]
            P = K[min(0, K.shape[0]-1)]
            fr = f0.copy()
            for a,b in pairs:
                if a<P.shape[0] and b<P.shape[0] and np.all(np.isfinite(P[a,:2])) and np.all(np.isfinite(P[b,:2])):
                    cv2.line(fr, (int(P[a,0]),int(P[a,1])), (int(P[b,0]),int(P[b,1])), (60,60,240), 2)
            for j in range(min(P.shape[0],17)):
                x,y = P[j,:2]
                if np.isfinite(x) and np.isfinite(y): cv2.circle(fr, (int(x),int(y)), 2, (0,200,255), -1)
            st.image(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), caption=f"Vista previa (backend: {used_backend})", use_container_width=True)
        elif ret and f0 is not None:
            st.image(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB), caption=f"Vista previa (sin pose · backend: {used_backend})", use_container_width=True)
    except Exception:
        pass

    # ② Features (Colab)
    status.info("② Calculando *features* (notebook)…")
    if K is None or not np.isfinite(K).any():
        # En estricto, ya habríamos parado; aquí solo si no estricto
        st.warning("Sin pose. El modelo puede no responder como en el Colab.")
        feats = {"amplitud_x":0.0,"amplitud_y":0.0,"velocidad_media":0.0,"simetria":0.0,"nivel_rango":0.0,"variedad_direcciones":0.0,"frames":0.0}
    else:
        Kc, used = _clean_nan_interpolate(K, min_valid_ratio=0.10)
        feats = features_coreograficos(Kc if used else K)
    progress.progress(70)

    # ③ Modelo
    status.info("③ Ejecutando modelo (thresholded bundle)…")
    try:
        labels, scores, prob_map, model_path = predict_with_model(feats, artifacts_dir=artifacts_dir)
        model_ok = True
    except Exception as e:
        model_ok = False; labels=[]; scores=[]; prob_map={}
        st.error("No fue posible ejecutar el modelo (¿archivo no encontrado o incompatible?).")
        st.code("".join(traceback.format_exception_only(type(e), e)))
    progress.progress(90)

    # ④ Sugerencias
    status.info("④ Generando sugerencias…")
    suggestions = generate_suggestions(feats, labels, scores)
    progress.progress(100); status.success("¡Listo!")

    if model_ok and model_path:
        st.success(f"✅ Modelo cargado: **{os.path.basename(model_path)}**")
    st.success(f"✅ Backend usado: **{used_backend}** · Frames analizados: **{max_frames}**")

    colA, colB = st.columns([1,1], gap="large")
    with colA:
        st.subheader("Etiquetas y puntuaciones (modelo)")
        if model_ok and labels:
            st.table([{"label": l, "score": round(float(s), 3)} for l, s in zip(labels, scores)])
        elif not model_ok:
            st.info("Se omitieron etiquetas del modelo por incidencia.")
        else:
            st.write("—")

        st.subheader("Probabilidades por clase")
        st.json({k: round(float(v), 4) for k, v in (prob_map or {}).items()} if prob_map else {}, expanded=False)

        st.subheader("Rasgos (features)")
        st.json({k: float(v) for k, v in feats.items()}, expanded=False)

    with colB:
        st.subheader("💡 Sugerencias coreográficas")
        if suggestions:
            for s in suggestions:
                st.markdown(
                    f"<div class='sugg'>"
                    f"<h4>• {s.get('title','Sugerencia')}</h4>"
                    f"<div class='why'><b>Motivo:</b> {s.get('why','')}</div>"
                    f"<div><b>Cómo aplicarlo:</b> {s.get('how','')}</div>"
                    f"<div class='small'>Severidad: <span class='badge'>{s.get('severity','—')}</span></div>"
                    f"</div>", unsafe_allow_html=True
                )
        else:
            st.write("—")

    export = {
        "video": os.path.basename(video_path),
        "backend": used_backend,
        "n_frames": max_frames,
        "features": feats,
        "labels": labels,
        "scores": [float(s) for s in scores],
        "probs": prob_map,
        "suggestions": suggestions,
        "model_file": os.path.basename(model_path) if model_ok and model_path else None,
    }
    st.download_button("⬇️ Descargar reporte (JSON)",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name="reporte_coreografico.json",
        mime="application/json",
    )

# ============================================================
# Lanzador
# ============================================================
if video_path:
    if st.button("🚀 Ejecutar análisis (modo Colab)"):
        _run_pipeline(video_path)
else:
    st.info("📌 Sube un vídeo o graba un clip para comenzar.")
