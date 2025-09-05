# ============================================================
# app.py — Asistente Coreográfico (Colab-compat ▸ YOLO Pose → Features → Modelo + Keyframes & Reporte)
# ============================================================

from __future__ import annotations
import os, json, base64, pickle, traceback, tempfile, importlib.util, io
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
:root{ --pri:#111827; --sec:#374151; --acc:#2563eb; --mut:#6b7280; --bg:#f6f7fb;}
.block-container{padding-top:1.2rem;}
.main-header{font-size:2.0rem;color:var(--pri);text-align:center;font-weight:700;margin:.25rem 0 1rem}
.card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin:8px 0;background:#fff}
.kpi{font-size:1.05rem;font-weight:600}
.small{font-size:.9rem;color:var(--mut)}
.badge{display:inline-block;padding:.2rem .5rem;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:.75rem;margin-left:.35rem}
.sugg{border-left:4px solid var(--acc);padding:.6rem .75rem;margin:.45rem 0;border-radius:8px;background:#f8fafc}
.sugg h4{margin:.2rem 0 .15rem 0;font-size:1rem}
.sugg .why{color:var(--mut);font-size:.9rem}
.thumb{border-radius:8px;border:1px solid #e5e7eb}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.hr{height:1px;background:#e5e7eb;margin:.75rem 0}
.caption{color:#6b7280;font-size:.85rem}
.stDownloadButton>button{border-radius:10px}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Utilidades de tiempo y vídeo
# ============================================================
def _nice_time(s: float) -> str:
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h>0 else f"{m:02d}:{s:02d}"

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

def _estimate_frames_for_minutes(fps: float, minutes: float) -> int:
    return int(round((fps if fps>0 else 25.0) * 60.0 * minutes))

def _frame_indices_for_limit(total_frames: int, target_count: int) -> np.ndarray:
    """Devuelve índices aproximadamente equiespaciados para limitar el coste."""
    target_count = max(1, min(target_count, total_frames))
    return np.linspace(0, total_frames-1, num=target_count, dtype=int)

# ============================================================
# Extracción de keypoints (YOLO Pose) con límite real de frames
# ============================================================
def _video_to_keypoints_yolo(video_path: str, weights_path: str, conf: float, frame_indices: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Devuelve:
      - K: (Tsel,17,2) en píxeles (x,y) con NaNs si no hay detección.
      - FPS del vídeo original.
      - t_indices: índices de frame reales usados (para mapear a tiempo).
    Selecciona la persona con mayor bbox por frame.
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics no está instalado. Añade 'ultralytics torch torchvision' a requirements.") from e

    model = YOLO(weights_path)  # path local o nombre (autodescarga si hay internet)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"No se pudo abrir el vídeo: {video_path}"
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    K = np.full((len(frame_indices), 17, 2), np.nan, dtype=np.float32)
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok: continue
        res = model.predict(source=frame, conf=conf, iou=0.5, verbose=False)
        if len(res):
            r = res[0]
            if getattr(r, "keypoints", None) is not None and getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                best = int(np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])))
                kps = r.keypoints.xy[best].cpu().numpy()  # (17,2)
                kps[:,0] = np.clip(kps[:,0], 0, W-1); kps[:,1] = np.clip(kps[:,1], 0, H-1)
                K[i] = kps
    cap.release()
    return K, FPS, frame_indices

# ============================================================
# Limpieza/interpolación + Features + Métricas temporales
# ============================================================
def _interpolate_nan_1d(y: np.ndarray) -> np.ndarray:
    y = y.astype(float); T = len(y); idx = np.arange(T)
    mask = np.isfinite(y)
    if mask.sum() == 0: return y
    if mask.sum() == 1: y[~mask] = y[mask][0]; return y
    y[~mask] = np.interp(idx[~mask], idx[mask], y[mask]); return y

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

def timeseries_metrics(K: Optional[np.ndarray]) -> Dict[str, float]:
    M: Dict[str, float] = {}
    if K is None or not np.isfinite(K).any():
        return { "pause_ratio":0.0, "jerk_mean":0.0, "turn_rate":0.0,
                 "expansion_var":0.0, "left_right_imbalance":0.0, "tempo_cv":0.0 }
    COM = _center_of_mass(K)
    disp = np.diff(COM, axis=0)
    speed = np.linalg.norm(disp, axis=1) if disp.size else np.array([])
    accel = np.diff(speed) if speed.size else np.array([])
    jerk = np.diff(accel) if accel.size else np.array([])

    M["pause_ratio"] = float(np.mean(speed < (np.nanmedian(speed)+1e-6)*0.15)) if speed.size else 0.0
    M["jerk_mean"] = float(np.nanmean(np.abs(jerk))) if jerk.size else 0.0
    if disp.size:
        dirs = np.arctan2(disp[:,1], disp[:,0])
        M["turn_rate"] = float(np.mean(np.abs(np.diff(dirs)) > 0.7))
        try:
            peaks = np.where((speed[1:-1] > speed[:-2]) & (speed[1:-1] > speed[2:]))[0] + 1
            if len(peaks) >= 3:
                intervals = np.diff(peaks)
                M["tempo_cv"] = float(np.std(intervals) / ( np.mean(intervals) + 1e-8 ))
            else:
                M["tempo_cv"] = 1.0
        except Exception:
            M["tempo_cv"] = 1.0
    else:
        M["turn_rate"] = 0.0; M["tempo_cv"] = 1.0

    try:
        areas = []
        for t in range(K.shape[0]):
            P = K[t]
            if np.isfinite(P).any():
                xs = P[:,0]; ys = P[:,1]
                areas.append( (np.nanmax(xs)-np.nanmin(xs)) * (np.nanmax(ys)-np.nanmin(ys)) )
        M["expansion_var"] = float(np.std(areas) / (np.mean(areas)+1e-8)) if areas else 0.0
    except Exception:
        M["expansion_var"] = 0.0

    pairs = [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    diffs = []
    for a,b in pairs:
        if a<K.shape[1] and b<K.shape[1]:
            da = np.linalg.norm(K[:,a,:]-K[:,b,:], axis=1)
            diffs.append(np.nanmean(da))
    M["left_right_imbalance"] = float(np.std(diffs)) if diffs else 0.0

    # Auxiliares para keyframes
    M["_aux_speed"] = speed
    M["_aux_dirs"]  = np.arctan2(disp[:,1], disp[:,0]) if disp.size else np.array([])
    return M

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
    if isinstance(th, dict):
        # robustez por si vienen dicts anidados desde Colab
        def _safe_float(v, default=0.5):
            try:
                if isinstance(v,(int,float,np.integer,np.floating)): return float(v)
                if isinstance(v,str): return float(v.strip())
                if isinstance(v,dict):
                    for k2 in ("thr","value","threshold","umbral","val"):
                        if k2 in v: return _safe_float(v[k2], default)
                return float(default)
            except Exception:
                return float(default)
        meta["thresholds"] = {str(k): _safe_float(v) for k, v in th.items()}
    return model, meta, path

def _vectorize(features: Dict[str,float], feature_order: Optional[List[str]]) -> Tuple[np.ndarray, List[str]]:
    if feature_order:
        keys = list(feature_order)
    else:
        prefer = ['amplitud_x','amplitud_y','amplitud_z','velocidad_media','simetria','nivel_rango','variedad_direcciones','frames']
        keys = [k for k in prefer if k in features] + [k for k in sorted(features.keys()) if k not in prefer]
    X = np.array([[float(features.get(k,0.0)) for k in keys]], dtype=np.float64)
    return X, keys

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

def _proba_from_list(probs_list, n_classes: int) -> Optional[np.ndarray]:
    try:
        arrs = []
        for p in probs_list:
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] == 2:
                arrs.append(p[:,1])
            elif p.ndim == 1:
                arrs.append(p)
            else:
                e = np.exp(p - np.max(p, axis=-1, keepdims=True))
                sm = e / (np.sum(e, axis=-1, keepdims=True) + 1e-8)
                if sm.shape[1] >= 2: arrs.append(sm[:,1])
                else: arrs.append(sm[:,0])
        return np.stack(arrs, axis=1).reshape(1, n_classes)
    except Exception:
        return None

def predict_with_model(features: Dict[str,float], artifacts_dir: str="artifacts"
) -> Tuple[List[str], List[float], Dict[str,float], Optional[str]]:
    model, meta, model_path = _load_model_and_meta(artifacts_dir)
    if model is None:
        raise RuntimeError("No se encontró un modelo en artifacts/ (p.ej., complete_model_thresholded_bundle.joblib).")

    X, _ = _vectorize(features, meta.get("feature_order"))
    classes = _get_model_classes(model, meta)
    nC = len(classes)

    probs = None
    try:
        est = model
        if hasattr(est,"predict_proba"):
            pr = est.predict_proba(X)
            if isinstance(pr, list):
                probs = _proba_from_list(pr, nC)
            else:
                pr = np.asarray(pr)
                if pr.ndim == 2 and pr.shape[1] in (nC, 2):
                    probs = pr if pr.shape[1] == nC else pr[:,1:].reshape(1,1) if nC==1 else None
                else:
                    probs = pr.reshape(1, -1)
        if probs is None and hasattr(est,"decision_function"):
            df = est.decision_function(X)
            df = np.asarray(df)
            if df.ndim == 1: df = df.reshape(1,-1)
            probs = 1/(1+np.exp(-df))
        if probs is None and hasattr(est,"predict"):
            y = est.predict(X)
            y = np.asarray(y)
            probs = np.zeros((1, nC), dtype=float)
            try:
                if y.ndim==2 and y.shape[1]==nC:
                    probs = y.astype(float)
                else:
                    idx = int(y[0])
                    if 0<=idx<nC: probs[0, idx] = 1.0
            except Exception:
                for i, c in enumerate(classes):
                    if str(y[0]) == str(c): probs[0, i] = 1.0; break
    except Exception as e:
        st.error("No se pudieron obtener puntuaciones del modelo (comprueba versiones sklearn/joblib).")
        st.code("".join(traceback.format_exception_only(type(e), e)))
        probs = None

    if probs is not None and probs.ndim == 1: probs = probs.reshape(1, -1)

    prob_map: Dict[str, float] = {}
    if probs is not None:
        if probs.shape[1] != nC:
            if probs.shape[1] == 1 and nC == 2:
                prob_map[classes[1]] = float(probs[0,0])
                prob_map[classes[0]] = float(1.0 - probs[0,0])
            else:
                for i in range(min(nC, probs.shape[1])):
                    prob_map[classes[i]] = float(probs[0, i])
                for i in range(probs.shape[1], nC):
                    prob_map[classes[i]] = 0.0
        else:
            for i, c in enumerate(classes):
                prob_map[c] = float(probs[0, i])
    else:
        for c in classes: prob_map[c] = 0.0

    th = _get_thresholds(classes, meta, default=0.5)
    labels, scores = [], []
    for c in classes:
        p = prob_map.get(c,0.0)
        if p >= th.get(c,0.5):
            labels.append(c); scores.append(p)
    if not labels:
        best = int(np.argmax([prob_map.get(c,0.0) for c in classes]))
        labels, scores = [classes[best]], [prob_map[classes[best]]]

    return labels, [float(s) for s in scores], prob_map, model_path

# ============================================================
# Sugerencias (con referencia temporal y mapa desde el notebook)
# ============================================================
LABEL_TO_TEXT_FALLBACK = {
    "amplitud_baja":      "Aumentar amplitud (extensiones y desplazamientos más amplios).",
    "variedad_baja":      "Introducir cambios de dirección y diagonales.",
    "mucha_simetria":     "Explorar asimetrías entre izquierda y derecha.",
    "poca_simetria":      "Equilibrar con momentos de simetría.",
    "fluidez_baja":       "Trabajar transiciones para mayor continuidad/fluidez.",
    "poco_rango_niveles": "Usar niveles alto y bajo además del medio.",
}

def _load_label_text_map(artifacts_dir: str) -> dict:
    """Carga mapa etiqueta→texto entrenado como en tu Colab."""
    # 1) suggestions.json (opcional)
    sjson = os.path.join(artifacts_dir, "suggestions.json")
    if os.path.exists(sjson):
        try:
            obj = json.load(open(sjson, "r", encoding="utf-8"))
            if isinstance(obj, dict) and obj:
                return {str(k): str(v) for k,v in obj.items()}
        except Exception:
            pass
    # 2) label_names.csv (si existe) → construye mapa con fallback
    ln = os.path.join(artifacts_dir, "label_names.csv")
    if os.path.exists(ln):
        try:
            import csv
            with open(ln, "r", encoding="utf-8") as f:
                names = [row[0] for row in csv.reader(f) if row]
            auto = {}
            for name in names:
                auto[name] = LABEL_TO_TEXT_FALLBACK.get(name, f"Aplicar pauta entrenada para «{name.replace('_',' ')}».")
            return auto
        except Exception:
            pass
    # 3) Fallback extraído del notebook
    return LABEL_TO_TEXT_FALLBACK.copy()

def _suggestions_from_labels(labels: List[str], scores: List[float], label_text_map: dict, frame_times: List[float], speed: np.ndarray) -> List[Dict[str,Any]]:
    out=[]
    def _time_from_speed():
        if speed is None or (isinstance(speed,np.ndarray) and speed.size==0) or not frame_times:
            return frame_times[len(frame_times)//2] if frame_times else 0.0
        peaks = np.where((speed[1:-1] > speed[:-2]) & (speed[1:-1] > speed[2:]))[0] + 1
        idx = int(peaks[0]) if len(peaks) else int(np.argmax(speed))
        idx = max(0, min(idx, len(frame_times)-1))
        return float(frame_times[idx])
    tref = _time_from_speed()
    for l, sc in zip(labels, scores):
        txt = label_text_map.get(str(l))
        if not txt: continue
        out.append({
            "title": str(l).replace("_"," ").capitalize(),
            "severity":"media",
            "why": f"Etiqueta del modelo: {l} (score {float(sc):.2f}).",
            "how": txt,
            "t": float(tref)
        })
    return out

def generate_suggestions(features: Dict[str,float], labels: List[str], scores: List[float],
                         M: Dict[str,float], frame_times: List[float]) -> List[Dict[str,Any]]:
    s: List[Dict[str,Any]] = []

    ax, ay = features.get("amplitud_x",0.0), features.get("amplitud_y",0.0)
    vel = features.get("velocidad_media",0.0)
    varD = features.get("variedad_direcciones",0.0)
    nivel = features.get("nivel_rango",0.0)

    pause = M.get("pause_ratio",0.0)
    jerk  = M.get("jerk_mean",0.0)
    turnR = M.get("turn_rate",0.0)
    expV  = M.get("expansion_var",0.0)
    lrimb = M.get("left_right_imbalance",0.0)
    tempo = M.get("tempo_cv",1.0)
    speed = M.get("_aux_speed", np.array([]))

    def _time_ref(criteria: str) -> float:
        if speed is None or (isinstance(speed,np.ndarray) and speed.size==0) or not frame_times:
            return frame_times[len(frame_times)//2] if frame_times else 0.0
        if criteria == "pausa":
            idx = int(np.argmin(speed))
        elif criteria == "acento":
            peaks = np.where((speed[1:-1] > speed[:-2]) & (speed[1:-1] > speed[2:]))[0] + 1
            idx = int(peaks[0]) if len(peaks) else int(np.argmax(speed))
        elif criteria == "irregular":
            idx = int(np.argmax(np.abs(np.diff(speed)))) if len(speed) > 1 else int(np.argmax(speed))
        else:
            idx = int(np.argmax(speed))
        idx = max(0, min(idx, len(frame_times)-1))
        return float(frame_times[idx])

    # Reglas genéricas (coherentes con métricas/feats que calculas en Colab)
    if max(ax,ay) < 60:
        s.append({"title":"Aumenta amplitud espacial","severity":"media",
                  "why":f"Amplitud baja (x≈{ax:.1f}, y≈{ay:.1f}).",
                  "how":"Introduce diagonales amplias y transiciones entre niveles alto/medio/bajo.",
                  "t": _time_ref("acento")})
    if vel < 1.5:
        s.append({"title":"Proyección dinámica","severity":"media",
                  "why":f"Velocidad media baja ({vel:.2f}).",
                  "how":"Acentos y aceleraciones puntuales en 8+8 para crear contraste.",
                  "t": _time_ref("acento")})
    if varD < 0.25:
        s.append({"title":"Explora direcciones","severity":"baja",
                  "why":f"Variedad direccional limitada ({varD:.2f}).",
                  "how":"Secuencia de giros y cambios de foco entre frontal/diagonales.",
                  "t": _time_ref("acento")})
    if nivel > -20:
        s.append({"title":"Trabaja niveles","severity":"baja",
                  "why":f"Rango vertical escaso ({nivel:.1f}).",
                  "how":"Incluye plié y bajadas al suelo para ampliar el rango.",
                  "t": _time_ref("acento")})

    if pause > 0.25:
        s.append({"title":"Rellena silencios de movimiento","severity":"media",
                  "why":f"Tiempo en pausa {pause*100:.0f}%.",
                  "how":"Usa micro-transiciones entre frases (desplazamientos cortos, respiración activa).",
                  "t": _time_ref("pausa")})
    if jerk > 0.35:
        s.append({"title":"Suaviza transiciones","severity":"media",
                  "why":f"Jerk medio alto ({jerk:.2f}).",
                  "how":"Añade curvas en trayectorias y anticipos de peso antes de cambios bruscos.",
                  "t": _time_ref("irregular")})
    if turnR < 0.10:
        s.append({"title":"Introduce más cambios de dirección","severity":"baja",
                  "why":f"Pocos giros marcados (rate {turnR:.2f}).",
                  "how":"Inserta pivots ¼-½ vuelta en remates intermedios.",
                  "t": _time_ref("acento")})
    if expV < 0.20:
        s.append({"title":"Varía tamaños corporales","severity":"baja",
                  "why":f"Baja variación de expansión (CV {expV:.2f}).",
                  "how":"Alterna gestos recogidos vs. extendidos cada 2 frases.",
                  "t": _time_ref("acento")})
    if lrimb > 5.0:
        s.append({"title":"Equilibra lateralidad","severity":"media",
                  "why":f"Desequilibrio izq-der ({lrimb:.1f}).",
                  "how":"Duplica la frase en espejo o alterna entradas por ambos lados.",
                  "t": _time_ref("acento")})
    if tempo > 0.35:
        s.append({"title":"Regular el pulso","severity":"media",
                  "why":f"Tempo irregular (CV {tempo:.2f}).",
                  "how":"Trabaja con metrónomo en 8+8 y fija acentos constantes.",
                  "t": _time_ref("irregular")})

    # Sugerencias aprendidas desde el modelo (texto del notebook / artifacts)
    label_text_map = globals().get("_LABEL_TEXT_MAP_RUNTIME", {})
    s += _suggestions_from_labels(labels, scores, label_text_map, frame_times, speed)

    # Remate útil
    s.append({"title":"Clarifica remates","severity":"baja",
              "why":"Mejora la legibilidad de las frases.",
              "how":"Pausa de ¼ tiempo y foco final en cada frase.",
              "t": _time_ref("acento")})
    return s

# ============================================================
# Keyframes (selección + render con esqueleto e IDs)
# ============================================================
_SKELETON_PAIRS = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,6),(11,12)]
_PAIR_COLOR = (60,60,240)    # líneas
_DOT_COLOR  = (0,200,255)    # puntos
_ID_COLOR   = (20,20,20)     # texto índice

def _pick_keyframes(K: np.ndarray, frame_indices: np.ndarray, max_k: int = 12) -> List[int]:
    """Heurística: picos de velocidad, mínimos (pausas), cambios de dirección; relleno uniforme."""
    if K is None or not np.isfinite(K).any() or K.shape[0] == 0:
        return list(frame_indices[:max_k])

    COM = _center_of_mass(K)
    disp = np.diff(COM, axis=0)
    speed = np.linalg.norm(disp, axis=1) if disp.size else np.zeros(0)
    dirs  = np.arctan2(disp[:,1], disp[:,0]) if disp.size else np.zeros(0)
    cand = set()

    if speed.size >= 3:
        peaks = np.where((speed[1:-1] > speed[:-2]) & (speed[1:-1] > speed[2:]))[0] + 1
        for p in peaks[:max_k//3]: cand.add(int(p))
    if speed.size:
        mins = np.argsort(speed)[:max(1, max_k//4)]
        for m in mins: cand.add(int(m))
    if dirs.size >= 2:
        turn = np.abs(np.diff(dirs))
        top_turn = np.argsort(-turn)[:max(1, max_k//4)]
        for t in top_turn: cand.add(int(t+1))

    while len(cand) < min(max_k, len(frame_indices)):
        cand.add(int(np.linspace(0, len(frame_indices)-1, num=min(max_k, len(frame_indices)), dtype=int)[len(cand)]))

    idx_list = sorted(list(cand))[:max_k]
    return [int(frame_indices[i]) for i in idx_list if 0 <= i < len(frame_indices)]

def _draw_pose_on_frame(frame: np.ndarray, P: np.ndarray, show_ids: bool=True, highlight_pairs: bool=True) -> np.ndarray:
    fr = frame.copy()
    if P is None or P.shape[0] < 17:
        return fr
    if highlight_pairs:
        for a,b in _SKELETON_PAIRS:
            if a<P.shape[0] and b<P.shape[0] and np.all(np.isfinite(P[a,:2])) and np.all(np.isfinite(P[b,:2])):
                cv2.line(fr, (int(P[a,0]),int(P[a,1])), (int(P[b,0]),int(P[b,1])), _PAIR_COLOR, 2)
    for j in range(min(P.shape[0],17)):
        x,y = P[j,:2]
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(fr, (int(x),int(y)), 3, _DOT_COLOR, -1)
            if show_ids:
                cv2.putText(fr, str(j), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _ID_COLOR, 1, cv2.LINE_AA)
    return fr

def _grab_frames(video_path: str, abs_indices: List[int]) -> List[np.ndarray]:
    out = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return out
    for idx in abs_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        out.append(fr if ok and fr is not None else None)
    cap.release()
    return out

def _b64_img_rgb(img: np.ndarray, quality: int = 85) -> str:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok: return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")

# ============================================================
# Cámara HTML5 (grabación simple)
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

st.sidebar.header("⚙️ Configuración")
colab_strict = st.sidebar.checkbox("Modo Colab estricto (sin fallbacks)", value=True)
weights_default = "artifacts/yolov8n-pose.pt" if os.path.exists("artifacts/yolov8n-pose.pt") else "yolov8n-pose.pt"
yolo_weights = st.sidebar.text_input("Pesos YOLO Pose", weights_default)
target_minutes = st.sidebar.slider("Minutos a analizar (desde el inicio)", 0.5, 5.0, 3.0, 0.5)
manual_max_frames = st.sidebar.checkbox("Fijar máx. frames manual", value=False)
max_frames_manual_value = st.sidebar.number_input("Máx. frames", min_value=10, max_value=20000, value=600, step=10)
overlay_pose = st.sidebar.checkbox("Sobreponer esqueleto en miniaturas", value=True)
show_kp_ids  = st.sidebar.checkbox("Mostrar índices de keypoints (0–16)", value=True)
highlight_pairs = st.sidebar.checkbox("Resaltar pares simétricos", value=True)
artifacts_dir = st.sidebar.text_input("Carpeta de artifacts", "artifacts")
st.sidebar.info("Modelo esperado: **complete_model_thresholded_bundle.joblib** (o .pkl) en `artifacts/`.\n"
                "Opcional: `feature_order.json`, `class_names.json`, `thresholds.json`, `label_names.csv`, `suggestions.json`.")

tab_upload, tab_camera = st.tabs(["📤 Subir vídeo", "🎥 Grabar con cámara (HTML5)"])
video_path: Optional[str] = None
with tab_upload:
    upv = st.file_uploader("Vídeo (mp4/mov/avi/mkv/webm)", type=["mp4","mov","avi","mkv","webm"])
    if upv:
        video_path = _save_uploaded_video_to_tmp(upv); st.video(video_path)
with tab_camera:
    st.caption("Graba un clip y úsalo directamente.")
    cam_saved = _camera_recorder_html_ui()
    if cam_saved: video_path = cam_saved

# ============================================================
# Pipeline
# ============================================================
def _run_pipeline(video_path: str):
    # Metadata del vídeo
    meta = _probe_video(video_path)
    if not meta.get("ok"): st.error(f"No se pudo leer el vídeo: {meta.get('reason')}"); st.stop()
    fps, total = meta["fps"], meta["total_frames"]
    dur = meta["duration_s"]; W, H = meta["width"], meta["height"]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c = st.columns(4)
    c[0].markdown(f"<div class='kpi'>FPS</div><div class='small'>{fps:.2f}</div>", unsafe_allow_html=True)
    c[1].markdown(f"<div class='kpi'>Frames</div><div class='small'>{total}</div>", unsafe_allow_html=True)
    c[2].markdown(f"<div class='kpi'>Duración</div><div class='small'>{_nice_time(dur)} ({dur:.1f}s)</div>", unsafe_allow_html=True)
    c[3].markdown(f"<div class='kpi'>Resolución</div><div class='small'>{W}×{H}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Límite real de frames
    frames_by_minutes = _estimate_frames_for_minutes(fps, target_minutes)
    max_frames = int(max_frames_manual_value) if manual_max_frames else int(min(frames_by_minutes, total))
    sel_indices = _frame_indices_for_limit(total, max_frames)
    st.markdown(f"**Se analizarán ~{len(sel_indices)} frames** (≈ {_nice_time(len(sel_indices) / (fps or 25))}).")

    progress = st.progress(0); status = st.empty()

    # ① YOLO Pose
    status.info("① Ejecutando YOLO Pose…"); progress.progress(15)
    K = None; used_backend = "yolo"; t_indices_used = sel_indices
    try:
        K, _FPS, t_indices_used = _video_to_keypoints_yolo(video_path, weights_path=yolo_weights, conf=0.25, frame_indices=sel_indices)
    except Exception as e:
        if colab_strict:
            st.error("❌ YOLO Pose no disponible. Instala 'ultralytics torch torchvision' o coloca `artifacts/yolov8n-pose.pt`.")
            st.code(str(e)); st.stop()
        else:
            st.warning(f"YOLO no disponible, continuar (no estricto): {e}")
            K = None; used_backend = "none"; t_indices_used = sel_indices

    progress.progress(45)

    # ② Features + métricas temporales
    status.info("② Calculando *features* y métricas temporales…")
    if K is None or not np.isfinite(K).any():
        st.warning("Sin pose. El modelo puede no responder como en el Colab.")
        feats = {"amplitud_x":0.0,"amplitud_y":0.0,"velocidad_media":0.0,"simetria":0.0,"nivel_rango":0.0,"variedad_direcciones":0.0,"frames":float(len(t_indices_used))}
        metr = {"pause_ratio":0.0,"jerk_mean":0.0,"turn_rate":0.0,"expansion_var":0.0,"left_right_imbalance":0.0,"tempo_cv":1.0,"_aux_speed":np.array([])}
        Kf = None
    else:
        Kc, used = _clean_nan_interpolate(K, min_valid_ratio=0.10)
        Kf = Kc if used else K
        feats = features_coreograficos(Kf)
        metr  = timeseries_metrics(Kf)

    # Mapeo a tiempos (segundos) de los frames seleccionados
    frame_times = [(idx / (fps if fps>0 else 25.0)) for idx in t_indices_used]
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
        # Estructura mínima
        prob_map = {"Clase_0":0.0,"Clase_1":0.0}
    progress.progress(88)

    # ④ Sugerencias con refs temporales (incluye las del modelo entrenado)
    status.info("④ Generando sugerencias…")
    # Carga mapa label->texto desde artifacts (o fallback del notebook)
    global _LABEL_TEXT_MAP_RUNTIME
    _LABEL_TEXT_MAP_RUNTIME = _load_label_text_map(artifacts_dir)
    suggestions = generate_suggestions(feats, labels, scores, metr, frame_times)
    progress.progress(94)

    # ⑤ Keyframes + galería (relación con sugerencias)
    status.info("⑤ Seleccionando y renderizando fotogramas clave…")
    if Kf is not None and np.isfinite(Kf).any():
        key_abs_indices = _pick_keyframes(Kf, t_indices_used, max_k=12)
        raw_frames = _grab_frames(video_path, key_abs_indices)
        thumbs_b64: Dict[int,str] = {}
        # mapa abs_index -> índice relativo en Kf
        rel_map = {abs_i: int(np.where(t_indices_used==abs_i)[0][0]) for abs_i in key_abs_indices if abs_i in t_indices_used}
        for abs_i, fr in zip(key_abs_indices, raw_frames):
            if fr is None: continue
            P = Kf[rel_map[abs_i]] if Kf is not None and rel_map.get(abs_i, None) is not None else None
            img = _draw_pose_on_frame(fr, P, show_ids=show_kp_ids, highlight_pairs=highlight_pairs) if overlay_pose else fr
            thumbs_b64[abs_i] = _b64_img_rgb(img, quality=85)
    else:
        key_abs_indices = list(t_indices_used[:12])
        raw_frames = _grab_frames(video_path, key_abs_indices)
        thumbs_b64 = {abs_i: _b64_img_rgb(fr, quality=85) for abs_i, fr in zip(key_abs_indices, raw_frames) if fr is not None}

    # Relaciona cada sugerencia con el fotograma más cercano a su timecode
    def _closest_abs_index(t_sec: float) -> Optional[int]:
        if not frame_times: return None
        diffs = [abs(t_sec - ft) for ft in frame_times]
        rel = int(np.argmin(diffs))
        return int(t_indices_used[rel])

    for sg in suggestions:
        t = float(sg.get("t", 0.0))
        abs_idx = _closest_abs_index(t)
        sg["frame_abs_index"] = abs_idx
        sg["timecode"] = _nice_time(t)
        if abs_idx in thumbs_b64:
            sg["thumb_b64"] = thumbs_b64[abs_idx]

    progress.progress(100); status.success("¡Listo!")

    if model_ok and model_path: st.success(f"✅ Modelo cargado: **{os.path.basename(model_path)}**")
    st.success(f"✅ Backend usado: **{used_backend}** · Frames analizados: **{len(t_indices_used)}**")

    # ====================== Salidas en UI ======================
    colA, colB = st.columns([1,1], gap="large")

    with colA:
        st.subheader("Etiquetas y puntuaciones (modelo)")
        if model_ok and labels:
            st.table([{"label": l, "score": round(float(s), 3)} for l, s in zip(labels, scores)])
        elif not model_ok:
            st.info("Se omitieron etiquetas del modelo por incidencia.")
        else:
            st.table([])

        st.subheader("Probabilidades por clase")
        st.json({k: round(float(v), 4) for k, v in (prob_map or {}).items()}, expanded=False)

        st.subheader("Rasgos (features)")
        st.json({k: float(v) for k, v in feats.items()}, expanded=False)

        st.subheader("Métricas temporales")
        metr_show = {k: float(v) for k, v in metr.items() if not str(k).startswith("_aux")}
        st.json(metr_show, expanded=False)

    with colB:
        st.subheader("💡 Sugerencias coreográficas (con referencia temporal)")
        if suggestions:
            for sg in suggestions:
                left, right = st.columns([1,3])
                with left:
                    if sg.get("thumb_b64"):
                        st.markdown(f"<img class='thumb' src='data:image/jpeg;base64,{sg['thumb_b64']}' width='150'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='caption'>⏱ {sg.get('timecode','00:00')}</div>", unsafe_allow_html=True)
                with right:
                    st.markdown(
                        f"<div class='sugg'>"
                        f"<h4>• {sg.get('title','Sugerencia')}</h4>"
                        f"<div class='why'><b>Motivo:</b> {sg.get('why','')}</div>"
                        f"<div><b>Cómo aplicarlo:</b> {sg.get('how','')}</div>"
                        f"<div class='small'>Severidad: <span class='badge'>{sg.get('severity','—')}</span></div>"
                        f"</div>", unsafe_allow_html=True
                    )
        else:
            st.write("—")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.subheader("🎞️ Fotogramas clave del análisis")
        if key_abs_indices:
            html_grid = "<div class='grid'>"
            for abs_i in key_abs_indices:
                tsec = (abs_i / (fps if fps>0 else 25.0))
                tc = _nice_time(tsec)
                b64 = thumbs_b64.get(abs_i, "")
                if b64:
                    html_grid += f"<div><img class='thumb' src='data:image/jpeg;base64,{b64}'/><div class='caption'>⏱ {tc} · f#{abs_i}</div></div>"
            html_grid += "</div>"
            st.markdown(html_grid, unsafe_allow_html=True)
        else:
            st.caption("No se pudieron extraer fotogramas clave.")

    # ====================== Exportables ======================
    export = {
        "video": os.path.basename(video_path),
        "backend": used_backend,
        "n_frames_selected": int(len(t_indices_used)),
        "fps": float(fps),
        "features": {k: float(v) for k,v in feats.items()},
        "timeseries_metrics": {k: (float(v) if not isinstance(v, np.ndarray) else None) for k, v in metr.items()},
        "labels": labels,
        "scores": [float(s) for s in scores],
        "probs": {k: float(v) for k,v in (prob_map or {}).items()},
        "suggestions": [
            {k: (v if k not in ("thumb_b64",) else None) for k, v in sg.items()}  # sin imagen en JSON
            for sg in suggestions
        ],
        "model_file": os.path.basename(model_path) if model_ok and model_path else None,
        "keyframes": [
            {"frame": int(abs_i), "time_s": float(abs_i / (fps if fps>0 else 25.0))}
            for abs_i in key_abs_indices
        ],
    }

    st.download_button(
        "⬇️ Descargar reporte (JSON)",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name="reporte_coreografico.json",
        mime="application/json",
    )

    # Reporte HTML legible con miniaturas embebidas
    def _build_html_report() -> bytes:
        def esc(s): 
            try: return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            except: return str(s)
        rows_probs = "".join([f"<tr><td>{esc(k)}</td><td>{float(v):.3f}</td></tr>" for k,v in (prob_map or {}).items()])
        rows_labels = "".join([f"<tr><td>{esc(l)}</td><td>{float(s):.3f}</td></tr>" for l,s in zip(labels, scores)])

        sugg_html = ""
        for sg in suggestions:
            img_html = f"<img style='width:160px;border-radius:8px;border:1px solid #eee' src='data:image/jpeg;base64,{sg.get('thumb_b64','')}'/>" if sg.get("thumb_b64") else ""
            sugg_html += f"""
            <div style="display:flex;gap:12px;margin:10px 0;padding:10px;border:1px solid #eee;border-radius:10px;background:#fafafa">
              <div>{img_html}<div style="color:#666;font-size:12px">⏱ {esc(sg.get('timecode','00:00'))}</div></div>
              <div>
                <div style="font-weight:700">{esc(sg.get('title','Sugerencia'))}
                  <span style="background:#eef2ff;color:#3730a3;border-radius:999px;padding:2px 8px;font-size:12px;margin-left:6px">{esc(sg.get('severity','—'))}</span>
                </div>
                <div style="color:#555;margin-top:2px"><b>Motivo:</b> {esc(sg.get('why',''))}</div>
                <div style="margin-top:2px"><b>Cómo aplicarlo:</b> {esc(sg.get('how',''))}</div>
              </div>
            </div>
            """

        grid_html = "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px'>"
        for abs_i in key_abs_indices:
            b64 = thumbs_b64.get(abs_i, "")
            if not b64: continue
            tsec = (abs_i / (fps if fps>0 else 25.0))
            grid_html += f"<div><img style='width:100%;border-radius:8px;border:1px solid #eee' src='data:image/jpeg;base64,{b64}'/><div style='color:#666;font-size:12px'>⏱ {_nice_time(tsec)} · f#{abs_i}</div></div>"
        grid_html += "</div>"

        html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Reporte Coreográfico</title>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <style>
        body{{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#fff;color:#111827;padding:18px}}
        h1{{font-size:22px;margin:0 0 10px}} h2{{font-size:18px;margin:18px 0 8px}}
        table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #eee;padding:6px 8px;text-align:left}}
        .mut{{color:#6b7280}} .kpi{{display:inline-block;margin-right:16px}}
        </style></head><body>
        <h1>🎭 Asistente Coreográfico — Reporte</h1>
        <div class="mut">Vídeo: {esc(os.path.basename(video_path))} · FPS {fps:.2f} · Duración {_nice_time(dur)} ({dur:.1f}s) · Resolución {W}×{H}</div>

        <h2>Etiquetas y puntuaciones (modelo)</h2>
        <table><thead><tr><th>Etiqueta</th><th>Score</th></tr></thead><tbody>{rows_labels}</tbody></table>

        <h2>Probabilidades por clase</h2>
        <table><thead><tr><th>Clase</th><th>Probabilidad</th></tr></thead><tbody>{rows_probs}</tbody></table>

        <h2>Rasgos (features)</h2>
        <table><tbody>""" + "".join([f"<tr><td>{esc(k)}</td><td>{float(v):.4f}</td></tr>" for k,v in feats.items()]) + """</tbody></table>

        <h2>Métricas temporales</h2>
        <table><tbody>""" + "".join([f"<tr><td>{esc(k)}</td><td>{float(v):.4f}</td></tr>" for k,v in metr_show.items()]) + """</tbody></table>

        <h2>Sugerencias coreográficas (con timecode)</h2>
        """ + sugg_html + """

        <h2>Fotogramas clave</h2>
        """ + grid_html + """

        <div class="mut" style="margin-top:12px">Modelo: """ + (os.path.basename(model_path) if model_ok and model_path else "—") + """</div>
        </body></html>"""
        return html.encode("utf-8")

    html_bytes = _build_html_report()
    st.download_button(
        "🧾 Descargar informe (HTML)",
        data=html_bytes,
        file_name="reporte_coreografico.html",
        mime="text/html",
    )

# ============================================================
# Lanzador
# ============================================================
if video_path:
    if st.button("🚀 Ejecutar análisis "):
        _run_pipeline(video_path)
else:
    st.info("📌 Sube un vídeo o graba un clip para comenzar.")
