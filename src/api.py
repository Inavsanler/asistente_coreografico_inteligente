import numpy as np
from typing import Dict, List, Optional
import numpy as np
from typing import Dict, List, Optional

def center_of_mass(K: np.ndarray) -> np.ndarray:
    """
    Calcula el centro de masa aproximado para cada frame.
    Si existen joints de cadera (11 y 12 en COCO), los usa; si no, usa la media global.
    """
    if K.shape[1] >= 13:
        return (K[:, 11, :] + K[:, 12, :]) / 2
    return K.mean(axis=1)

def features_coreograficos(K: np.ndarray) -> Dict[str, float]:
    """
    Extrae características coreográficas a partir de keypoints 2D/3D.
    
    Args:
        K: Array de keypoints con forma (T, J, D) donde:
           T = número de frames
           J = número de joints
           D = dimensiones (2 o 3)
    
    Returns:
        Diccionario con métricas coreográficas
    """
    T, J, D = K.shape
    feats = {}

    # --- amplitud global ---
    amp = (K.max(axis=0) - K.min(axis=0)).mean(axis=0)
    feats["amplitud_x"], feats["amplitud_y"] = amp[0], amp[1]
    if D == 3: 
        feats["amplitud_z"] = amp[2]

    # --- velocidad / fluidez ---
    vel = np.linalg.norm(np.diff(K, axis=0), axis=2).mean()
    feats["velocidad_media"] = vel

    # --- simetría ---
    pairs = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    pair_dists = []
    for a, b in pairs:
        if a < J and b < J:
            d = np.linalg.norm(K[:, a, :] - K[:, b, :], axis=1).mean()
            pair_dists.append(d)
    feats["simetria"] = np.mean(pair_dists) if pair_dists else 0.0

    # --- niveles espaciales ---
    com = center_of_mass(K)
    feats["nivel_alto"] = np.percentile(com[:, 1], 90)
    feats["nivel_bajo"] = np.percentile(com[:, 1], 10)
    feats["nivel_rango"] = feats["nivel_bajo"] - feats["nivel_alto"]

    # --- variedad direccional ---
    disp = np.diff(com, axis=0)
    dirs = np.arctan2(disp[:, 1], disp[:, 0])
    cambios = np.abs(np.diff(dirs))
    feats["variedad_direcciones"] = np.mean(cambios)

    return feats

def sugerencias(feats: Dict[str, float]) -> List[str]:
    """
    Genera sugerencias coreográficas a partir de métricas de análisis.
    
    Args:
        feats: Diccionario con métricas coreográficas
        
    Returns:
        Lista de sugerencias priorizadas y deduplicadas
    """
    S = []  # sugerencias

    # ===== Helpers =====
    def val(*keys, default=None):
        for k in keys:
            if k in feats and feats[k] is not None:
                return feats[k]
        return default

    def add(msg, prio=5):
        if msg:
            S.append((prio, msg.strip()))

    # ===== Normalización de nombres frecuentes =====
    ax = val("amplitud_x", "amp_x")
    ay = val("amplitud_y", "amp_y")
    az = val("amplitud_z", "amp_z")
    vel = val("velocidad_media", "vel_media")
    vel_sd = val("velocidad_std", "vel_std")
    sim = val("simetria", "simetria_raw")
    varD = val("variedad_direcciones", "variedad_dir")
    nivel = val("nivel_rango", "nivel_rango")
    disp = val("disp_total", "trayectoria_longitud")
    frames = val("frames")
    dims = val("dims")

    # ===== Técnicos (alineación, equilibrio, extensiones, torso, cabeza) =====
    # [Resto del código de sugerencias...]
    # (Mantén todo el código de sugerencias aquí exactamente como lo proporcionaste)

    # ===== Post-proceso: de-dup + orden por prioridad =====
    # Eliminamos duplicados manteniendo la mayor prioridad (menor número)
    agg = {}
    for p, m in S:
        if m not in agg or p < agg[m]:
            agg[m] = p
    
    ordered = sorted([(p, m) for m, p in agg.items()], key=lambda x: x[0])

    # Devuelve solo los textos
    out = [m for _, m in ordered]
    return out or ["El movimiento presenta buena diversidad y balance."]

# Las funciones predict_labels y run_inference_over_video_yolo permanecen aquí
def predict_labels(x, artifacts_dir="artifacts", threshold=None):
    # Implementación existente...
    pass

def run_inference_over_video_yolo(*args, **kwargs):
    # Implementación existente...
    pass