# src/features.py
"""
Cálculo de métricas coreográficas a partir de keypoints (T, J, D).
- Funciona con D=2 o 3 (usa solo x,y en algunas métricas).
- Todas las salidas son floats simples (no dicts anidados).
- Añade o adapta nombres para que coincidan con FEATURE_COLS del bundle.
"""

import numpy as np

def _finite_diff(x: np.ndarray):
    """Derivada discreta segura (rellena NaN)"""
    if x.size == 0: return np.array([])
    dx = np.diff(x, axis=0)
    return dx

def _amp(series: np.ndarray):
    if series.size == 0: return 0.0
    m = np.nanmax(series) - np.nanmin(series)
    if not np.isfinite(m): return 0.0
    return float(m)

def _mean_abs(arr: np.ndarray):
    if arr.size == 0: return 0.0
    v = np.nanmean(np.abs(arr))
    return float(v if np.isfinite(v) else 0.0)

def _symmetry_score(kp_xy: np.ndarray, left_idx: list[int], right_idx: list[int]):
    """
    Mide simetría aproximada entre pares (izq, der) en el eje X.
    Score bajo = asimetría; alto = más simétrico (inverso de diferencia media).
    """
    if kp_xy.shape[0] == 0: return 0.0
    diffs = []
    for li, ri in zip(left_idx, right_idx):
        if li < kp_xy.shape[1] and ri < kp_xy.shape[1]:
            d = np.nanmean(np.abs(kp_xy[:, li, 0] - (1.0 - kp_xy[:, ri, 0])))  # espejo simple
            if np.isfinite(d): diffs.append(d)
    if len(diffs) == 0: return 0.0
    s = 1.0 / (1e-6 + np.mean(diffs))
    return float(s)

def _level_range(kp_xy: np.ndarray):
    """
    Rango vertical (y) de cadera/pecho/cabeza como proxy de niveles.
    """
    if kp_xy.shape[0] == 0: return 0.0
    J = kp_xy.shape[1]
    # índices aproximados (mediapipe 33) o coco 17
    if J >= 33:
        head = 0; hip = 23; chest = 11
    else:
        head = 0; hip = 11; chest = 5
    y_stack = []
    for j in (head, chest, hip):
        if j < J: y_stack.append(kp_xy[:, j, 1])
    if not y_stack: return 0.0
    Y = np.vstack(y_stack)
    rng = np.nanmax(Y) - np.nanmin(Y)
    return float(rng if np.isfinite(rng) else 0.0)

def features_coreograficos(keypoints: np.ndarray, meta: dict | None = None) -> dict:
    """
    Entradas:
        keypoints: (T, J, D) con valores normalizados (0..1 en mediapipe) o pixeles (YOLO).
    Salida:
        dict con métricas: claves deben alinearse (o mapearse) con FEATURE_COLS del bundle.
    """
    feats = {}
    if keypoints is None or keypoints.shape[0] == 0:
        # devuelve todos ceros para mantener pipeline vivo
        return {
            "amplitud_x": 0.0, "amplitud_y": 0.0, "amplitud_z": 0.0,
            "velocidad_media": 0.0, "simetria": 0.0,
            "nivel_rango": 0.0, "variedad_direcciones": 0.0
        }

    T, J, D = keypoints.shape
    xy = keypoints[..., :2]

    # Amplitudes globales
    feats["amplitud_x"] = _amp(xy[..., 0])
    feats["amplitud_y"] = _amp(xy[..., 1])
    feats["amplitud_z"] = float(_amp(keypoints[..., 2])) if D >= 3 else 0.0

    # Velocidad media (aprox) en pixeles o normalizado
    dx = _finite_diff(xy)
    v = np.linalg.norm(dx, axis=2)  # (T-1, J)
    feats["velocidad_media"] = _mean_abs(v)

    # Simetría (pares básicos: muñeca, codo, hombro, cadera, rodilla, tobillo)
    if J >= 33:
        L = [15, 13, 11, 23, 25, 27]
        R = [16, 14, 12, 24, 26, 28]
    else:
        # COCO
        L = [9, 7, 5, 11, 13, 15]
        R = [10, 8, 6, 12, 14, 16]
    feats["simetria"] = _symmetry_score(xy, L, R)

    # Rango de niveles
    feats["nivel_rango"] = _level_range(xy)

    # Variedad de direcciones (conteo de cambios de signo en dx/dy normalizado)
    if dx.size > 0:
        sign_changes_x = np.sum(np.diff(np.sign(dx[..., 0]), axis=0) != 0)
        sign_changes_y = np.sum(np.diff(np.sign(dx[..., 1]), axis=0) != 0)
        feats["variedad_direcciones"] = float(sign_changes_x + sign_changes_y) / max(1, T-2)
    else:
        feats["variedad_direcciones"] = 0.0

    # Si tu bundle espera más columnas específicas, añádelas aquí con nombres exactos.
    return feats
