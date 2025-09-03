# -*- coding: utf-8 -*-
"""
model.py — carga de artefactos y predicción para el Asistente Coreográfico.
Solución definitiva contra: float() argument must be a string or a real number, not 'dict'

Estrategia:
- safe_float: convierte cualquier valor (incl. dicts anidados) a float o usa default.
- sanitize_thresholds: normaliza mapas de umbrales a {label: float}.
- Se aplica al cargar artefactos, a los thresholds recibidos y al aplicar t por etiqueta.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import json
import joblib
import numpy as np


# -------------------------
# Helpers anti-errores
# -------------------------
def safe_float(v: Any, default: float) -> float:
    """Convierte v a float de forma segura. Acepta números, strings numéricos y dicts tipo {'thr':0.6}."""
    try:
        # números ya numéricos
        if isinstance(v, (int, float, np.floating, np.integer)):
            return float(v)
        # strings numéricos
        if isinstance(v, str):
            return float(v.strip())
        # dicts con claves típicas
        if isinstance(v, dict):
            for k in ("thr", "value", "threshold", "umbral", "val"):
                if k in v:
                    return safe_float(v[k], default)
        # todo lo demás
        return float(default)
    except Exception:
        return float(default)


def sanitize_thresholds(maybe_map: Any, default: float) -> Dict[str, float]:
    """Normaliza cualquier mapa de umbrales a {str(label): float} usando safe_float."""
    if not isinstance(maybe_map, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in maybe_map.items():
        out[str(k)] = safe_float(v, default)
    return out


# -------------------------
# Utilidades internas
# -------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_2d_array(X: Any) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"Esperado X con 2 dimensiones [N, F], recibido shape={X.shape}")
    return X


# -------------------------
# Carga de artefactos
# -------------------------
def _try_load_bundle(artifacts_dir: str) -> Optional[Tuple[Any, Any, Any, List[str], Dict[str, float]]]:
    art = Path(artifacts_dir)
    bundle = art / "complete_model_thresholded_bundle.joblib"
    if not bundle.exists():
        return None

    b = joblib.load(bundle)

    pipe = b.get("pipeline")
    if pipe is None:
        return None

    named = getattr(pipe, "named_steps", {})
    imputer = named.get("imputer") or named.get("imp") or b.get("imputer")
    scaler = named.get("scaler") or named.get("std") or b.get("scaler")
    clf = named.get("clf") or named.get("classifier") or named.get("logreg") or b.get("clf")
    if clf is None:
        return None

    labels = b.get("label_names") or b.get("labels") or []
    if not labels:
        labels_csv = art / "complete_label_names.csv"
        if labels_csv.exists():
            import pandas as pd
            labels = pd.read_csv(labels_csv, header=None)[0].astype(str).tolist()
        else:
            return None
    labels = list(map(str, labels))

    thr_any = b.get("thresholds")
    if isinstance(thr_any, dict):
        thr_map = sanitize_thresholds(thr_any, default=0.5)
    elif thr_any is not None:
        arr = np.asarray(thr_any, dtype=float).ravel()
        thr_map = {labels[i]: float(arr[i]) for i in range(min(len(labels), len(arr)))}
    else:
        thr_map = {}

    return imputer, scaler, clf, labels, thr_map


def load_artifacts(artifacts_dir: str) -> Tuple[Any, Any, Any, List[str], Dict[str, float]]:
    # 1) bundle
    out = _try_load_bundle(artifacts_dir)
    if out is not None:
        # out ya trae thr_map saneado en bundle
        return out

    # 2) legacy
    art = Path(artifacts_dir)
    imp_path = art / "imputer.joblib"
    scaler_path = art / "scaler.joblib"
    clf_path = art / "model_ovr_logreg.joblib"
    if not (imp_path.exists() and scaler_path.exists() and clf_path.exists()):
        raise FileNotFoundError("Faltan artefactos de modelo (bundle o legacy).")

    imputer = joblib.load(imp_path)
    scaler = joblib.load(scaler_path)
    clf = joblib.load(clf_path)

    labels_json = art / "labels.json"
    labels_csv = art / "complete_label_names.csv"
    if labels_json.exists():
        with open(labels_json, "r", encoding="utf-8") as f:
            labels = json.load(f)
    elif labels_csv.exists():
        import pandas as pd
        labels = pd.read_csv(labels_csv, header=None)[0].astype(str).tolist()
    else:
        raise FileNotFoundError("No se encontraron nombres de etiquetas.")
    labels = list(map(str, labels))

    thr_map: Dict[str, float] = {}
    for cand in ("thresholds.json", "complete_thresholds.json"):
        p = art / cand
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                thr_any = json.load(f)
            if isinstance(thr_any, dict):
                thr_map = sanitize_thresholds(thr_any, default=0.5)
            elif isinstance(thr_any, (list, tuple)):
                arr = np.asarray(thr_any, dtype=float).ravel()
                thr_map = {labels[i]: float(arr[i]) for i in range(min(len(labels), len(arr)))}
            break

    return imputer, scaler, clf, labels, thr_map


# -------------------------
# Predicción
# -------------------------
def _predict_proba(clf: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        P = clf.predict_proba(X)
        P = np.asarray(P)
        if isinstance(P, list):  # OneVsRest
            pos = [np.asarray(pi)[:, 1] if pi.shape[1] == 2 else np.asarray(pi).ravel() for pi in P]
            P = np.stack(pos, axis=1)
        return P

    if hasattr(clf, "decision_function"):
        dec = np.asarray(clf.decision_function(X))
        if dec.ndim == 1:
            dec = dec.reshape(-1, 1)
        return _sigmoid(dec)

    raise RuntimeError("El clasificador no soporta predict_proba ni decision_function.")


def predict_labels(
    X: Any,
    artifacts_dir: str = "artifacts",
    thr_default: float = 0.5,
    thr_per_label: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    - thr_default: float global
    - thr_per_label: dict {label: threshold(any)} (se sanea)
    - retro-compat: threshold= (alias)
    """
    # retro-compat
    if "threshold" in kwargs and kwargs["threshold"] is not None and thr_default == 0.5:
        # si pasan un dict aquí, safe_float lo ignorará y quedará 0.5
        thr_default = safe_float(kwargs["threshold"], default=thr_default)

    X = _ensure_2d_array(X)

    imputer, scaler, clf, labels, saved_thr_map = load_artifacts(artifacts_dir)

    X_imp = imputer.transform(X)
    X_std = scaler.transform(X_imp)

    P = _predict_proba(clf, X_std)  # [N, C]
    N, C = P.shape
    if C != len(labels):
        raise ValueError(f"Inconsistencia: P.shape[1]={C} != len(labels)={len(labels)}")

    # thresholds: saneo total (prioridad a los que vienen por parámetro)
    if isinstance(thr_per_label, dict) and thr_per_label:
        thr_map = sanitize_thresholds(thr_per_label, default=thr_default)
    elif isinstance(saved_thr_map, dict) and saved_thr_map:
        thr_map = sanitize_thresholds(saved_thr_map, default=thr_default)
    else:
        thr_map = {}

    labels = list(map(str, labels))
    y_hat = np.zeros((N, C), dtype=int)
    for j, lab in enumerate(labels):
        t = safe_float(thr_map.get(lab, thr_default), default=thr_default)
        y_hat[:, j] = (P[:, j] >= t).astype(int)

    out_labels: List[List[str]] = []
    out_scores: List[List[float]] = []
    for i in range(N):
        idx = np.where(y_hat[i] == 1)[0]
        out_labels.append([labels[k] for k in idx])
        out_scores.append([float(P[i, k]) for k in idx])

    return out_labels, out_scores
