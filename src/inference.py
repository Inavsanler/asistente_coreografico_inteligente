# src/inference.py
from __future__ import annotations

import os
import math
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2

# Dependencias opcionales
try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None  # se manejará en _infer_mediapipe

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # se manejará en _infer_yolo

__all__ = ["run_inference_over_video"]

# ---------------------------
# Utilidades internas
# ---------------------------
def _safe_video_capture(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap or not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {path}")
    return cap

def _read_all_frames(cap: cv2.VideoCapture, max_frames: Optional[int] = None) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    return frames

# ---------------------------
# Inferencias
# ---------------------------
def _infer_mediapipe(frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    Devuelve un diccionario con keypoints (si mediapipe está disponible).
    Estructura de ejemplo:
    {
      "backend": "mediapipe",
      "keypoints": [  # por frame
          {"pose": [(x,y,score), ...]},
          ...
      ]
    }
    """
    if mp is None:
        return {"backend": "mediapipe", "available": False, "keypoints": []}

    mp_pose = mp.solutions.pose
    results = []
    with mp_pose.Pose(static_image_mode=True) as pose:
        for f in frames:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            out = pose.process(f_rgb)
            kps = []
            if out.pose_landmarks:
                for lm in out.pose_landmarks.landmark:
                    kps.append((lm.x, lm.y, lm.visibility))
            results.append({"pose": kps})
    return {"backend": "mediapipe", "available": True, "keypoints": results}


def _infer_yolo(frames: List[np.ndarray], model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Corre YOLOv8/YOLO-Pose si está disponible (ultralytics). Devuelve bounding boxes / keypoints si el modelo es de pose.
    """
    if YOLO is None:
        return {"backend": "yolo", "available": False, "detections": []}

    # Si no se pasa ruta, intentamos por defecto
    model_path = model_path or os.environ.get("YOLO_MODEL_PATH", "artifacts/yolo.pt")
    if not os.path.exists(model_path):
        return {"backend": "yolo", "available": False, "detections": [], "reason": f"modelo no encontrado: {model_path}"}

    model = YOLO(model_path)
    detections = []
    for f in frames:
        res = model.predict(f, verbose=False)
        # Estandarizamos una salida compacta
        frame_out = []
        for r in res:
            boxes = getattr(r, "boxes", None)
            keypoints = getattr(r, "keypoints", None)
            entry: Dict[str, Any] = {}
            if boxes is not None and boxes.xyxy is not None:
                entry["boxes"] = boxes.xyxy.cpu().numpy().tolist()
                if hasattr(boxes, "conf") and boxes.conf is not None:
                    entry["scores"] = boxes.conf.cpu().numpy().tolist()
                if hasattr(boxes, "cls") and boxes.cls is not None:
                    entry["classes"] = boxes.cls.cpu().numpy().tolist()
            if keypoints is not None and keypoints.xy is not None:
                entry["keypoints"] = keypoints.xy.cpu().numpy().tolist()
            frame_out.append(entry)
        detections.append(frame_out)

    return {"backend": "yolo", "available": True, "detections": detections}


# ---------------------------
# Orquestador
# ---------------------------
def run_inference_over_video(
    video_path: str,
    backend: str = "mediapipe",   # "mediapipe" | "yolo"
    max_frames: Optional[int] = None,
    yolo_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Carga el vídeo, extrae frames y despacha al backend seleccionado.
    Retorna un diccionario con resultados normalizados.
    """
    cap = _safe_video_capture(video_path)
    try:
        frames = _read_all_frames(cap, max_frames=max_frames)
    finally:
        cap.release()

    if backend.lower() in ("mediapipe", "mp"):
        res = _infer_mediapipe(frames)
    elif backend.lower() in ("yolo", "yolov8", "yolo-pose"):
        res = _infer_yolo(frames, model_path=yolo_model_path)
    else:
        raise ValueError(f"Backend desconocido: {backend}")

    # Normalización mínima común
    return {
        "video_path": video_path,
        "n_frames": len(frames),
        "backend": res.get("backend"),
        "available": res.get("available", True),
        "data": res,
    }
