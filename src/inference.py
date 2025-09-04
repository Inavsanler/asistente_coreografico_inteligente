# src/inference.py
"""
Extracción de keypoints a partir de vídeo.
- MediaPipe (CPU): ruta por defecto.
- YOLOv8-Pose (GPU): opcional si tienes ultralytics instalado y runtime con CUDA.
Devuelve:
    keypoints: np.ndarray de forma (T, J, D) con D=2 o 3
    meta: dict con info auxiliar (fps, etc.)
"""
import os, numpy as np, cv2

def _infer_mediapipe(video_path: str, stride: int = 2, conf: float = 0.5):
    try:
        import mediapipe as mp
    except Exception:
        raise RuntimeError("MediaPipe no está instalado. Añade mediapipe a requirements.txt o usa YOLO.")

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el vídeo.")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    keypoints = []
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=conf, min_tracking_confidence=conf) as pose:
        t = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if t % stride != 0:
                t += 1; continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                J = []
                for lm in res.pose_landmarks.landmark:
                    J.append([lm.x, lm.y, lm.z])  # coordenadas normalizadas
                keypoints.append(J)
            t += 1
    cap.release()
    if len(keypoints) == 0:
        # fallback: array vacío consistente
        return np.zeros((0, 33, 3), dtype=float), {"fps": fps, "backend": "mediapipe"}
    kp = np.array(keypoints, dtype=float)  # (T, 33, 3)
    return kp, {"fps": fps, "backend": "mediapipe"}

def _infer_yolo(video_path: str, stride: int = 2, conf: float = 0.5):
    try:
        from ultralytics import YOLO
    except Exception:
        raise RuntimeError("Ultralytics YOLO no está instalado. Añade ultralytics a requirements.txt o usa MediaPipe.")

    model = YOLO("yolov8s-pose.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el vídeo.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    t = 0
    keypoints = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        if t % stride != 0:
            t += 1; continue
        # Inferencia
        res = model.predict(frame, conf=conf, verbose=False)
        # Tomar solo la primera persona con mayor confianza
        if len(res) > 0 and len(res[0].keypoints) > 0:
            k = res[0].keypoints.xy  # (N, 17, 2) si coco
            if k is not None and len(k) > 0:
                person0 = k[0].cpu().numpy()  # (17, 2)
                # Expandir a (17, 3) con z=0
                z = np.zeros((person0.shape[0], 1), dtype=float)
                J = np.concatenate([person0, z], axis=1)
                keypoints.append(J)
        t += 1
    cap.release()
    if len(keypoints) == 0:
        return np.zeros((0, 17, 3), dtype=float), {"fps": fps, "backend": "yolo"}
    kp = np.array(keypoints, dtype=float)  # (T, 17, 3)
    return kp, {"fps": fps, "backend": "yolo"}

def run_inference_over_video(video_path: str, backbone: str = "mediapipe", conf: float = 0.5, stride: int = 2):
    """
    backbone: 'mediapipe' | 'yolo'
    """
    if backbone == "yolo":
        return _infer_yolo(video_path, stride=stride, conf=conf)
    else:
        return _infer_mediapipe(video_path, stride=stride, conf=conf)
