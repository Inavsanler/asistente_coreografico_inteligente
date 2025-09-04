# src/inference.py
import numpy as np
import cv2

def _infer_mediapipe(video_path: str, stride: int = 2, conf: float = 0.5, return_frame_size: bool = False):
    """
    Devuelve:
      - keypoints: np.ndarray (T, J, 3)
      - meta: dict con fps, backend, frame_w, frame_h
    """
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError("MediaPipe no está instalado. Añádelo a requirements.txt.") from e

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el vídeo.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    keypoints = []
    with mp_pose.Pose(static_image_mode=False,
                      min_detection_confidence=conf,
                      min_tracking_confidence=conf) as pose:
        t = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if t % stride != 0:
                t += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                J = [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark]  # 33 joints
                keypoints.append(J)
            t += 1
    cap.release()

    kp = np.array(keypoints, dtype=float) if keypoints else np.zeros((0, 33, 3), dtype=float)
    meta = {"fps": fps, "backend": "mediapipe", "frame_w": fw, "frame_h": fh}
    return kp, meta

def _infer_yolo(video_path: str, stride: int = 2, conf: float = 0.5, return_frame_size: bool = False):
    """
    Devuelve:
      - keypoints: np.ndarray (T, 17, 3) con z=0
      - meta: dict con fps, backend, frame_w, frame_h
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics YOLO no está instalado. Añade 'ultralytics' a requirements.txt.") from e

    model = YOLO("yolov8s-pose.pt")  # asegúrate de que el peso es accesible
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el vídeo.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    keypoints = []
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if t % stride != 0:
            t += 1
            continue
        res = model.predict(frame, conf=conf, verbose=False)
        if len(res) > 0 and len(res[0].keypoints) > 0:
            k = res[0].keypoints.xy  # (N, 17, 2) en píxeles (COCO)
            if k is not None and len(k) > 0:
                person0 = k[0].cpu().numpy()  # (17, 2)
                z = np.zeros((person0.shape[0], 1), dtype=float)
                J = np.concatenate([person0, z], axis=1)  # (17, 3)
                keypoints.append(J)
        t += 1
    cap.release()

    kp = np.array(keypoints, dtype=float) if keypoints else np.zeros((0, 17, 3), dtype=float)
    meta = {"fps": fps, "backend": "yolo", "frame_w": fw, "frame_h": fh}
    return kp, meta

def run_inference_over_video(video_path: str,
                             backbone: str = "mediapipe",
                             conf: float = 0.5,
                             stride: int = 2,
                             return_frame_size: bool = False,
                             **kwargs):
    """
    Wrapper compatible con app.py. Ignora kwargs extra.
    """
    if backbone == "yolo":
        return _infer_yolo(video_path, stride=stride, conf=conf, return_frame_size=return_frame_size)
    else:
        return _infer_mediapipe(video_path, stride=stride, conf=conf, return_frame_size=return_frame_size)
