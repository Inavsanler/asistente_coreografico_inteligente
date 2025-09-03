# src/backends.py
import numpy as np

def mediapipe_video_to_keypoints(video_path, sample_stride=2):
    import cv2, mediapipe as mp, numpy as np
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    frames=[]; idx=0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if (idx % max(1, int(sample_stride))) != 0:
                idx+=1; continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lmks = res.pose_landmarks.landmark
                xy = np.array([[l.x, l.y] for l in lmks], dtype=np.float32)
            else:
                xy = np.full((33,2), np.nan, dtype=np.float32)
            frames.append(xy); idx+=1
    finally:
        cap.release(); pose.close()
    K = np.asarray(frames, dtype=np.float32)
    return K, float(fps)

def yolo_video_to_keypoints(video_path, model_name="yolov8n-pose.pt", conf=0.25, iou=0.5, stride=1):
    # Importamos aquí (lazy) para no romper en entornos sin ultralytics
    try:
        from ultralytics import YOLO  # <- ojo al nombre correcto
    except Exception as e:
        raise RuntimeError(
            "YOLOv8-Pose no disponible. Instala 'ultralytics' (entorno GPU) "
            "o desactiva YOLO en la app."
        ) from e

    import cv2, numpy as np
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames=[]; idx=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if (idx % max(1,int(stride))) != 0:
            idx+=1; continue
        res = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
        if res and res[0].keypoints is not None and len(res[0].keypoints) > 0:
            boxes = res[0].boxes.xyxy
            areas = [(float(b[2]-b[0]) * float(b[3]-b[1])) for b in boxes]
            person_idx = int(np.argmax(areas)) if len(areas) else 0
            kps = res[0].keypoints.xyn[person_idx].cpu().numpy()
        else:
            kps = np.full((17,2), np.nan, dtype=np.float32)
        frames.append(kps); idx+=1
    cap.release()
    K = np.asarray(frames, dtype=np.float32)
    return K, float(fps)
