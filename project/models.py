import threading
from ultralytics import YOLO
from config import MODEL_PATH_V8, MODEL_PATH_V11n, MODEL_PATH_V11s

# --- Load all models ---
yolo_v8 = YOLO(MODEL_PATH_V8)
yolo_v11n = YOLO(MODEL_PATH_V11n)
yolo_v11s = YOLO(MODEL_PATH_V11s)

# --- Default active model and thread-safe lock ---
active_model = yolo_v8
active_model_lock = threading.Lock()
active_model_name = "YOLOv8"