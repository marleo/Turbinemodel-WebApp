import cv2
import os

basedir = os.path.abspath(os.path.dirname(__file__))

# --- File Paths and Extensions ---
STATIC_FOLDER = os.path.join(basedir, "static")
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, "uploads")
OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, "outputs")
ALLOWED_EXTS = {"mp4", "mov", "avi", "mkv"}
MODEL_PATH_V8 = r"C:/PATH/TO/YOLO_Training/runs/segment/yolov8n_seg_custom/weights/best.pt"
MODEL_PATH_V11n = r"C:/PATH/TO/YOLO_Training/runs/segment/yolov11n_seg_custom/weights/best.pt"
MODEL_PATH_V11s = r"C:/PATH/TO/YOLO_Training/runs/segment/yolov11s_seg_custom/weights/best.pt"

# --- Blade and Hub Configuration ---
N_BLADES = 3
HUB_CX = None  # e.g., 960
HUB_CY = None  # e.g., 540
HUB_SMOOTHING_ALPHA = 0.1  # Smoothing factor for the dynamic hub position

# --- YOLO/Tracker Parameters ---
IMG_SIZE = 640
CONF_THRES = 0.15
IOU_THRES = 0.5
TRACKER_CFG = "bytetrack.yaml"

# --- Overlay Style ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 1
BLADE_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]