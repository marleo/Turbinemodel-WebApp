import time
import math
import numpy as np
import cv2

# Import from our new, organized modules
from config import (
    N_BLADES, IMG_SIZE, CONF_THRES, IOU_THRES, TRACKER_CFG, HUB_SMOOTHING_ALPHA,
    HUB_CX, HUB_CY, BLADE_COLORS, FONT, FONT_SCALE, THICKNESS
)
from .models import yolo_v11n, active_model, active_model_lock # Note the '.' for relative import

# -------------------- Angle-Stable ID logic --------------------
def ang_diff(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d

class AngleIDStabilizer:
    def __init__(self, n_blades=N_BLADES, max_missing=10, alpha=0.7):
        self.n_blades = n_blades
        self.stable_id_map = {}
        self.stable_id_counter = 1
        self.last_known_angles = {}
        self.angular_velocities = {}
        self.missing_frames = {}
        self.max_missing = max_missing
        self.alpha = alpha

    def get_stable_ids(self, detections_with_ids, hub_cx, hub_cy, fps=None):
        current_tracker_ids = set()
        current_angles = {}
        for d in detections_with_ids:
            tracker_id, (det_cx, det_cy) = d['id'], d['centroid']
            current_tracker_ids.add(tracker_id)
            current_angles[tracker_id] = math.atan2(det_cy - hub_cy, det_cx - hub_cx)

        new_assignments = {}
        predicted_angles = {}
        for stable_id, last_angle in self.last_known_angles.items():
            if self.missing_frames[stable_id] < self.max_missing:
                pred_angle = last_angle + self.angular_velocities.get(stable_id, 0)
                pred_angle = (pred_angle + math.pi) % (2 * math.pi) - math.pi
                predicted_angles[stable_id] = pred_angle
                self.missing_frames[stable_id] += 1
            else:
                predicted_angles[stable_id] = last_angle

        unassigned_tracker_ids = list(current_tracker_ids)
        assigned_stable_ids = set()

        while unassigned_tracker_ids and len(assigned_stable_ids) < self.n_blades:
            best_match = None
            min_diff = float('inf')

            for stable_id, pred_angle in predicted_angles.items():
                if stable_id in assigned_stable_ids:
                    continue

                for tracker_id in unassigned_tracker_ids:
                    curr_angle = current_angles[tracker_id]
                    diff = abs(ang_diff(curr_angle, pred_angle))
                    if diff < min_diff:
                        min_diff = diff
                        best_match = (tracker_id, stable_id)

            if best_match:
                tracker_id, stable_id = best_match
                new_assignments[tracker_id] = stable_id
                self.angular_velocities[stable_id] = self.alpha * ang_diff(current_angles[tracker_id], self.last_known_angles.get(stable_id, current_angles[tracker_id])) + (1 - self.alpha) * self.angular_velocities.get(stable_id, 0)
                self.last_known_angles[stable_id] = current_angles[tracker_id]
                self.missing_frames[stable_id] = 0
                unassigned_tracker_ids.remove(tracker_id)
                assigned_stable_ids.add(stable_id)
                if stable_id in predicted_angles:
                    del predicted_angles[stable_id]
            else:
                break

        for tracker_id in unassigned_tracker_ids:
            if self.stable_id_counter <= self.n_blades:
                new_assignments[tracker_id] = self.stable_id_counter
                self.last_known_angles[self.stable_id_counter] = current_angles[tracker_id]
                self.angular_velocities[self.stable_id_counter] = 0
                self.missing_frames[self.stable_id_counter] = 0
                self.stable_id_counter += 1
            else:
                new_assignments[tracker_id] = -1

        return new_assignments


# -------------------- Helpers --------------------
def mask_centroid(mask_poly_xy):
    if mask_poly_xy is None or len(mask_poly_xy) == 0:
        return None
    x = mask_poly_xy[:, 0]
    y = mask_poly_xy[:, 1]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    if abs(a) < 1e-6:
        return float(np.mean(x)), float(np.mean(y))
    cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - y * np.roll(x, -1))) / (3 * a)
    cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - y * np.roll(x, -1))) / (3 * a)
    return float(cx), float(cy)


# -------------------- Video Processing --------------------
def process_video(in_path, out_path):
    print(f"\n[INFO] Starting video processing for '{in_path}'...")
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # initialize hub at image center, with smoothing
    smoothed_hub_pos = np.array([W/2, H/2])

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    stabilizer = AngleIDStabilizer(n_blades=N_BLADES)
    start_time = time.time()

    for frame_count in range(1, total_frames + 1):
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_v11n.track(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            tracker=TRACKER_CFG,
            persist=True,
            verbose=False
        )
        detections_with_ids = []
        r0 = results[0]
        if r0.boxes and r0.boxes.id is not None:
            track_ids = r0.boxes.id.int().cpu().tolist()
            if r0.masks:
                for i, poly in enumerate(r0.masks.xy):
                    c = mask_centroid(np.array(poly, dtype=np.float32))
                    if c:
                        detections_with_ids.append({
                            'id': track_ids[i],
                            'centroid': c,
                            'poly': np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                        })
            else:
                for i, (x1, y1, x2, y2) in enumerate(r0.boxes.xyxy.cpu().numpy()):
                    detections_with_ids.append({
                        'id': track_ids[i],
                        'centroid': ((x1+x2)/2, (y1+y2)/2),
                        'poly': None
                    })

        # --- NEW: update hub position based on centroids ---
        centroids = [d['centroid'] for d in detections_with_ids]
        if len(centroids) >= 2:
            current_hub_pos = np.mean(centroids, axis=0)
            smoothed_hub_pos = HUB_SMOOTHING_ALPHA * current_hub_pos + \
                               (1 - HUB_SMOOTHING_ALPHA) * smoothed_hub_pos
        hub_cx, hub_cy = smoothed_hub_pos

        stable_id_map = stabilizer.get_stable_ids(detections_with_ids, hub_cx, hub_cy, fps=fps)
        annotated = frame.copy()
        for d in detections_with_ids:
            stable_id = stable_id_map.get(d['id'])
            if stable_id and stable_id > 0:
                px, py = d['centroid']
                color = BLADE_COLORS[(stable_id - 1) % len(BLADE_COLORS)]
                if d['poly'] is not None:
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [d['poly']], color)
                    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
                cv2.putText(annotated, f"Blade {stable_id}", (int(px)+6,int(py)-6),
                            FONT, FONT_SCALE, (0,0,0), THICKNESS+2, cv2.LINE_AA)
                cv2.putText(annotated, f"Blade {stable_id}", (int(px)+6,int(py)-6),
                            FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)
                cv2.circle(annotated, (int(px), int(py)), 4, color, -1)

        cv2.circle(annotated, (int(hub_cx), int(hub_cy)), 5, (255,255,255), -1)
        cv2.circle(annotated, (int(hub_cx), int(hub_cy)), 9, (0,0,0), 2)

        out.write(annotated)
        print(f"\r[INFO] Processing frame {frame_count}/{total_frames}", end="")

    cap.release()
    out.release()
    print(f"\n[INFO] Video processing completed in {time.time() - start_time:.2f}s.")

# -------------------- Webcam Streaming --------------------
def gen_webcam_frames():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    stabilizer = AngleIDStabilizer(n_blades=N_BLADES)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    smoothed_hub_pos = np.array([W/2, H/2])

    while True:
        success, frame = cap.read()
        if not success: break
        with active_model_lock:
            results = active_model.track(frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, tracker=TRACKER_CFG, persist=True, verbose=False)

        detections_with_ids = []
        r0 = results[0]
        if r0.boxes and r0.boxes.id is not None:
            track_ids = r0.boxes.id.int().cpu().tolist()
            if r0.masks:
                for i, poly in enumerate(r0.masks.xy):
                    c = mask_centroid(np.array(poly, dtype=np.float32))
                    if c: detections_with_ids.append({'id': track_ids[i], 'centroid': c, 'poly': np.array(poly, dtype=np.int32).reshape((-1,1,2))})
            else:
                for i, (x1, y1, x2, y2) in enumerate(r0.boxes.xyxy.cpu().numpy()):
                    detections_with_ids.append({'id': track_ids[i], 'centroid': ((x1+x2)/2, (y1+y2)/2), 'poly': None})

        centroids = [d['centroid'] for d in detections_with_ids]
        if len(centroids) >= 2:
            current_hub_pos = np.mean(centroids, axis=0)
            smoothed_hub_pos = HUB_SMOOTHING_ALPHA * current_hub_pos + (1-HUB_SMOOTHING_ALPHA)*smoothed_hub_pos
        hub_cx, hub_cy = smoothed_hub_pos

        stable_id_map = stabilizer.get_stable_ids(detections_with_ids, hub_cx, hub_cy)
        annotated_frame = frame.copy()
        for d in detections_with_ids:
            stable_id = stable_id_map.get(d['id'])
            if stable_id and stable_id > 0:
                px, py = d['centroid']
                color = BLADE_COLORS[(stable_id-1) % len(BLADE_COLORS)]
                if d['poly'] is not None:
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, [d['poly']], color)
                    cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)
                cv2.putText(annotated_frame, f"Blade {stable_id}", (int(px)+6,int(py)-6), FONT, FONT_SCALE, (0,0,0), THICKNESS+2, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"Blade {stable_id}", (int(px)+6,int(py)-6), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)
                cv2.circle(annotated_frame, (int(px),int(py)), 4, color, -1)
        cv2.circle(annotated_frame, (int(hub_cx), int(hub_cy)), 5, (255,255,255), -1)
        cv2.circle(annotated_frame, (int(hub_cx), int(hub_cy)), 9, (0,0,0), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')