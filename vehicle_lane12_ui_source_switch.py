#!/usr/bin/env python3
import argparse
import json
import math
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox

MODEL_PATH = "yolo11n.pt"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30
YOLO_IMAGE_SIZE = 640
CONFIDENCE = 0.45
VEHICLE_CLASSES = [2, 5, 7]
VEHICLE_NAMES = { 2: "CAR", 5: "BUS", 7: "TRUCK"}
VEHICLE_WIDTHS = { 2: 1.80, 5: 2.50, 7: 2.50}
FOCAL_LENGTH_PX = 700.0
LANE_SMOOTH_ALPHA = 0.12
LANE_MAX_MISSING = 10
DEFAULT_POI_FILE = "enforcement_points.json"
WARNING_RADIUS_M = 500.0


def open_camera(camera_id):
    camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{camera_id}")
    ok, frame = camera.read()
    if not ok or frame is None:
        camera.release()
        raise RuntimeError(f"Cannot read /dev/video{camera_id}")
    return camera


def open_video_file(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path}")

    ok, frame = cap.read()

    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Cannot read video file: {path}")

    # rewind after validation
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap


def estimate_distance(class_id, box_width):
    if box_width <= 0:
        return None
    vehicle_width = VEHICLE_WIDTHS.get(class_id)
    if vehicle_width is None:
        return None
    return vehicle_width * FOCAL_LENGTH_PX / box_width


def region_of_interest(image):
    height, width = image.shape[:2]
    polygon = np.array([[
        (0, height - 1),
        (int(width * 0.30), int(height * 0.50)),
        (int(width * 0.70), int(height * 0.50)),
        (width - 1, height - 1),
    ]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)


def make_lane_line(frame, slope, intercept):
    height, width = frame.shape[:2]
    if abs(slope) < 0.01:
        return None
    y1 = height - 1
    y2 = int(height * 0.56)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    x1 = int(np.clip(x1, -width, width * 2))
    x2 = int(np.clip(x2, -width, width * 2))
    return x1, y1, x2, y2


def detect_lane_lines(frame):
    height, width = frame.shape[:2]
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(
        hls,
        np.array([0, 140, 0], dtype=np.uint8),
        np.array([255, 255, 255], dtype=np.uint8),
    )
    yellow_mask = cv2.inRange(
        hls,
        np.array([10, 25, 70], dtype=np.uint8),
        np.array([50, 255, 255], dtype=np.uint8),
    )
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered = cv2.bitwise_and(frame, frame, mask=color_mask)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 25, 100)
    cropped = region_of_interest(edges)
    lines = cv2.HoughLinesP(
        cropped, rho=1, theta=np.pi / 180,
        threshold=18, minLineLength=20, maxLineGap=100
    )
    left_lines, right_lines = [], []
    if lines is not None:
        normalized_lines = np.asarray(lines, dtype=np.int32).reshape(-1, 4)
        for x1, y1, x2, y2 in normalized_lines:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.30 or abs(slope) > 4.0:
                continue
            intercept = y1 - slope * x1
            center_x = (x1 + x2) / 2
            if slope < 0 and center_x < width * 0.62:
                left_lines.append((slope, intercept))
            elif slope > 0 and center_x > width * 0.38:
                right_lines.append((slope, intercept))
    left_lane = right_lane = None
    if left_lines:
        slope, intercept = np.median(np.asarray(left_lines), axis=0)
        left_lane = make_lane_line(frame, float(slope), float(intercept))
    if right_lines:
        slope, intercept = np.median(np.asarray(right_lines), axis=0)
        right_lane = make_lane_line(frame, float(slope), float(intercept))
    return left_lane, right_lane


class LaneTracker:
    def __init__(self, alpha=LANE_SMOOTH_ALPHA, max_missing=LANE_MAX_MISSING):
        self.alpha = alpha
        self.max_missing = max_missing
        self.left = self.right = None
        self.left_missing = self.right_missing = 0

    def smooth_line(self, previous, current):
        if previous is None:
            return tuple(int(v) for v in current)
        return tuple(
            int(old * (1.0 - self.alpha) + new * self.alpha)
            for old, new in zip(previous, current)
        )

    def update_side(self, previous, current, missing_count):
        if current is not None:
            return self.smooth_line(previous, current), 0
        missing_count += 1
        if previous is not None and missing_count <= self.max_missing:
            return previous, missing_count
        return None, missing_count

    def valid_pair(self, left, right, frame_width):
        if left is None or right is None:
            return True
        bottom_width = right[0] - left[0]
        top_width = right[2] - left[2]
        if bottom_width <= 0 or top_width <= 0:
            return False
        if bottom_width < frame_width * 0.25 or bottom_width > frame_width * 1.30:
            return False
        if top_width >= bottom_width:
            return False
        return True

    def update(self, new_left, new_right, frame_width):
        candidate_left, left_missing = self.update_side(
            self.left, new_left, self.left_missing
        )
        candidate_right, right_missing = self.update_side(
            self.right, new_right, self.right_missing
        )
        if self.valid_pair(candidate_left, candidate_right, frame_width):
            self.left = candidate_left
            self.right = candidate_right
            self.left_missing = left_missing
            self.right_missing = right_missing
        else:
            self.left_missing += 1
            self.right_missing += 1
        return self.left, self.right


def draw_lane_lines(frame, left_lane, right_lane):
    overlay = frame.copy()
    if left_lane is not None:
        cv2.line(overlay, left_lane[:2], left_lane[2:], (0, 255, 255), 6)
    if right_lane is not None:
        cv2.line(overlay, right_lane[:2], right_lane[2:], (0, 255, 255), 6)
    if left_lane is not None and right_lane is not None:
        polygon = np.array([[
            left_lane[:2], left_lane[2:], right_lane[2:], right_lane[:2]
        ]], dtype=np.int32)
        cv2.fillPoly(overlay, polygon, (0, 120, 0))
    return cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)


def is_front_vehicle(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    return (
        frame_width * 0.25 <= center_x <= frame_width * 0.75
        and y2 >= frame_height * 0.35
    )


def detect_vehicles(model, frame, output, image_size, confidence):
    frame_height, frame_width = frame.shape[:2]
    results = model.predict(
        source=frame, device=0, imgsz=image_size, conf=confidence,
        classes=VEHICLE_CLASSES, half=True, verbose=False
    )
    boxes = results[0].boxes
    candidates = []
    vehicle_count = 0
    if boxes is None:
        return output, vehicle_count, None
    for item in boxes:
        x1, y1, x2, y2 = (
            item.xyxy[0].detach().cpu().numpy().astype(int).tolist()
        )
        class_id = int(item.cls[0].item())
        score = float(item.conf[0].item())
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            continue
        vehicle_count += 1
        distance = estimate_distance(class_id, box_width)
        name = VEHICLE_NAMES.get(class_id, "VEHICLE")
        label = f"{name} {score:.2f}"
        if distance is not None:
            label += f" {distance:.1f}m"
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 160, 0), 2)
        cv2.putText(
            output, label, (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2
        )
        box = (x1, y1, x2, y2)
        if is_front_vehicle(box, frame_width, frame_height):
            candidates.append({
                "box": box,
                "distance": distance,
                "area": box_width * box_height,
            })
    front_vehicle = None
    if candidates:
        front_vehicle = max(candidates, key=lambda item: item["area"])
        x1, y1, x2, y2 = front_vehicle["box"]
        distance = front_vehicle["distance"]
        label = "FRONT VEHICLE" if distance is None else f"FRONT {distance:.1f}m"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(
            output, label, (x1, max(35, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
        )
    return output, vehicle_count, front_vehicle


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    )
    return 2.0 * r * math.asin(math.sqrt(a))


def load_pois(path):
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("points", [])
    result = []
    for item in data:
        try:
            result.append({
                "name": str(item.get("name", "Enforcement")),
                "type": str(item.get("type", "camera")),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
            })
        except Exception:
            pass
    return result


def nearest_warning(lat, lon, pois):
    if lat is None or lon is None or not pois:
        return None
    best = None
    for poi in pois:
        distance = haversine_m(lat, lon, poi["lat"], poi["lon"])
        if best is None or distance < best["distance"]:
            best = dict(poi)
            best["distance"] = distance
    return best


class DualCameraUI:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.root.title("12-inch Dual Camera Driving Assist")
        self.root.configure(bg="#101010")
        self.root.geometry(args.window)
        self.root.minsize(1000, 420)

        self.running = True
        self.yolo_enabled = False
        self.detect_rear = False
        self.lane_enabled = True
        self.front_source_mode = "webcam"
        self.rear_source_mode = "webcam"

        self.front_source_path = None
        self.rear_source_path = None

        self.front_cap = open_camera(args.front_camera)
        self.rear_cap = open_camera(args.rear_camera)

        self.model = None
        self.model_loading = False
        self.front_lane_tracker = LaneTracker()
        self.current_lat = None
        self.current_lon = None
        self.pois = load_pois(args.poi)
        self.front_fps = 0.0
        self.rear_fps = 0.0
        self.last_front_time = time.perf_counter()
        self.last_rear_time = time.perf_counter()

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_loop()

    def build_ui(self):
        top = tk.Frame(self.root, bg="#171717", height=52)
        top.pack(side=tk.TOP, fill=tk.X)

        self.detect_button = tk.Button(
            top, text="偵測 OFF", command=self.toggle_detection,
            font=("Arial", 14, "bold"), bg="#6b1c1c", fg="white",
            activebackground="#a52a2a", relief=tk.FLAT, padx=18
        )
        self.detect_button.pack(side=tk.LEFT, padx=8, pady=6)

        self.lane_button = tk.Button(
            top, text="車道線 ON", command=self.toggle_lane,
            font=("Arial", 11, "bold"), bg="#167339", fg="white",
            relief=tk.FLAT, padx=12
        )
        self.lane_button.pack(side=tk.LEFT, padx=4, pady=6)

        self.front_source_button = tk.Button(
            top,
            text="前方: Webcam",
            command=self.switch_front_source,
            font=("Arial", 10, "bold"),
            bg="#303030",
            fg="white",
            relief=tk.FLAT,
            padx=10,
        )
        self.front_source_button.pack(side=tk.LEFT, padx=4, pady=6)

        self.rear_source_button = tk.Button(
            top,
            text="後方: Webcam",
            command=self.switch_rear_source,
            font=("Arial", 10, "bold"),
            bg="#303030",
            fg="white",
            relief=tk.FLAT,
            padx=10,
        )
        self.rear_source_button.pack(side=tk.LEFT, padx=4, pady=6)

        self.rear_detect_button = tk.Button(
            top, text="後鏡 YOLO OFF", command=self.toggle_rear_detection,
            font=("Arial", 11, "bold"), bg="#303030", fg="white",
            relief=tk.FLAT, padx=12
        )
        self.rear_detect_button.pack(side=tk.LEFT, padx=4, pady=6)

        self.map_button = tk.Button(
            top, text="Google Maps", command=self.open_google_maps,
            font=("Arial", 11, "bold"), bg="#244a6b", fg="white",
            relief=tk.FLAT, padx=12
        )
        self.map_button.pack(side=tk.LEFT, padx=4, pady=6)

        self.poi_button = tk.Button(
            top, text="載入警示點", command=self.choose_poi_file,
            font=("Arial", 11), bg="#303030", fg="white",
            relief=tk.FLAT, padx=10
        )
        self.poi_button.pack(side=tk.LEFT, padx=4, pady=6)

        self.status_label = tk.Label(
            top, text="YOLO OFF | GPS WAIT",
            font=("Arial", 12, "bold"),
            bg="#171717", fg="#7CFC00"
        )
        self.status_label.pack(side=tk.RIGHT, padx=14)

        main = tk.Frame(self.root, bg="#000000")
        main.pack(fill=tk.BOTH, expand=True)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        left_panel = tk.Frame(
            main, bg="#000000",
            highlightthickness=1, highlightbackground="#444444"
        )
        left_panel.grid(row=0, column=0, sticky="nsew")

        right_panel = tk.Frame(
            main, bg="#000000",
            highlightthickness=1, highlightbackground="#444444"
        )
        right_panel.grid(row=0, column=1, sticky="nsew")

        tk.Label(
            left_panel, text="前方畫面", font=("Arial", 15, "bold"),
            bg="#101010", fg="white"
        ).pack(fill=tk.X)
        self.front_view = tk.Label(left_panel, bg="black")
        self.front_view.pack(fill=tk.BOTH, expand=True)
        self.front_status = tk.Label(
            left_panel, text="前鏡頭", font=("Arial", 11),
            bg="#101010", fg="#dddddd", anchor="w"
        )
        self.front_status.pack(fill=tk.X, padx=6)

        tk.Label(
            right_panel, text="後方路況", font=("Arial", 15, "bold"),
            bg="#101010", fg="white"
        ).pack(fill=tk.X)
        self.rear_view = tk.Label(right_panel, bg="black")
        self.rear_view.pack(fill=tk.BOTH, expand=True)
        self.rear_status = tk.Label(
            right_panel, text="後鏡頭", font=("Arial", 11),
            bg="#101010", fg="#dddddd", anchor="w"
        )
        self.rear_status.pack(fill=tk.X, padx=6)

        warning_bar = tk.Frame(self.root, bg="#202020", height=54)
        warning_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.warning_label = tk.Label(
            warning_bar, text="測速 / 科技執法：GPS 尚未設定",
            font=("Arial", 14, "bold"),
            bg="#202020", fg="#f0f0f0"
        )
        self.warning_label.pack(side=tk.LEFT, padx=14, pady=8)

        gps_frame = tk.Frame(warning_bar, bg="#202020")
        gps_frame.pack(side=tk.RIGHT, padx=8)

        tk.Label(gps_frame, text="Lat", bg="#202020", fg="white").grid(row=0, column=0)
        self.lat_entry = tk.Entry(gps_frame, width=11)
        self.lat_entry.grid(row=0, column=1, padx=3)

        tk.Label(gps_frame, text="Lon", bg="#202020", fg="white").grid(row=0, column=2)
        self.lon_entry = tk.Entry(gps_frame, width=11)
        self.lon_entry.grid(row=0, column=3, padx=3)

        tk.Button(
            gps_frame, text="更新 GPS", command=self.update_manual_gps
        ).grid(row=0, column=4, padx=5)

    def load_model_async(self):
        if self.model is not None or self.model_loading:
            return
        self.model_loading = True

        def worker():
            try:
                self.model = YOLO(self.args.model)
            except Exception as exc:
                self.root.after(
                    0,
                    lambda: messagebox.showerror("YOLO", str(exc))
                )
            finally:
                self.model_loading = False

        threading.Thread(target=worker, daemon=True).start()

    def toggle_detection(self):
        self.yolo_enabled = not self.yolo_enabled
        if self.yolo_enabled:
            self.load_model_async()
            self.detect_button.configure(text="偵測 ON", bg="#167339")
        else:
            self.detect_button.configure(text="偵測 OFF", bg="#6b1c1c")

    def toggle_lane(self):
        self.lane_enabled = not self.lane_enabled

        if self.lane_enabled:
            self.lane_button.configure(
                text="車道線 ON",
                bg="#167339",
            )
        else:
            self.lane_button.configure(
                text="車道線 OFF",
                bg="#6b1c1c",
            )

            self.front_lane_tracker.left = None
            self.front_lane_tracker.right = None
            self.front_lane_tracker.left_missing = 0
            self.front_lane_tracker.right_missing = 0

    def replace_capture(self, side, mode, path=None):
        if side == "front":
            old_cap = self.front_cap
            camera_id = self.args.front_camera
        else:
            old_cap = self.rear_cap
            camera_id = self.args.rear_camera

        try:
            if mode == "webcam":
                new_cap = open_camera(camera_id)
            else:
                new_cap = open_video_file(path)
        except Exception as exc:
            messagebox.showerror(
                "Source",
                str(exc),
            )
            return False

        if old_cap is not None:
            old_cap.release()

        if side == "front":
            self.front_cap = new_cap
            self.front_source_mode = mode
            self.front_source_path = path
        else:
            self.rear_cap = new_cap
            self.rear_source_mode = mode
            self.rear_source_path = path

        return True

    def switch_front_source(self):
        if self.front_source_mode == "webcam":
            path = filedialog.askopenfilename(
                title="選擇前方 MOV/MP4",
                filetypes=[
                    ("Video", "*.mov *.mp4 *.avi *.mkv"),
                    ("MOV", "*.mov"),
                    ("MP4", "*.mp4"),
                    ("All Files", "*.*"),
                ],
            )

            if not path:
                return

            if self.replace_capture("front", "video", path):
                self.front_source_button.configure(
                    text="前方: MOV",
                    bg="#244a6b",
                )
        else:
            if self.replace_capture("front", "webcam"):
                self.front_source_button.configure(
                    text="前方: Webcam",
                    bg="#303030",
                )

    def switch_rear_source(self):
        if self.rear_source_mode == "webcam":
            path = filedialog.askopenfilename(
                title="選擇後方 MOV/MP4",
                filetypes=[
                    ("Video", "*.mov *.mp4 *.avi *.mkv"),
                    ("MOV", "*.mov"),
                    ("MP4", "*.mp4"),
                    ("All Files", "*.*"),
                ],
            )

            if not path:
                return

            if self.replace_capture("rear", "video", path):
                self.rear_source_button.configure(
                    text="後方: MOV",
                    bg="#244a6b",
                )
        else:
            if self.replace_capture("rear", "webcam"):
                self.rear_source_button.configure(
                    text="後方: Webcam",
                    bg="#303030",
                )

    def read_source_frame(self, cap, mode):
        ok, frame = cap.read()

        if ok and frame is not None:
            return True, frame

        # MOV/MP4 到結尾時，自動循環播放
        if mode == "video":
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()

            if ok and frame is not None:
                return True, frame

        return False, None

    def toggle_rear_detection(self):
        self.detect_rear = not self.detect_rear
        self.rear_detect_button.configure(
            text="後鏡 YOLO ON" if self.detect_rear else "後鏡 YOLO OFF",
            bg="#167339" if self.detect_rear else "#303030"
        )

    def update_manual_gps(self):
        try:
            self.current_lat = float(self.lat_entry.get())
            self.current_lon = float(self.lon_entry.get())
        except ValueError:
            messagebox.showwarning("GPS", "請輸入有效的經緯度")

    def set_gps_position(self, lat, lon):
        self.current_lat = float(lat)
        self.current_lon = float(lon)

    def choose_poi_file(self):
        path = filedialog.askopenfilename(
            title="選擇測速/科技執法 JSON",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")]
        )
        if path:
            self.pois = load_pois(path)

    def open_google_maps(self):
        if self.current_lat is not None and self.current_lon is not None:
            url = (
                "https://www.google.com/maps/search/"
                f"?api=1&query={self.current_lat},{self.current_lon}"
            )
        else:
            url = "https://www.google.com/maps"
        webbrowser.open(url)

    def update_warning(self):
        warning = nearest_warning(
            self.current_lat, self.current_lon, self.pois
        )
        if warning is None:
            if self.current_lat is None or self.current_lon is None:
                text = "測速 / 科技執法：GPS 尚未設定"
            elif not self.pois:
                text = "測速 / 科技執法：尚未載入警示點資料"
            else:
                text = "測速 / 科技執法：附近無警示"
            self.warning_label.configure(text=text, fg="#f0f0f0")
            return

        distance = warning["distance"]
        if distance <= WARNING_RADIUS_M:
            self.warning_label.configure(
                text=f"⚠ {warning['type'].upper()}：{warning['name']} {distance:.0f} m",
                fg="#ff5252"
            )
        else:
            self.warning_label.configure(
                text=f"最近警示：{warning['name']} {distance/1000.0:.1f} km",
                fg="#f0f0f0"
            )

    def process_front(self, frame):
        frame_width = frame.shape[1]

        left_lane = None
        right_lane = None

        if self.lane_enabled:
            detected_left, detected_right = detect_lane_lines(frame)

            left_lane, right_lane = self.front_lane_tracker.update(
                detected_left,
                detected_right,
                frame_width,
            )

            output = draw_lane_lines(
                frame,
                left_lane,
                right_lane,
            )
        else:
            output = frame.copy()

        vehicle_count = 0
        front_vehicle = None
        if self.yolo_enabled and self.model is not None:
            output, vehicle_count, front_vehicle = detect_vehicles(
                self.model, frame, output,
                self.args.imgsz, self.args.confidence
            )
        return output, vehicle_count, front_vehicle, left_lane, right_lane

    def process_rear(self, frame):
        output = frame.copy()
        vehicle_count = 0
        if self.yolo_enabled and self.detect_rear and self.model is not None:
            output, vehicle_count, _ = detect_vehicles(
                self.model, frame, output,
                self.args.imgsz, self.args.confidence
            )
        return output, vehicle_count

    def frame_to_tk(self, frame, widget):
        target_w = max(320, widget.winfo_width())
        target_h = max(200, widget.winfo_height())
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(
            frame, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(image=Image.fromarray(rgb))

    def calc_fps(self, previous, old_fps):
        now = time.perf_counter()
        elapsed = now - previous
        current = 1.0 / elapsed if elapsed > 0 else 0.0
        smooth = current if old_fps == 0.0 else old_fps * 0.90 + current * 0.10
        return now, smooth

    def update_loop(self):
        if not self.running:
            return

        ok_front, front_frame = self.read_source_frame(
            self.front_cap,
            self.front_source_mode,
        )

        ok_rear, rear_frame = self.read_source_frame(
            self.rear_cap,
            self.rear_source_mode,
        )

        front_vehicle_count = 0
        rear_vehicle_count = 0
        front_vehicle = None
        left_lane = right_lane = None

        if ok_front and front_frame is not None:
            (
                front_output,
                front_vehicle_count,
                front_vehicle,
                left_lane,
                right_lane,
            ) = self.process_front(front_frame)

            self.last_front_time, self.front_fps = self.calc_fps(
                self.last_front_time, self.front_fps
            )

            front_image = self.frame_to_tk(front_output, self.front_view)
            self.front_view.configure(image=front_image)
            self.front_view.image = front_image

        if ok_rear and rear_frame is not None:
            rear_output, rear_vehicle_count = self.process_rear(rear_frame)

            self.last_rear_time, self.rear_fps = self.calc_fps(
                self.last_rear_time, self.rear_fps
            )

            rear_image = self.frame_to_tk(rear_output, self.rear_view)
            self.rear_view.configure(image=rear_image)
            self.rear_view.image = rear_image

        lane_l = "YES" if left_lane else "NO"
        lane_r = "YES" if right_lane else "NO"

        front_source_text = (
            "WEBCAM"
            if self.front_source_mode == "webcam"
            else "MOV"
        )

        rear_source_text = (
            "WEBCAM"
            if self.rear_source_mode == "webcam"
            else "MOV"
        )

        front_info = (
            f"{front_source_text} | FPS {self.front_fps:.1f} | "
            f"車輛 {front_vehicle_count} | "
            f"LANE L:{lane_l} R:{lane_r}"
        )
        if front_vehicle is not None and front_vehicle.get("distance") is not None:
            front_info += f" | FRONT {front_vehicle['distance']:.1f}m"

        self.front_status.configure(text=front_info)
        self.rear_status.configure(
            text=(
                f"{rear_source_text} | "
                f"FPS {self.rear_fps:.1f} | "
                f"車輛 {rear_vehicle_count}"
            )
        )

        detect_text = "YOLO ON" if self.yolo_enabled else "YOLO OFF"
        lane_text = "LANE ON" if self.lane_enabled else "LANE OFF"

        if self.model_loading:
            detect_text = "YOLO LOADING"
        gps_text = (
            "GPS OK"
            if self.current_lat is not None and self.current_lon is not None
            else "GPS WAIT"
        )
        self.status_label.configure(text=f"{detect_text} | {lane_text} | {gps_text}")
        self.update_warning()

        self.root.after(15, self.update_loop)

    def close(self):
        self.running = False

        if self.front_cap is not None:
            self.front_cap.release()

        if self.rear_cap is not None:
            self.rear_cap.release()

        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front-camera", type=int, default=0)
    parser.add_argument("--rear-camera", type=int, default=2)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMAGE_SIZE)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--poi", default=DEFAULT_POI_FILE)
    parser.add_argument("--window", default="1280x480")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("OpenCV:", cv2.__version__)

    root = tk.Tk()
    DualCameraUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
