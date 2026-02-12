import os
import time
import math
import csv
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# 1) CONFIG
# =========================
YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"

MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.45
TRACKER = "bytetrack.yaml"

# COCO vehicle classes: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}

HISTORY_LEN = 15
METERS_PER_PIXEL = 0.03

RECONNECT_SLEEP_SEC = 0.8

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEOS_DIR = os.path.join(OUTPUT_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

OUTPUT_VIDEO_PATH = os.path.join(VIDEOS_DIR, "output_two_lines.mp4")
CSV_EVENTS_PATH = os.path.join(OUTPUT_DIR, "two_lines_events.csv")

SAVE_VIDEO = True
WIN = "Traffic Live - Two Vertical Lines"


# =========================
# 2) YouTube stream url
# =========================
def get_youtube_stream_url(youtube_url: str) -> str:
    import yt_dlp
    ydl_opts = {"quiet": True, "no_warnings": True, "format": "best[ext=mp4]/best"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


# =========================
# 3) Helpers
# =========================
def bbox_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def point_side(p, a, b):
    # sign of cross product (b-a) x (p-a)
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

def speed_kmh_from_history(hist: deque, meters_per_pixel: float):
    if len(hist) < 2:
        return None
    t0, x0, y0 = hist[0]
    t1, x1, y1 = hist[-1]
    dt = t1 - t0
    if dt <= 1e-6:
        return None
    dp = math.hypot(x1 - x0, y1 - y0)
    ds = dp * meters_per_pixel
    return (ds / dt) * 3.6

def clamp_speed(v, vmin=0.0, vmax=250.0):
    if v is None:
        return None
    if v < vmin or v > vmax:
        return None
    return v


# =========================
# 4) Click UI state
# =========================
class ClickState:
    def __init__(self):
        self.mode = None   # None / "lineA" / "lineB" / "calib"
        self.points = []

        # Стартові "вертикальні" лінії (підкрутиш клавішами 1/2 + 2 кліки)
        # Line A (Start) — горизонтальна
        self.lineA_p1 = (120, 250)
        self.lineA_p2 = (1180, 250)

        # Line B (Zebra) — горизонтальна
        self.lineB_p1 = (120, 520)
        self.lineB_p2 = (1180, 520)

        # calibration
        self.calib_known_meters = 0.50
        self.meters_per_pixel = METERS_PER_PIXEL

    def reset(self):
        self.mode = None
        self.points.clear()


click_state = ClickState()

def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if click_state.mode in ("lineA", "lineB", "calib"):
        click_state.points.append((x, y))
        print(f"[CLICK] {click_state.mode} point: {(x, y)}")


# =========================
# 5) CSV
# =========================
def init_csv(path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp_unix",
            "track_id",
            "class_id",
            "class_name",
            "speed_kmh_at_lineB",
            "lineA_time", "lineA_speed_kmh",
            "lineB_time",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "center_x", "center_y"
        ])

def append_csv(path: str, row: list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# =========================
# 6) MAIN
# =========================
def main():
    print("=== Starting ===")
    print("Hotkeys:")
    print("  1 : set Line A (start of road)  - 2 clicks")
    print("  2 : set Line B (on zebra)       - 2 clicks")
    print("  c : calibrate meters-per-pixel  - 2 clicks")
    print("  [ / ] : decrease/increase known meters for calibration")
    print("  q : quit")

    init_csv(CSV_EVENTS_PATH)

    model = YOLO(MODEL_PATH)

    stream_url = get_youtube_stream_url(YOUTUBE_URL)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError("Не вдалося відкрити YouTube stream. Перевір yt-dlp/ffmpeg.")

    writer = None

    # tracking state
    track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))

    # For crossing detection: store last side per line
    last_side_A = {}  # tid -> side sign for line A
    last_side_B = {}  # tid -> side sign for line B

    # Gate logic:
    # - we consider a car "entered road zone" if it crossed line A at least once
    # - we count a car when it later crosses line B
    crossed_A_info = {}  # tid -> (timeA, speedA)
    counted_B_ids = set()
    total_crossed_B = 0
    speeds_at_B = []  # for avg speed at zebra line

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(RECONNECT_SLEEP_SEC)
            cap.release()
            stream_url = get_youtube_stream_url(YOUTUBE_URL)
            cap = cv2.VideoCapture(stream_url)
            continue

        now = time.time()
        h, w = frame.shape[:2]

        if SAVE_VIDEO and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps != fps:
                fps = 30
            writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

        # Draw the two vertical lines
        A1, A2 = click_state.lineA_p1, click_state.lineA_p2
        B1, B2 = click_state.lineB_p1, click_state.lineB_p2

        cv2.line(frame, A1, A2, (0, 255, 255), 2)     # yellow Line A
        # cv2.putText(frame, "Line A (Start)", (A1[0] + 5, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.line(frame, B1, B2, (255, 0, 0), 2)       # blue Line B
        # cv2.putText(frame, "Line B (Zebra)", (B1[0] + 5, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # Info overlay
        cv2.putText(frame, f"m/px: {click_state.meters_per_pixel:.5f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Calib meters: {click_state.calib_known_meters:.2f} ([ ])", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # YOLO track
        results = model.track(frame, conf=CONF_THRESH, tracker=TRACKER, persist=True, verbose=False)
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                class_id = int(cls[i])
                if class_id not in VEHICLE_CLASSES:
                    continue

                x1, y1b, x2, y2b = xyxy[i].astype(int)
                tid = int(ids[i])
                class_name = model.names[class_id]

                cx, cy = bbox_center(x1, y1b, x2, y2b)

                # speed
                track_history[tid].append((now, cx, cy))
                v_kmh = clamp_speed(speed_kmh_from_history(track_history[tid], click_state.meters_per_pixel))

                # Draw bbox + speed
                cv2.rectangle(frame, (x1, y1b), (x2, y2b), (0, 220, 0), 2)
                sp = "N/A" if v_kmh is None else f"{v_kmh:.1f} km/h"
                label = f"{class_name} ID:{tid} {sp}"
                cv2.putText(frame, label, (x1 + 5, y1b + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                # --- Crossing Line A ---
                sideA = point_side((cx, cy), A1, A2)
                crossedA = False
                if tid in last_side_A:
                    prev = last_side_A[tid]
                    if (sideA == 0) or (prev == 0) or (sideA * prev < 0):
                        if tid not in crossed_A_info:
                            crossedA = True
                            crossed_A_info[tid] = (now, v_kmh if v_kmh is not None else "")
                last_side_A[tid] = sideA

                if crossedA:
                    cv2.putText(frame, "CROSSED A", (x1 + 5, y1b + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

                # --- Crossing Line B (Zebra) ---
                sideB = point_side((cx, cy), B1, B2)
                crossedB = False
                if tid in last_side_B:
                    prev = last_side_B[tid]
                    if (sideB == 0) or (prev == 0) or (sideB * prev < 0):
                        # Count only if crossed A before, and not counted at B yet
                        if (tid in crossed_A_info) and (tid not in counted_B_ids):
                            crossedB = True
                            counted_B_ids.add(tid)
                            total_crossed_B += 1

                            speedB = v_kmh if v_kmh is not None else ""
                            if isinstance(speedB, float):
                                speeds_at_B.append(speedB)

                            timeA, speedA = crossed_A_info.get(tid, ("", ""))
                            # CSV event on crossing B
                            append_csv(CSV_EVENTS_PATH, [
                                now,
                                tid,
                                class_id,
                                class_name,
                                speedB,
                                timeA,
                                speedA,
                                now,
                                int(x1), int(y1b), int(x2), int(y2b),
                                float(cx), float(cy)
                            ])
                last_side_B[tid] = sideB

                if crossedB:
                    cv2.putText(frame, "CROSSED B (ZEBRA)", (x1 + 5, y1b + 68),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA)

        # HUD: total at zebra line + avg speed at zebra line
        avgB = float(np.mean(speeds_at_B)) if len(speeds_at_B) else None
        cv2.putText(frame, f"Count (crossed Line B): {total_crossed_B}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        avg_txt = "Avg speed: N/A" if avgB is None else f"Avg speed: {avgB:.1f} km/h"
        cv2.putText(frame, avg_txt, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        # ---- Apply click modes ----
        # Line A: 2 clicks
        if click_state.mode == "lineA" and len(click_state.points) >= 2:
            click_state.lineA_p1 = click_state.points[0]
            click_state.lineA_p2 = click_state.points[1]
            click_state.reset()
            print(f"[LINE A] Set: {click_state.lineA_p1} -> {click_state.lineA_p2}")

        # Line B: 2 clicks
        if click_state.mode == "lineB" and len(click_state.points) >= 2:
            click_state.lineB_p1 = click_state.points[0]
            click_state.lineB_p2 = click_state.points[1]
            click_state.reset()
            print(f"[LINE B] Set: {click_state.lineB_p1} -> {click_state.lineB_p2}")

        # Calibration: 2 clicks
        if click_state.mode == "calib" and len(click_state.points) >= 2:
            p1, p2 = click_state.points[0], click_state.points[1]
            dist_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist_px > 1e-6:
                click_state.meters_per_pixel = click_state.calib_known_meters / dist_px
                print(f"[CALIB] dist_px={dist_px:.2f}, meters={click_state.calib_known_meters:.2f} => m/px={click_state.meters_per_pixel:.6f}")
            click_state.reset()

        # show/save
        cv2.imshow(WIN, frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('1'):
            click_state.mode = "lineA"
            click_state.points = []
            print("[MODE] Set Line A: click 2 points (make it vertical)")

        if key == ord('2'):
            click_state.mode = "lineB"
            click_state.points = []
            print("[MODE] Set Line B: click 2 points (make it vertical)")

        if key == ord('c'):
            click_state.mode = "calib"
            click_state.points = []
            print("[MODE] CALIB: click 2 points with known real distance")

        if key == ord('['):
            click_state.calib_known_meters = max(0.05, click_state.calib_known_meters - 0.05)
            print(f"[CALIB] known_meters = {click_state.calib_known_meters:.2f}")

        if key == ord(']'):
            click_state.calib_known_meters = min(20.0, click_state.calib_known_meters + 0.05)
            print(f"[CALIB] known_meters = {click_state.calib_known_meters:.2f}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("=== Done ===")
    print("CSV:", CSV_EVENTS_PATH)
    if SAVE_VIDEO:
        print("Video:", OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()
