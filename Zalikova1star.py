import os
import cv2
import time

from gast import descr
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, '')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(OUT_DIR, exist_ok = True)


USE_WEBCAM = False
RESIZE_WIDTH = 960
CONF_TRESHOLD = 0.5

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, "13403360-hd_1920_1080_30fps.mp4")
    print(VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
# ---- ADD: параметри вихідного відео + VideoWriter ----
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# FPS з вхідного відео (для такої ж тривалості)
in_fps = cap.get(cv2.CAP_PROP_FPS)
if not in_fps or in_fps <= 1:
    in_fps = 30.0  # fallback (особливо для webcam)

# Розмір кадру (важливо: якщо ресайзиш - writer має знати саме кінцевий розмір!)
in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if RESIZE_WIDTH is not None and in_w > 0:
    scale = RESIZE_WIDTH / in_w
    out_w = int(in_w * scale)
    out_h = int(in_h * scale)
else:
    out_w, out_h = in_w, in_h

out_name = f"result_{int(time.time())}.mp4"
OUT_PATH = os.path.join(OUT_DIR, out_name)

writer = cv2.VideoWriter(OUT_PATH, fourcc, in_fps, (out_w, out_h))
if not writer.isOpened():
    raise RuntimeError(f"VideoWriter не открылся. OUT_PATH={OUT_PATH}, fps={in_fps}, size={(out_w, out_h)}")
# -----------------------------------------------

model = YOLO('yolov8n.pt')





prev_time = time.time()
FPS = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    result = model(frame, conf = CONF_TRESHOLD, verbose = False)

    transport_count = 0
    psevdo_id = 0

    PERSONS_CLASS_ID = 0

    TRANSPORT_DICTIONARY = {
        1: ("bicycle", 0),
        2: ("car", 0),
        3: ("motorcycle", 0),
        4: ("airplane", 0),
        5: ("bus", 0),
        6: ("train", 0),
        7: ("truck", 0),
        8: ("boat", 0)
    }



    TRANSPORT_CLASS_ID = list(TRANSPORT_DICTIONARY.keys())

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls in TRANSPORT_CLASS_ID:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                descr = TRANSPORT_DICTIONARY[cls][0]
                TRANSPORT_DICTIONARY[cls] = (descr, TRANSPORT_DICTIONARY[cls][1] + 1)

                cv2.putText(frame, descr, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

                now = time.time()
                dt = now - prev_time
                prev_time = now

                if dt > 0:
                    fps = 1.0 / dt

                str = ''

                for key, value in TRANSPORT_DICTIONARY.items():
                    if value[1]>0.5:
                        str += f' {value[0]}:{value[1]} '

                #cv2.putText(frame, f'Transport_count: {str}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                cv2.imshow("YOLO", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    # --- текстовый блок (на текущий кадр) ---
    s = ""
    for _, (name, cnt) in TRANSPORT_DICTIONARY.items():
        if cnt > 0:
            s += f" {name}:{cnt} "
    cv2.putText(frame, f"Transport_count:{s}", (20, 40),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # --- ВОТ ГЛАВНОЕ: запись кадра в файл ---
    writer.write(frame)

    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
writer.release()
cv2.destroyAllWindows()
print("Saved to:", OUT_PATH)