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

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, "13403360-hd_1920_1080_30fps.mp4")
    print(VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)


model = YOLO('yolov8n.pt')

CONF_TRESHOLD = 0.5

RESIZE_WIDTH = 960

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
                transport_count += 1
                psevdo_id += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                descr = TRANSPORT_DICTIONARY.get(cls)[0]
                descr_count = TRANSPORT_DICTIONARY.get(cls)[1] + 1
                TRANSPORT_DICTIONARY[cls] = (descr, descr_count)

                label = f'{descr}'
                cv2.putText(frame, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)


                now = time.time()
                dt = now - prev_time
                prev_time = now

                if dt > 0:
                    fps = 1.0 / dt

                str = ''

                for key, value in TRANSPORT_DICTIONARY.items():
                    if value[1]>0.5:
                        str += f' {value[0]}:{value[1]} '

                cv2.putText(frame, f'Transport_count: {str}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                cv2.imshow("YOLO", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
cv2.destroyAllWindows()