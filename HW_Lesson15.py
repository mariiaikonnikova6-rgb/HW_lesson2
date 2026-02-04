import os
import cv2
import time
from ultralytics import YOLO


PROJECT_DIR = os.path.dirname(__file__)

VIDEO_DIR = os.path.join(PROJECT_DIR, '')

OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(OUT_DIR, exist_ok=True)



USE_WEBCAM = False

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)

else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, '1477827_Nature_Animal_3840x2160.mp4')
    print(VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)


model = YOLO('yolov8n.pt')

CONFIDENCE_THRESHOLD = 0.4

RESIZE_WIDTH = 960 #None


prev_time = time.time()
fps = 0.0
while True:
    ret, frame = cap.read()

    if not ret:
        break


    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]

        scale = RESIZE_WIDTH / w

        new_w = int(scale * w)
        new_h = int(scale * h)

        frame = cv2.resize(frame, (new_w, new_h))


    result = model(frame, conf = CONFIDENCE_THRESHOLD, verbose = False)

    people_count = 0
    cat_count = 0
    dog_count = 0
    psevdo_id = 0

    PERSON_CLASS_ID = 0
    CAT_CLASS_ID = 15
    DOG_CLASS_ID = 16
    print(1)
    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])


            if cls == CAT_CLASS_ID or cls == DOG_CLASS_ID:
                psevdo_id += 1
                if cls == CAT_CLASS_ID:
                    cat_count += 1
                if cls == DOG_CLASS_ID:
                    dog_count += 1



                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                label = f'ID: {psevdo_id} conf {conf:.2f}'
                cv2.putText(frame, label, (x1, max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)


                now = time.time()
                dt = now - prev_time
                prev_time = now

                if dt > 0:
                    fps = 1.0 / dt

                cv2.putText(frame, f'Animals count: {people_count}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
                cv2.putText(frame, f'Cat count: {cat_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
                cv2.putText(frame, f'Dog count: {dog_count}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

                cv2.putText(frame, f'FPS: {fps:.1f}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

    cv2.imshow('YOLO', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()