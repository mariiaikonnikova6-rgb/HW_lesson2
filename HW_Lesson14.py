import cv2
import numpy as np
import shutil
import os

SCRIPT_DIR = os.path.dirname(__file__)

PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'out')
PEOPLE_DIR = os.path.join(OUTPUT_DIR, 'people')
NO_PEOPLE_DIR = os.path.join(OUTPUT_DIR, 'no_people')


os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

MODEL_DIR_CANDIDATES = [
    MODELS_DIR,
    os.path.join(SCRIPT_DIR, 'models'),
    os.path.join(PROJECT_DIR, '..', 'data', 'MobileNet'),
]

prototxt_candidates = []
model_candidates = []
for model_dir in MODEL_DIR_CANDIDATES:
    prototxt_candidates.extend([
        os.path.join(model_dir, 'MobileNetSSD_deploy.prototxt'),
        os.path.join(model_dir, 'MobileNetSSD_deploy.prototxt.txt'),
        os.path.join(model_dir, 'mobilenet_deploy.prototxt'),
    ])
    model_candidates.extend([
        os.path.join(model_dir, 'MobileNetSSD.caffemodel'),
        os.path.join(model_dir, 'MobileNetSSD_deploy.caffemodel'),
        os.path.join(model_dir, 'mobilenet.caffemodel'),
    ])

PROTOTXT_PATH = first_existing(prototxt_candidates)
MODEL_PATH = first_existing(model_candidates)

if PROTOTXT_PATH is None:
    raise FileNotFoundError("Model config not found (.prototxt)")
if MODEL_PATH is None:
    raise FileNotFoundError("Model weights not found (.caffemodel)")

net = cv2.dnn.readNet(PROTOTXT_PATH, MODEL_PATH)


CLASSES = [
"background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor"
]


PERSON_CLASS_ID = CLASSES.index('person')

CONF_THRESHOLD = 0.1

def detect_person(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (h, w) = image.shape[:2]


    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), (127.5, 127.5, 127.5))

    net.setInput(blob)
    detections = net.forward()

    best_conf = 0.0
    best_box = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = detections[0, 0, i, 1]

        if class_id == PERSON_CLASS_ID and confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]

            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            if confidence > best_conf:
                best_conf = confidence
                best_box = (x1, y1, x2, y2)

    found = best_box is not None
    return found, best_box, best_conf


def detect_people(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (h, w) = image.shape[:2]


    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), (127.5, 127.5, 127.5))

    net.setInput(blob)
    detections = net.forward()
    list = []
    best_conf = 0.0
    best_box = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = detections[0, 0, i, 1]

        if class_id == PERSON_CLASS_ID and confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]

            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            # if confidence > best_conf:
            best_conf = confidence
            best_box = (x1, y1, x2, y2)
            list.append(best_box)

    found = best_box is not None
    return list, best_conf

allow_extensions = ['jpg', 'jpeg', 'png', 'webp']
files = os.listdir(IMAGES_DIR)

count_people = 0
count_no_people = 0

for file in files:
    if not file.lower().endswith(tuple(allow_extensions)):
        continue

    in_path = os.path.join(IMAGES_DIR, file)

    img = cv2.imread(in_path)
    if img is None:
        continue

    list, best_conf = detect_people(img)
    if len(list)>0:
        out_path = os.path.join(PEOPLE_DIR, file)
        shutil.copyfile(in_path, out_path)
        count_people += 1

        boxed = img.copy()

        for box in list:
            x1, y1, x2, y2 = box
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 2)

        label = f"Person {best_conf:.2f}, People count {len(list)}"
        cv2.putText(boxed, label, (0, max(0 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



        boxed_path = os.path.join(PEOPLE_DIR, "boxed_" + file)
        cv2.imwrite(boxed_path, boxed)

    else:
        count_no_people += 1
        out_path = os.path.join(NO_PEOPLE_DIR, file)
        shutil.copyfile(in_path, out_path)

print(f'people: {count_people}, no_people: {count_no_people}')
