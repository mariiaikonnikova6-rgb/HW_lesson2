import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_red1 = np.array([0, 100, 100])
uper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
uper_red2 = np.array([180, 255, 255])

points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, uper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, uper_red2)

    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    isNotDetected = False
    isDetected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (2, 2), (640, 400), (0, 255, 0), 2)
            isDetected = True
        elif area < 1000:
            cv2.rectangle(frame, (2, 2), (640, 400), (0, 0, 255), 2)
            isNotDetected = True

    if isDetected:
        cv2.putText(frame, f'object detected', (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elif isNotDetected:
        cv2.putText(frame, f'object not detected', (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('video', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
