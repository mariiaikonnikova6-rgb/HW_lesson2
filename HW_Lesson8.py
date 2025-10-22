
import cv2
import numpy as np

# face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('data/hear_cascad/haarcascade_frontalface_alt2.xml')
face_cascade_cat = cv2.CascadeClassifier('data/hear_cascad/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('data/hear_cascad/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('data/hear_cascad/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10, minSize = (30, 30))# 2 - масштабування 3 - кількість перевірок 4 - мін розмір
    # print(faces) #coordinates
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, minNeighbors = 15, minSize=(15,15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 10, minSize = (20, 20))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

    faces_cat= face_cascade_cat.detectMultiScale(gray, 1.1, 10, minSize = (30, 30))# 2 - масштабування 3 - кількість перевірок 4 - мін розмір
    # print(faces) #coordinates
    for (x, y, w, h) in faces_cat:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)



    cv2.putText(frame, f'Faces detected: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))



    cv2.imshow("tracking face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()