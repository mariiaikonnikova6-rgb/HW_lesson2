import cv2
import numpy as np

card = cv2.imread("ph/background.png")
card = cv2.resize (card, (600, 400))

cv2.rectangle(card,  (5, 5),  (595, 395), (240, 27, 20), 2)

photo = cv2.imread("mar30-09.jpg")
max_w, max_h = 120, 120
h, w = photo.shape[:2]
scale = min(max_w / w, max_h / h)
new_w = int(w * scale)
new_h = int(h * scale)
photo_resized = cv2.resize(photo, (new_w, new_h), interpolation=cv2.INTER_AREA)

x, y = 20, 50
h, w = photo_resized.shape[:2]
card[y:y+h, x:x+w] = photo_resized


cv2.putText(card, "Mariia Ikonnikova", (190, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 0))
cv2.putText(card, "Computer Vision Student ", (190, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 4, 13))
cv2.putText(card, "Email: mariiaikonnikova@gmail.com ", (190, 200), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (20,  20, 240))
cv2.putText(card, "Phone: +380508625573 ", (190, 230), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (20,  20, 240))
cv2.putText(card, "16/06/2010", (190, 270), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (20,  20, 240))

photo_2 = cv2.imread("ph/qr_github.png")
max_w, max_h = 100, 100
h, w = photo_2.shape[:2]
scale = min(max_w / w, max_h / h)
new_w = int(w * scale)
new_h = int(h * scale)
photo_resized_2 = cv2.resize(photo_2, (new_w, new_h), interpolation=cv2.INTER_AREA)

x, y = 450, 270
h, w = photo_resized_2.shape[:2]
card[y:y+h, x:x+w] = photo_resized_2

cv2.putText(card, "OpenCV Business Card ", (80, 340), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0))

cv2.imshow("image", card)

cv2.imwrite("business_card.png", card)
cv2.waitKey(0)
cv2.destroyAllWindows()
