import cv2
import numpy as np
image = cv2.imread('mar30-09.jpg')
image = cv2.resize(image, (400, 450))


cv2.rectangle(image,  (110, 20), (280, 250), (135, 90, 31), 2)
cv2.putText(image, "Mariia Ikonnikova", (50, 290), cv2.FONT_HERSHEY_TRIPLEX, 1, (135, 90, 31))

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()