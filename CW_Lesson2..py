import cv2
import numpy as np
img = np.zeros((500, 400, 3), np.uint8)
# img[:] = 135, 90, 31
# # rgb = bgr

# img[100:150, 200:250] = 135, 90, 31

cv2.rectangle(img,  (100, 100), (200, 200), (135, 90, 31), 1)

cv2.line(img, (100, 100), (200, 200), (135, 90, 31), 1)

print(img.shape)
cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (135, 90, 31), 1)
cv2.line(img, ( img.shape[1] // 2, 0 ), (img.shape[1] // 2, img.shape[0]), (135, 90, 31), 1)

cv2.circle(img, (200, 200), 30, (92, 232, 62), -1)
cv2.putText(img, "Komarov Ivan", (40, 150), cv2.FONT_HERSHEY_PLAIN, 2, (135, 90, 31))

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
