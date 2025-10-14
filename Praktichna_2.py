import cv2
import numpy as np

img = cv2.imread('image/figures.png')
scale = 2
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
img_copy = img.copy()

img = cv2.GaussianBlur(img, (5, 5), 4)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower = np.array([2, 17, 0])
# upper = np.array([179, 255, 255])
#
# mask = cv2.inRange(img, lower, upper)
# img = cv2.bitwise_and(img, img, mask=mask)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 62, 0])
upper_red = np.array([179, 255, 255])
mask_red = cv2.inRange(hsv, lower_red, upper_red)
lower_blue = np.array([16, 23, 23])
upper_blue = np.array([179, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
lower_green = np.array([179, 255, 0])
upper_green = np.array([179, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)

mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_green)

img = cv2.bitwise_and(img, img, mask=mask_total)

contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)

        perimeter = cv2.arcLength(cnt, True)



        M = cv2.moments(cnt)

        if M['m00'] == 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        aspect_ratio = round(w / h)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4:
            shape = "square"
        elif len(approx) == 3:
            shape = "triangle"
        elif len(approx) > 8:
            shape ="oval"
        else: shape = "inshe"

        perimeter = cv2.arcLength(cnt, True)
        cv2.putText(img_copy, f'S:{area}, P : {perimeter}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.circle(img_copy, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img_copy, f'shape:{shape}', (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


for i in hsv:
    if h < 10 or h > 160:
        a = 'red'
    elif h < 25 or h > 11:
        a = 'green'
    elif h < 35 or h > 26:
        a = 'blue'
    else:
        a = 'another color'

    cv2.putText(img_copy, 'a', (70,400),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)



cv2.imshow('img', img)
cv2.imshow('mask_total', img_copy)

cv2.imwrite('result.png', img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()
