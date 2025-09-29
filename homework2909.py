import cv2
import numpy as np

#1
image = cv2.imread('mariiaphoto.jpg')
cv2.imshow('image', image)
image = cv2.resize(image, (200, 400))
cv2.imshow('size', image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)
edges = cv2.Canny(image, 100, 200)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#2
image = cv2.imread('myemail.jpg')
cv2.imshow('image', image)
image = cv2.resize(image, (450, 200))
cv2.imshow('size', image)
edges = cv2.Canny(image, 100, 450)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()



