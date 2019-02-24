import cv2
import numpy as np

img = cv2.imread("bien.jpg",1)



img_copy = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_copy = cv2.adaptiveThreshold(img_copy,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,1)
cv2.imshow("bien",img_copy)
