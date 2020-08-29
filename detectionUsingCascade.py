import numpy as numpy
import cv2

cascade = cv2.CascadeClassifier('cascade.xml')

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

boxes = cascade.detectMultiScale(gray, 1.01, 7)

for (x,y,w,h) in boxes:
    img = cv2.rectangle(img,(x,y),(x+w,h+h),(255,0,0),2)

cv2.imwrite('output.jpg', img)

