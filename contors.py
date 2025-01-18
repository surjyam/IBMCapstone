import cv2 as cv
import numpy as np

#Read Image
img = cv.imread('Photos/Cat.jpg')
cv.imshow("Cat", img)

# convert to grayscale
blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

Canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', Canny)

# Another way to find the contours
ret, thresh = cv.threshold(Canny, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh ',thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} controur(s) found!')

cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)