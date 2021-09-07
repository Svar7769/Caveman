# this reduces the background noise

import cv2

img_file = 'image.jpg'
img = cv2.imread(img_file, cv2.IMREAD_COLOR)
img = cv2.blur(img, (5, 5))

ret, thresh_img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('grey image1',thresh_img)
cv2.waitKey(0)
