#import cv2
from cv2 import cv2
img = cv2.imread("image.jpg")
print(img.shape)
shape = img.shape

imgResize = cv2.resize(img,(shape[0]//2, shape[1]//2))   #缩小
imgResize2 = cv2.resize(img,(shape[0]*4, shape[1]*4)) #放大
cv2.imshow("image",img)
cv2.imshow("image resize",imgResize)
cv2.imshow("image increase size",imgResize2)
print(imgResize.shape)
cv2.waitKey(0)