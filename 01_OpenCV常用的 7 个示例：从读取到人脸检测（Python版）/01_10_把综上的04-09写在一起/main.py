import cv2
import numpy as np

img = cv2.imread("image.jpg")
print(img.shape)

#转换image到gray convert image to gray
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#转换image到HSV convert image to gray
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#高斯模糊image Blur image
imgBlur = cv2.GaussianBlur(img, (3, 3), 1)

#边缘检测 edge detector
imgCanny = cv2.Canny(img, 100, 150)

kernel = np.ones((5, 5), np.uint8)
#image dialation(making edge bigger)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)

#image eroded(making edge lighter)
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

cv2.imshow("Image", img)
cv2.imshow("Image Gray", imgGray)
cv2.imshow("Image HSV", imgHSV)
cv2.imshow("Image Blur", imgBlur)
cv2.imshow("Image Canny", imgCanny)
cv2.imshow("Image Dialation", imgDialation)
cv2.imshow("Image Eroded", imgEroded)

cv2.waitKey(0)
