import cv2

img = cv2.imread("image.jpg")
print(img.shape)
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #cv2.COLOR_BGR2HSV属于cv.CODE
cv2.imshow("Image", img)
cv2.imshow("Image HSV", imgHSV)

cv2.waitKey(0)