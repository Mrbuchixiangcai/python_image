import cv2

img = cv2.imread("image.jpg")
print(img.shape)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #cv2.COLOR_BGR2GRAY属于cv.CODE
cv2.imshow("Image", img)
cv2.imshow("Image Gray", imgGray)

cv2.waitKey(0)