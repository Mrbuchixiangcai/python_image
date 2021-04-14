import cv2

img = cv2.imread("image.jpg")
print(img.shape)
imgCropped = img[50:250, 120:330]
cv2.imshow("Image cropped", imgCropped)
cv2.imshow("Image", img)

cv2.waitKey(0)