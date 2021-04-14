import cv2

img = cv2.imread("image.jpg")
print(img.shape)

imgResize = cv2.resize(img,(224,224))   #缩小
imgResize2 = cv2.resize(img,(1024,1024)) #放大
cv2.imshow("image",img)
cv2.imshow("image resize",imgResize)
cv2.imshow("image increase size",imgResize2)
print(imgResize.shape)
cv2.waitKey(0)