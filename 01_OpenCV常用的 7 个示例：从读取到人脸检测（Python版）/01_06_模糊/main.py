import cv2

#/*! 模糊用于去除图像中的噪声，也称为平滑。它是对图像应用低通滤波器的过程。在 OpenCV 中对图像进行模糊，我们常用 GaussianBlur。 */

img = cv2.imread("image.jpg")
print(img.shape)
#imgBlur = cv2.GaussianBlur(img,(sigmaX,sigmaY),kernalSize)
#kernalsize-表示内核大小的Size对象。
#sigmaX-代表X方向上高斯核标准偏差的变量。
#sigmaY-与sigmaX相同
imgBlur = cv2.GaussianBlur(img, (15,15), 0) #低通滤波
cv2.imshow("Image", img)
cv2.imshow("Image blur", imgBlur)

cv2.waitKey(0)