import cv2
import numpy as np

#/*! 膨胀被用来增加图像中边缘的大小。首先，我们定义了奇数(5,5)的核矩阵大小。然 */
#/*! 后使用内核，我们对图像执行膨胀。下面我们对 Canny 算子的输出图像进行了膨胀 */

img = cv2.imread("image.jpg")
print(img.shape)

imgCanny = cv2.Canny(img, 100, 150)   #Canny算子
kernel = np.ones((5, 5), np.uint8)   #界定内核为5*5
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)   #dialation膨胀

cv2.imshow("Image", img)
cv2.imshow("Image Canny", imgCanny)
cv2.imshow("Image Dialation", imgDialation)

cv2.waitKey(0)
