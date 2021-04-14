import cv2
import numpy as np

#/*! 侵蚀与膨胀正好相反。该算法用于减小图像中边缘的大小。首先，我们定义了奇数(5,5) */
#/*! 矩阵大小。然后使用内核，我们对图像执行腐蚀。下面我们对 Canny 算子的输出图像 */
#/*! 蚀处理。 */

img = cv2.imread("image.jpg")
print(img.shape)

imgCanny = cv2.Canny(img, 100, 150)   #Canny算子
kernel = np.ones((2, 2), np.uint8)   #界定内核为5*5
imgerode = cv2.erode(imgCanny, kernel, iterations=1)   #erosion腐蚀

cv2.imshow("Image", img)
cv2.imshow("Image Canny", imgCanny)
cv2.imshow("Image erode", imgerode)

cv2.waitKey(0)
