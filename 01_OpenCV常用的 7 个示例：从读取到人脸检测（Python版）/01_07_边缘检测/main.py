import cv2

#/*! 在 OpenCV 中，我们使用 Canny算子来检测图像中的边缘。也有不同的边缘检 */
#/*! 测器，但最著名的是 Canny算子。Canny算子边缘检测是一种边缘检测算子，它 */
#/*! 使用多级算法来检测图像中的大范围边缘，是由 John F. Canny 在1986年提 */
#/*! 出的。 */

img = cv2.imread("image.jpg")
print(img.shape)
#imgCanny = cv2.Canny(img,threshold1,threshold2)
#threshold1，threshold2：每个图像的阈值不同
imgCanny = cv2.Canny(img, 100, 150) #Canny算子
cv2.imshow("Image", img)
cv2.imshow("Image Canny", imgCanny)

cv2.waitKey(0)