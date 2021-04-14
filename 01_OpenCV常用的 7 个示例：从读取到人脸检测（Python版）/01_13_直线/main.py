import cv2
import numpy as np

#/*! 要绘制一条直线，我们使用 cv2.line 函数传递起始点(x1，y1)、终点(x2，y2)、 RGB 格式 */
#/*! 的颜色、深度作为参数。*/

img = cv2.imread("image.jpg")
print(img.shape)

# cv2.line(img,(x1,y1),(x2,y2),(R,G,B),THICKNESS)x1,y1: start point of line (integer)
# x2，y2：线的终点（整数）
# R，G，B：RGB形式的颜色（255,255,0）
# THICKNESS：矩形的厚度（整数）
cv2.line(img, (110, 260), (300, 260), (0, 255, 0), 3)

cv2.imshow("Image", img)

cv2.waitKey(0)
