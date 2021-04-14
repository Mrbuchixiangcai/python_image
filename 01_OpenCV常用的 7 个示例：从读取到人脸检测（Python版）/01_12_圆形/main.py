import cv2
import numpy as np

#/*! 为了绘制一个圆形，我们使用 cv2.circle 函数。我们传递 x，y，半径大小，RGB 颜色，深 */
#/*! 度作为参数*/

img = cv2.imread("image.jpg")
print(img.shape)

# cv2.circle(img,(x,y),radius,(R,G,B),THICKNESS)
# x：距x轴的距离
# y：与y轴的距离
# radius：半径大小（整数）
# R，G，B：RGB形式的颜色（255,255,0）
# 厚度：矩形的厚度（整数）
cv2.circle(img, (200, 130), 90, (255, 255, 0), 2)

cv2.imshow("Image", img)

cv2.waitKey(0)
