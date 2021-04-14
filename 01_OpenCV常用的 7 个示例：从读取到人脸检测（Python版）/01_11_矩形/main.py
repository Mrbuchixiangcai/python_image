import cv2
import numpy as np

#/*! 要在图像上绘制一个矩形，我们使用 cv2.rectangle 函数。在函数中，我们将宽度、 */
#/*! 高度、x、 y、 RGB 中的颜色、深度作为参数传递。*/

img = cv2.imread("image.jpg")
print(img.shape)

# cv2.rectangle(img,(w,h),(x,y),(R,G,B),THICKNESS)
# w：宽度
# h：身高
# x：距x轴的距离
# y：与y轴的距离
# R，G，B：RGB形式的颜色（255,255,0）
# 厚度：矩形的厚度（整数）
cv2.rectangle(img, (100, 300), (200, 300), (255, 0, 255), 2)

cv2.imshow("Image", img)

cv2.waitKey(0)
