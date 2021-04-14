import cv2
import numpy as np

#/*! 在 OpenCV 中，我们有一个函数 cv2.puttext，用于在特定位置在图像上写入文本。它以图 */
#/*! 像、文本、 x、 y、颜色、字体、字号、粗细作为输入参数。 */

img = cv2.imread("image.jpg")
print(img.shape)

# cv2.putText(img,text,(x,y),FONT,FONT_SCALE,(R,G,B),THICKNESS)
# img：放置文字的图片
# text：要放在图片上的文字
# X：距X轴的文字距离
# Y：距Y轴的文字距离
# FONT：字体类型（所有字体类型）
# FONT_SCALE：字体大小（整数）
# R，G，B：RGB形式的颜色（255,255,0）
# THICKNESS：矩形的厚度（整数）
cv2.putText(img, "Hello,small Yabo", (120, 250), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Image", img)

cv2.waitKey(0)
