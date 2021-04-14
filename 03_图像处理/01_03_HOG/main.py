# -*- coding: utf-8 -*-

"""
/***********************
* * 来源网址及使用说明:https://blog.csdn.net/hujingshuang/article/details/47337707
* *                https://blog.csdn.net/ppp8300885/article/details/71078555
***********************/
"""

"""
/***********************
* * Part1
* * code1：只有灰度和gamma校正
* * 1、灰度化
* * 对于彩色图像，将RGB分量转化成灰度图像，其转化公式为：
* *		Gray = 0.3 * R + 0.59 * G + 0.11 * B
* * 2、Gamma校正
* * 在图像照度不均匀的情况下，可以通过Gamma校正，将图像整体亮度提高或降低。在实际中可以采用两种不同的方式进行Gamma
* * 标准化，平方根、对数法。这里我们采用平方根的办法，公式如下（其中γ=0.5）：
* *		Y(x, y) = I(x,y)^γ
***********************/
"""

import sys
import dlib
import cv2
import math
import numpy as np

# opencv 读取图片，并显示
f = "image_1.jpg"

# 这里是Grayscale灰度
img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
print(img)
print(img.shape)

img3 = np.array(img, dtype=np.float) # 把float转换成int
print(img3)

img3 = np.sqrt(img3 / (255.0))  # 数组开方
# img3 = np.array(img3, dtype=np.int16) # 把float转换成int
print(img3.shape)
print(img3)

cv2.imshow("原图", img)
cv2.imshow("Gamma校正", img3)

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)



"""
#first part

import cv2
import numpy as np
img = cv2.imread('image_1.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('原图', img)
# cv2.imwrite("Image-test.jpg", img)
# cv2.waitKey(0)
print(img)
img2 = np.max(img)
print(img)
print(img2)

float(img2)
print(img2)

img = np.sqrt(img / img2)
print(img)
print(img2)
cv2.imshow('Gamma校正', img)
print(img)
# cv2.imwrite("Image-test2.jpg", img)
cv2.waitKey(0)
"""
