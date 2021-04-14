# -*- coding: utf-8 -*-

'''
/***********************
* * 来源网址及使用说明:https://mp.weixin.qq.com/s/RYIEZVPdITcUlXl-xBeDmA
***********************/
'''

'''
/***********************
* * 简介：要创造卡通效果，我们需要注意两件事：边缘和调色板。这就是照片和卡通的区别所在。
* * 为了调整这两个主要部分，我们将经历四个主要步骤：
* * 1.加载图像
* * 2.创建边缘蒙版
* * 3.减少调色板
* * 4.结合边缘蒙版和彩色图像
***********************/
'''

import cv2
import numpy as np
# required if you use Google Colab
from google.colab.patches import cv2_imshow
from google.colab import files

# 创建加载图像函数
def read_file(filename):
    img = cv2.imread(filename)
    cv2_imshow(img)
    return img

# 调用创建的read_file()函数来加载图像
uploaded = files.upload()
filename = next(iter(uploaded))
img = read_file(filename)

# 创建边缘蒙版函数。 # 卡通效果强调图像中边缘的厚度，可以使用cv2.adaptiveThreshold()函数监测图像中的边缘
def edge_mask(img, line_size, blur_value):      # blur_value:模糊值
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 将图像转化为灰度图像
    gray_blur = cv2.medianBlur(gray, blur_value)    # 对模糊灰度图像进行去噪处理，模糊值越大，图像中出现的黑色噪声就越少
    # 自适应阈值函数，定义边缘的线条尺寸，较大的线条尺寸意味着图像中强调的较厚边缘
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

# 调用edge_mask()函数，查看结果
line_size = 7
blur_value = 7
edges = edge_mask(img, line_size, blur_value)
cv2_imshow(edges)

'''
/***********************
* * 说明：减少调色板
* * 照片和图画之间的主要区别--就颜色而言--是每一张照片中不同颜色的数量，图画的颜色比照片的颜色少，因此，
* * 我们使用颜色量化来减少照片中的颜色数目。
* * 色彩量化：
* * 为了进行颜色量化，我们采用OpenCV库提供的K-Means聚类算法。
***********************/

# 颜色数字化
def color_quantization(img, k):
    # Transform the image 转化图片
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria 确定标准
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means 执行K-Means算法
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

# 通过调整k值来确定要应用于图像的颜色数
total_color = 9
img = color_quantization(img, total_color)


/***********************
* * 说明：双边滤波器
* * 在进行颜色量化之后，我们可以使用双边滤波器来降低图像中的噪声。它会给图像带来一点模糊和锐度降低的效果
***********************/

# d:每个像素邻域的直径；   sigmaColor：参数值越大，表示半等色区域越大（等色解释：https://baike.baidu.com/item/%E7%AD%89%E8%89%B2%E5%87%BD%E6%95%B0/22271280?fr=aladdin）
# sigmaSpace:参数的值越大，意味着更远的像素将相互影响，只要他们的颜色足够接近。
blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)


/***********************
* * 说明：结合边缘蒙版和彩色图像
* * 最后一步是将我们之前创建的边缘蒙版与彩色处理图像相结合，为此，请使用cv2.bitwise_add函数
***********************/

cartoon = cv2.bitwise_add(blurred, blurred, mask=edges)
'''

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)


