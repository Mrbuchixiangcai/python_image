# -*- coding: utf-8 -*-

"""
/***********************
* * 来源网址及使用说明:https://blog.csdn.net/hujingshuang/article/details/47337707
* *                https://blog.csdn.net/ppp8300885/article/details/71078555
***********************/
"""


"""
/***********************
* * First Part
* * code1：只有灰度和gamma校正
* * 读入彩色图像，并转换为灰度值图像, 获得图像的宽和高。采用Gamma校正法对输入图像进行颜色空间的标准化（归一化），
* * 目的是调节图像的对比度，降低图像局部的阴影和光照变化所造成的影响，同时可以抑制噪音。采用的gamma值为0.5。
* * 1、灰度化
* * 对于彩色图像，将RGB分量转化成灰度图像，其转化公式为：
* *		Gray = 0.3 * R + 0.59 * G + 0.11 * B
* * 2、Gamma校正
* * 在图像照度不均匀的情况下，可以通过Gamma校正，将图像整体亮度提高或降低。在实际中可以采用两种不同的方式进行Gamma
* * 标准化，平方根、对数法。这里我们采用平方根的办法，公式如下（其中γ=0.5）：
* *		Y(x, y) = I(x,y)^γ
***********************/
"""

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
cv2.imshow("原图", img)

# img = np.array(img, dtype=np.float) # 把全部的img数组从int型转换成float
# print(img)

img = np.sqrt(img / float(np.max(img)))  # 数组开方//Gamma校正：对每隔像素开方之后值的范围为[0,1]，
# img3 = np.array(img3, dtype=np.int16) # 把float转换成int
print(img.shape)
print(img)

cv2.imshow("Gamma校正", img)

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
"""

# ******************************************************************************************************** #

"""
/***********************
* * Second Part
* * code2：灰度、gamma校正和梯度
* * 计算图像横坐标和纵坐标方向的梯度，并据此计算每个像素位置的梯度方向值；求导操作不仅能够捕获轮廓，人影和一些纹理信息，
* * 还能进一步弱化光照的影响。在求出输入图像中像素点（x,y）处的水平方向梯度、垂直方向梯度和像素值，从而求出梯度幅值和方
* * 向。
* * 用梯度算子对原图做卷积运算：
* * 常用的方法是：首先用[-1,0,1]梯度算子对原图像做卷积运算，得到x方向（水平方向，以向右为正方向）的梯度分量gradscalx，
* * 然后用[1,0,-1]T梯度算子对原图像做卷积运算，得到y方向（竖直方向，以向上为正方向）的梯度分量gradscaly。然后再用以上
* * 公式计算该像素点的梯度大小和方向。
***********************/
"""

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
cv2.imshow("原图", img)

# img = np.array(img, dtype=np.float) # 把全部的img数组从int型转换成float
# print(img)

img = np.sqrt(img / float(np.max(img)))  # 数组开方//Gamma校正：对每隔像素开方之后值的范围为[0,1]，
# img3 = np.array(img3, dtype=np.int16) # 把float转换成int
print(img.shape)
print(img)

cv2.imshow("Gamma校正", img)

height, width = img.shape   # 得到图片的高、宽
gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)   # 求x方向梯度
gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)   # 求y方向梯度
gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0) # 求梯度
gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)   # 求角度
print(gradient_magnitude.shape)
print(gradient_angle.shape)

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
"""

# ******************************************************************************************************** #

"""
/***********************
* * Third Part
* * code3：为每个细胞单元构建梯度方向直方图
* * 我们将图像分成若干个“单元格cell”，默认我们将cell设为8*8个像素。假设我们采用8个bin的直方图来统计这6*6个像素的梯
* * 度信息。也就是将cell的梯度方向360度分成8个方向块，例如：如果这个像素的梯度方向是0-22.5度，直方图第1个bin的计数就
* * 加一，这样，对cell内每个像素用梯度方向在直方图中进行加权投影（映射到固定的角度范围），就可以得到这个cell的梯度方向
* * 直方图了，就是该cell对应的8维特征向量而梯度大小作为投影的权值。
***********************/
"""

"""
import sys
import dlib
import cv2
import math
import numpy as np

# opencv 读取图片，并显示
f = "image_1.jpg"

# part1 这里是Grayscale灰度
img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

print(img)
print(img.shape)
cv2.imshow("原图", img)

# img = np.array(img, dtype=np.float) # 把全部的img数组从int型转换成float
# print(img)

img = np.sqrt(img / float(np.max(img)))  # 数组开方//Gamma校正：对每隔像素开方之后值的范围为[0,1]，
# img3 = np.array(img3, dtype=np.int16) # 把float转换成int
print(img.shape)
print(img)

cv2.imshow("Gamma校正", img)

# part2 求梯度和角度
height, width = img.shape   # 得到图片的高、宽
gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)   # 求x方向梯度
gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)   # 求y方向梯度
gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0) # 求梯度
gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)   # 求角度
print(gradient_magnitude.shape)
print(gradient_angle.shape)

# part3 为每个细胞单元构建梯度方向直方图
cell_size = 8   # cell设定为8*8
bin_size = 8    # 8和bin代表8维特征向量
angle_unit = 360 / bin_size # angle_unit个方向块
gradient_magnitude = abs(gradient_magnitude)
cell_gradient_vector = np.zeros((int(height / cell_size), int(width / cell_size), bin_size))  # 细胞梯度向量
print (cell_gradient_vector.shape)

def cell_gradient(cell_magnitude, cell_angle):
    print ("******** Func cell_gradient ********")
    orientation_centers = [0] * bin_size    # 方向中心点
    # print (cell_magnitude.shape[0])
    # print(cell_magnitude.shape[1])
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]    # 梯度强度
            gradient_angle = cell_angle[k][l]           # 梯度角度
            min_angle = int(gradient_angle / angle_unit) % 8
            max_angle = (min_angle + 1) % bin_size
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    print("******** Func cell_gradient End ********")
    return orientation_centers

print ("******** Start for 1 ********")
for i in range(cell_gradient_vector.shape[0]):
    for j in range(cell_gradient_vector.shape[1]):
        cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
        cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                     j * cell_size:(j + 1) * cell_size]
        print (cell_angle.max())

        cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)
print ("******** Start for 1 End ********")

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
"""

# ******************************************************************************************************** #

"""
/***********************
* * Fourth Part
* * code4：可视化Cell梯度直方图
* * 将得到的每个cell的梯度方向直方图绘出，得到特征图
***********************/
"""

import sys
import dlib
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

...
# /***********************
# * * matplotilib.pyplot解释
# * * 是一个有命令风格的函数集合，它看起来和MATLAB很相似。每一个pyplot函数都使一副图像做出些许改变，例如创建一幅图，在图中
# * * 创建一个绘图区域，在绘图区域中添加一条线等等。在matplotlib.pyplot中，各种状态通过函数调用保存起来，以便于可以随时跟
# * * 踪像当前图像和绘图区域这样的东西。绘图函数是直接作用于当前axes（matplotlib中的专有名词，图形中组成部分，不是数学中的
# * * 坐标系。）
# ***********************/
...

"""
# opencv 读取图片，并显示
f = "image_1.jpg"

# /***********************
# * * part1 这里是Grayscale灰度
# ***********************/
img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

print(img)
print(img.shape)
cv2.imshow("原图", img)

# img = np.array(img, dtype=np.float) # 把全部的img数组从int型转换成float
# print(img)

img = np.sqrt(img / float(np.max(img)))  # 数组开方//Gamma校正：对每隔像素开方之后值的范围为[0,1]，
# img3 = np.array(img3, dtype=np.int16) # 把float转换成int
print(img.shape)
print(img)

cv2.imshow("Gamma校正", img)

# /***********************
# * * part2 求梯度和角度
# ***********************/
height, width = img.shape   # 得到图片的高、宽
gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)   # 求x方向梯度
gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)   # 求y方向梯度
gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0) # 求梯度
gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)   # 求角度
print(gradient_magnitude.shape)
print(gradient_angle.shape)

# /***********************
# * * part3 为每个细胞单元构建梯度方向直方图
# ***********************/
cell_size = 8   # cell设定为8*8
bin_size = 8    # 8和bin代表8维特征向量
angle_unit = 360 / bin_size # angle_unit个方向块
gradient_magnitude = abs(gradient_magnitude)
cell_gradient_vector = np.zeros((int(height / cell_size), int(width / cell_size), bin_size))  # 细胞梯度向量
print (cell_gradient_vector.shape)

def cell_gradient(cell_magnitude, cell_angle):
    print ("******** Func cell_gradient ********")
    orientation_centers = [0] * bin_size    # 方向中心点
    # print (cell_magnitude.shape[0])
    # print(cell_magnitude.shape[1])
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]    # 梯度强度
            gradient_angle = cell_angle[k][l]           # 梯度角度
            min_angle = int(gradient_angle / angle_unit) % 8
            max_angle = (min_angle + 1) % bin_size
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    print("******** Func cell_gradient End ********")
    return orientation_centers

print ("******** Start for 1 ********")
for i in range(cell_gradient_vector.shape[0]):
    for j in range(cell_gradient_vector.shape[1]):
        cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
        cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                     j * cell_size:(j + 1) * cell_size]
        print (cell_angle.max())

        cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)
print ("******** Start for 1 End ********")

# /***********************
# * * part4 将得到的每个cell的梯度方向直方图绘出，得到特征图
# ***********************/
hog_image = np.zeros([height, width])
cell_gradient = cell_gradient_vector
cell_width = cell_size / 2
max_mag = np.array(cell_gradient).max()
for x in range(cell_gradient.shape[0]):
    for y in range(cell_gradient.shape[1]):
        cell_grad = cell_gradient[x][y]
        cell_grad /= max_mag
        angle = 0
        angle_gap = angle_unit
        for magnitude in cell_grad:
            angle_radian = math.radians(angle)
            x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
            y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
            x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
            y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
            cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
            angle += angle_gap

plt.imshow(hog_image, cmap=plt.cm.gray)
plt.show()

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
"""

# ******************************************************************************************************** #

"""
/***********************
* * Fifth Part
* * code5：统计Block的梯度信息
* * 把细胞单元组合成大的块(block），块内归一化梯度直方图
* * 由于局部光照的变化以及前景-背景对比度的变化，使得梯度强度的变化范围非常大。这就需要对梯度强度做归一化。归一化能够进一步
* * 地对光照、阴影和边缘进行压缩。
* * 把各个细胞单元组合成大的、空间上连通的区间（blocks）。这样，一个block内所有cell的特征向量串联起来便得到该block的HOG
* * 特征。这些区间是互有重叠的，
* * 本次实验采用的是矩阵形区间，它可以有三个参数来表征：每个区间中细胞单元的数目、每个细胞单元中像素点的数目、每个细胞的直方
* * 图通道数目。
* * 本次实验中我们采用的参数设置是：2*2细胞／区间、8*8像素／细胞、8个直方图通道,步长为1。则一块的特征数为2*2*8。
***********************/
"""
"""
import sys
import dlib
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

...
# /***********************
# * * matplotilib.pyplot解释
# * * 是一个有命令风格的函数集合，它看起来和MATLAB很相似。每一个pyplot函数都使一副图像做出些许改变，例如创建一幅图，在图中
# * * 创建一个绘图区域，在绘图区域中添加一条线等等。在matplotlib.pyplot中，各种状态通过函数调用保存起来，以便于可以随时跟
# * * 踪像当前图像和绘图区域这样的东西。绘图函数是直接作用于当前axes（matplotlib中的专有名词，图形中组成部分，不是数学中的
# * * 坐标系。）
# ***********************/
...

# opencv 读取图片，并显示
f = "image_1.jpg"

# /***********************
# * * part1 这里是Grayscale灰度
# ***********************/
img = cv2.imread(f, cv2.IMREAD_COLOR)   # 读取图片，格式为灰度
cv2.imshow("IMREAD_COLOR", img)

img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
print(img)
print(img.shape)
cv2.imshow("IMREAD_GRAYSCALE", img)

# img = np.array(img, dtype=np.float) # 把全部的img数组从int型转换成float
# print(img)

img = np.sqrt(img / float(np.max(img)))  # 数组开方//Gamma校正：对每隔像素开方之后值的范围为[0,1]，
# img3 = np.array(img3, dtype=np.int16) # 把float转换成int
print(img.shape)
print(img)

cv2.imshow("Gamma校正", img)

# /***********************
# * * part2 求梯度和角度
# ***********************/
height, width = img.shape   # 得到图片的高、宽
gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)   # 求x方向梯度
gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)   # 求y方向梯度
gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0) # 求梯度
gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)   # 求角度
print(gradient_magnitude.shape)
print(gradient_angle.shape)

# /***********************
# * * part3 为每个细胞单元构建梯度方向直方图
# ***********************/
cell_size = 8   # cell设定为8*8
bin_size = 8    # 8和bin代表8维特征向量
angle_unit = 360 / bin_size # angle_unit个方向块
gradient_magnitude = abs(gradient_magnitude)
cell_gradient_vector = np.zeros((int(height / cell_size), int(width / cell_size), bin_size))  # 细胞梯度向量
print (cell_gradient_vector.shape)

def cell_gradient(cell_magnitude, cell_angle):
    # print ("******** Func cell_gradient ********")
    orientation_centers = [0] * bin_size    # 方向中心点
    # print (cell_magnitude.shape[0])
    # print(cell_magnitude.shape[1])
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]    # 梯度强度
            gradient_angle = cell_angle[k][l]           # 梯度角度
            min_angle = int(gradient_angle / angle_unit) % 8
            max_angle = (min_angle + 1) % bin_size
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    # print("******** Func cell_gradient End ********")
    return orientation_centers

print ("******** Start for 1 ********")
for i in range(cell_gradient_vector.shape[0]):
    for j in range(cell_gradient_vector.shape[1]):
        cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
        cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                     j * cell_size:(j + 1) * cell_size]
        # print (cell_angle.max())

        cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)
print ("******** Start for 1 End ********")

# /***********************
# * * part5 统计Block的梯度信息
# ***********************/
hog_vector = []
for i in range(cell_gradient_vector.shape[0] - 1):
    for j in range(cell_gradient_vector.shape[1] - 1):
        block_vector = []
        block_vector.extend(cell_gradient_vector[i][j])
        block_vector.extend(cell_gradient_vector[i][j + 1])
        block_vector.extend(cell_gradient_vector[i + 1][j])
        block_vector.extend(cell_gradient_vector[i + 1][j + 1])
        mag = lambda vector: math.sqrt(sum(i ** 1 for i in vector))
        magnitude = mag(block_vector)
        if magnitude != 0:
            normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
            block_vector = normalize(block_vector, magnitude)
        hog_vector.append(block_vector)
print (np.array(hog_vector).shape)

# /***********************
# * * part4 将得到的每个cell的梯度方向直方图绘出，得到特征图
# ***********************/
hog_image = np.zeros([height, width])
cell_gradient = cell_gradient_vector
cell_width = cell_size / 2
max_mag = np.array(cell_gradient).max()
for x in range(cell_gradient.shape[0]):
    for y in range(cell_gradient.shape[1]):
        cell_grad = cell_gradient[x][y]
        cell_grad /= max_mag
        angle = 0
        angle_gap = angle_unit
        for magnitude in cell_grad:
            angle_radian = math.radians(angle)
            x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
            y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
            x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
            y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
            cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
            angle += angle_gap

plt.imshow(hog_image, cmap=plt.cm.gray)
plt.show()


# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
"""

# ******************************************************************************************************** #

"""
/***********************
* * Sixth Part
* * code6：代码封装
***********************/
"""

import sys
import dlib
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

...
# /***********************
# * * matplotilib.pyplot解释
# * * 是一个有命令风格的函数集合，它看起来和MATLAB很相似。每一个pyplot函数都使一副图像做出些许改变，例如创建一幅图，在图中
# * * 创建一个绘图区域，在绘图区域中添加一条线等等。在matplotlib.pyplot中，各种状态通过函数调用保存起来，以便于可以随时跟
# * * 踪像当前图像和绘图区域这样的东西。绘图函数是直接作用于当前axes（matplotlib中的专有名词，图形中组成部分，不是数学中的
# * * 坐标系。）
# ***********************/
...

# /***********************
# * * part2 HOG封装成类
# ***********************/
class hog_descriptor():
    # /***********************
    # * * 初始化
    # ***********************/
    def __init__(self, img, cell_size=8, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / np.max(img))   # Gamma校正
        self.img = img * 255                    # 之前代码里面没有这个乘以255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = int(360 / self.bin_size)   # angle_unit个方向块
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        # /***********************
        # * * part 为每个细胞单元构建梯度方向直方图
        # ***********************/
        height, width = self.img.shape  # 读取长宽
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size)) # 细胞梯度向量
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                # print (cell_angle.max())

                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)

        # /***********************
        # * * part 统计Block的梯度信息
        # ***********************/
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
        return hog_vector, hog_image

    # /***********************
    # * * part 求梯度和角度
    # ***********************/
    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)  # 求梯度
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)  # 求角度
        return gradient_magnitude, gradient_angle

    # /***********************
    # * * part
    # ***********************/
    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size # 方向中心点
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    # /***********************
    # * * part
    # ***********************/
    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    # /***********************
    # * * part
    # ***********************/
    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

# opencv 读取图片位置，并显示
f = "image_1.jpg"
# /***********************
# * * part1 代码开始位置
# ***********************/
img = cv2.imread(f, cv2.IMREAD_COLOR)   # 读取图片，格式为彩色
cv2.imshow("IMREAD_COLOR", img)

img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)   # 读取图片，格式为灰度
print(img)
print(img.shape)
cv2.imshow("IMREAD_GRAYSCALE", img)

hog = hog_descriptor(img, cell_size=8, bin_size=8)

vector, image = hog.extract()
print (np.array(vector).shape)
plt.imshow(image, cmap=plt.cm.gray)
plt.show()

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)


# ******************************************************************************************************** #

"""
/***********************
* * Seventh Part
* * code7：这是老版本，不用
***********************/
"""
"""
import cv2
import numpy as np
import math


def showImage(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)


def hog_detector(img):
    height, width = img.shape
    gradient_magnitude, gradient_angle = calc_gradient(img)
    # showImage(gradient_magnitude)
    hog_vector = []
    num_horizontal_blocks = width / 8
    num_vertical_blocks = height / 8
    cell_gradients = np.zeros((int(height / 8), int(width / 8)))
    cell_gradient_vector = []

    ##
    #  calculating cell gradient vector
    ##
    for i in range(0, (height - (height % 8) - 1), 8):
        horizontal_vector = []
        for j in range(0, width - (width % 8) - 1, 8):
            cell_magnitude = gradient_magnitude[i:i + 8, j:j + 8]
            cell_angle = gradient_angle[i:i + 8, j:j + 8]
            horizontal_vector.append(calc_cell_gradient(cell_magnitude, cell_angle))
        cell_gradient_vector.append(horizontal_vector)

    print ("rendering height, width", height, width)
    render_gradient(np.zeros([int(height), int(width)]), cell_gradient_vector)
    height = len(cell_gradient_vector)
    width = len(cell_gradient_vector[0])
    ##
    #  calculating final gradient hog vector
    ##

    for i in range(height - 1):
        for j in range(width - 1):
            vector = []
            vector.extend(cell_gradient_vector[i][j])
            vector.extend(cell_gradient_vector[i][j + 1])
            vector.extend(cell_gradient_vector[i + 1][j])
            vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(vector)
            if magnitude != 0:
                normalize = lambda vector, magnitude: [element / magnitude for element in vector]
                vector = normalize(vector, magnitude)
            hog_vector.append(vector)


##
#   calculate gradient magnitude and gradient angle for image using sobel
##
def calc_gradient(img):
    gradient_values_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    cv2.convertScaleAbs(gradient_values_x)

    gradient_values_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    cv2.convertScaleAbs(gradient_values_y)

    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    return gradient_magnitude, gradient_angle


##
#   calculate gradient for cells into a vector of 9 values
##

def calc_cell_gradient(cell_magnitude, cell_angle):
    orientation_centers = dict([(20, 0), (60, 0), (100, 0), (140, 0), (180, 0), (220, 0), (260, 0), (300, 0), (340, 0)])
    x_left = 0
    y_left = 0
    cell_magnitude = abs(cell_magnitude)
    for i in range(x_left, x_left + 8):
        for j in range(y_left, y_left + 8):
            gradient_strength = cell_magnitude[i][j]
            gradient_angle = cell_angle[i][j]
            min_angle, max_angle = get_closest_bins(gradient_angle, orientation_centers)
            # print gradient_angle, min_angle, max_angle
            if min_angle == max_angle:
                orientation_centers[min_angle] += gradient_strength
            else:
                orientation_centers[min_angle] += gradient_strength * (abs(gradient_angle - max_angle) / 40)
                orientation_centers[max_angle] += gradient_strength * (abs(gradient_angle - min_angle) / 40)
    cell_gradient = []
    for key in orientation_centers:
        cell_gradient.append(orientation_centers[key])
    # print "Cell Magnitude Array : "
    # print cell_magnitude
    # print
    # print "Cell Gradient Vector : "
    # print cell_gradient
    # print
    return cell_gradient


def get_closest_bins(gradient_angle, orientation_centers):
    angles = []
    # print math.degrees(gradient_angle)
    for angle in orientation_centers:
        if abs(gradient_angle - angle) < 40:
            angles.append(angle)
    angles.sort()
    if len(angles) == 1:
        # print gradient_angle, angles[0]
        return angles[0], angles[0]

    return angles[0], angles[1]


##
#	render gradient
##

def render_gradient(image, cell_gradient):
    height, width = image.shape
    height = height - (height % 8) - 1
    width = width - (width % 8) - 1
    x_start = 4
    y_start = 4
    # x_start = height - 8
    # y_start = width - 8
    cell_width = 4
    for x in range(x_start, height, 8):
        for y in range(y_start, width, 8):
            cell_x = x / 8
            cell_y = y / 8
            cell_grad = cell_gradient[int(cell_x)][int(cell_y)]
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            normalize = lambda vector, magnitude: [element / magnitude for element in
                                                   vector] if magnitude > 0 else vector
            cell_grad = normalize(cell_grad, mag(cell_grad))
            angle = 0
            angle_gap = 40
            print (x, y, cell_grad)
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y - magnitude * cell_width * math.sin(angle_radian))
                test_a = (x1, y1)
                test_b = (x2, y2)
                print (test_a)
                print (test_b)
                cv2.line(image, (y1, x1), (y2, x2), 255)
                angle += angle_gap
    h, w = image.shape
    print ("image shpae", h, w)
    showImage(image)


img = cv2.imread('image_1.jpg', 0)
cv2.imshow('原图', img)

height, width = img.shape
print (height, width)

hog_detector(img)

# 等待按键，随后退出，销毁窗口
k = cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
"""
