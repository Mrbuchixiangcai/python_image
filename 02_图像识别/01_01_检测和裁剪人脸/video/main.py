'''
/***********************
* * 来源网址及使用说明:https://mp.weixin.qq.com/s/nHn-ogRG_1cVwiZHuNeCmQ
***********************/
'''

import cv2
import numpy as np

#/*! 使用 haarcascade_frontalface_default.xml 分类器来检测图像中的人脸。它将返回图 */
#/*! 像的四个坐标(w，h，x，y)。使用这些坐标，我们要在脸上画一个矩形，然后使用相同的坐标， */
#/*! 继续裁剪人脸。最后使用 imwrite，把裁剪后的图像保存到目录中。 */
# 这个路径参考：https://blog.csdn.net/qq_34713361/article/details/105547389
face_cascade = cv2.CascadeClassifier('D:/Program Files/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# img = cv2.imread("image_1.jpg")   #read the input image
video = cv2.VideoCapture("people.mp4")   #read the imput video

while True:
    success, img = video.read()
    #print(img.shape)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert into grayscale 转成灰度值
    faces = face_cascade.detectMultiScale(imgGray, 1.3, 4)  #detect faces

    # Draw rectangle around the faces 围绕脸部绘制矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # cropping face 裁减脸部
        crop_face = img[y:y + h, x:x + w]
        #saving cropped face
        cv2.imwrite(str(w) + str(h) + '_faces.jpg', crop_face)

    cv2.imshow("Image", img)
    cv2.imshow("Imacropped", crop_face)
#    cv2.imshow("Video", img)
#    if cv2.waitKey(1) & 0xff==ord('q'): ##key 'q' will break the loop
#        break

cv2.waitKey(0)
