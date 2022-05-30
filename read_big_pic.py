from cv2 import cv2
import numpy as np


pic_path = 'C:/Users/MT/Desktop/test.tiff'  # 分割的图片的位置
pic_target = 'C:/Users/MT/Desktop/big/'  # 分割后的图片保存的文件夹
# 要分割后的尺寸
cut_width = 500
cut_length = 500
# 读取要分割的图片，以及其尺寸等数据
picture = cv2.imread(pic_path)
cv2.imshow('1',picture)

(width, length, depth) = picture.shape
# 预处理生成0矩阵
pic = np.zeros((cut_width, cut_length, depth))
# 计算可以划分的横纵的个数
num_width = int(width / cut_width)
num_length = int(length / cut_length)
# for循环迭代生成
for i in range(0, num_width):
    for j in range(0, num_length):
        pic = picture[i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length, :]
        result_path = pic_target + '{}_{}.jpg'.format(i + 1, j + 1)
        cv2.imwrite(result_path, pic)

print("done!!!")
