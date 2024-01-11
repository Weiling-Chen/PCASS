import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as scio
import torch.nn.functional as F
import cv2
from numpy import *
import numpy as np
import scipy
from timeit import default_timer as timer

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()

"""
去平均值，即每一位特征减去各自的平均值。
计算协方差矩阵。
通过SVD计算协方差矩阵的特征值与特征向量。
对特征值从大到小排序，选择其中最大的k个。然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵
将数据转换到k个特征向量构建的新空间中。
"""
model = models.vgg16(pretrained=True)
model.eval()

def gaussian_filter(img, Hsize, sigma, boedertype=cv2.BORDER_REPLICATE):
    """
    对图像进行高斯滤波
    :param img: 待滤波图像
    :param Hsize: 核的尺寸
    :param sigma: 标准差
    :param boedertype: 边界填充模式
    :return: 滤波后的图像
    """
    r, c = Hsize
    gaussian_kernel = np.multiply(cv2.getGaussianKernel(r, sigma), (cv2.getGaussianKernel(c, sigma)).T)

    filter_img = cv2.filter2D(img, -1, gaussian_kernel, borderType=boedertype)
    return filter_img

def semanticFeature(img):
    # model = models.vgg16(pretrained=True)
    # model.eval()
    n = 0
    #我们枚举整个网络的所有层
    for i,m in enumerate(model.modules()):   #遍历所有的子层
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m,nn.Dropout):
            img = m(img)
        elif isinstance(m, nn.Linear):
            n = n +1
            img = torch.flatten(img, 1)
            #获取第二个全连接层的输出
            img = m(img)
            if n==2:
                break
    fc7 = img.detach().numpy()
    return fc7

dist_txt = open('Path\to\your\image_path.txt', 'r')
dist_path = dist_txt.readlines()
fc7_list = []
fc7_3_list = []
fc7_9_list = []
time = 0
for mos_name in dist_path:
    mos_name = mos_name.rstrip('\n')
    mos_name = mos_name.split(' ')
    mos_name = mos_name[0]
    print(mos_name)
    gray_img = Image.open(mos_name)  # 读取图片
    gray_img = np.array(gray_img)
    gauss_img_3 = gaussian_filter(gray_img, (3,3), 2)
    gauss_img_9 = gaussian_filter(gray_img,(9,9), 4)
    # stacked_img = gray_img
    # gauss_stacked_img_3 = gauss_img_3
    # gauss_stacked_img_9 = gauss_img_9
    stacked_img = np.stack((gray_img,) * 3, axis=-1)
    gauss_stacked_img_3 = np.stack((gauss_img_3,) * 3, axis=-1)
    gauss_stacked_img_9 = np.stack((gauss_img_9,) * 3, axis=-1)
    stacked_img = Image.fromarray(stacked_img)     #需转成Image格式才可以转成tensor
    gauss_img_3 = Image.fromarray(gauss_stacked_img_3)
    gauss_img_9 = Image.fromarray(gauss_stacked_img_9)
    stacked_img = stacked_img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    gauss_img_3 = gauss_img_3.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    gauss_img_9 = gauss_img_9.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    stacked_img = img_to_tensor(stacked_img)  # 将图片转化成tensor
    gauss_img_3 = img_to_tensor(gauss_img_3)
    gauss_img_9 = img_to_tensor(gauss_img_9)
    stacked_img = stacked_img.unsqueeze(0)
    gauss_img_3 = gauss_img_3.unsqueeze(0)
    gauss_img_9 = gauss_img_9.unsqueeze(0)
    fc7 = semanticFeature(stacked_img)
    fc7_3 = semanticFeature(gauss_img_3)
    fc7_9 = semanticFeature(gauss_img_9)
    fc7_list.append(fc7)
    fc7_3_list.append(fc7_3)
    fc7_9_list.append(fc7_9)

fc7_list = np.array(fc7_list)
fc7 = mat(fc7_list)
scipy.io.savemat('1.mat',{'fc7':fc7})
fc7_3_list = np.array(fc7_3_list)
fc7_3 = mat(fc7_3_list)
scipy.io.savemat('2.mat',{'fc7_3':fc7_3})
fc7_9_list = np.array(fc7_9_list)
fc7_9 = mat(fc7_9_list)
scipy.io.savemat('3.mat',{'fc7_9':fc7_9})