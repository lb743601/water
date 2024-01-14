import numpy as np
import math
from osgeo import gdal, gdal_array  # 导入读取遥感影像的库
from numba import jit
import math
import cv2
from matplotlib import pyplot as plt


def normalize(x):
    x = x.astype(np.float64)
    y = (x - x.min()) / (x.max() - x.min())
    return y




def apply_mask(bgr, mask, index, i):
    # 创建一个与图像尺寸相同的全零数组，作为输出图像
    color_map = [cv2.COLORMAP_PARULA, cv2.COLORMAP_COOL, cv2.COLORMAP_HOT, cv2.COLORMAP_AUTUMN]
    z = np.zeros_like(bgr)
    mask_inv = cv2.bitwise_not(mask)
    # 将mask区域内的像素复制到输出图像中
    output = cv2.bitwise_and(index,index,mask=mask)
    output = cv2.applyColorMap(output, color_map[i])
    bgr_ = cv2.bitwise_and(bgr,bgr,mask=mask_inv)
    bgr = cv2.bitwise_and(output,output,mask=mask)
    bgr = bgr + bgr_
    return bgr

i = 4
color_map = [plt.cm.summer, plt.cm.cool, plt.cm.hot, plt.cm.autumn_r, plt.cm.hot]
q = ['DO', 'NH3N', 'sd', 'BOI', 'COD']
mask = cv2.imread('./quality/6/mask.png', 0)
bgr = cv2.imread(r'C:\Users\Lenovo\Desktop\sprebuilding\output\rgb\5\0.png')
index = cv2.imread('./quality/6/%s.tif' % q[i], 0)
print(index.max())
index[mask == 0] = 0
print(index.max())
index = index/index.max()
print(index.max())
index = index*30
index = np.rot90(index, k=3)
mean = np.mean(index[index != 0])
print('平均值', mean)
# sc = plt.imshow(index, vmin=0, vmax=1, cmap=plt.cm.jet)
sc = plt.imshow(index, cmap=color_map[i])
# sc.set_cmap('hot')# 这里可以设置多种模式
plt.axes().get_xaxis().set_visible(False) # 隐藏x坐标轴
plt.axes().get_yaxis().set_visible(False) # 隐藏y坐标轴
plt.colorbar()# 显示色度条
plt.show()

# dst = apply_mask(bgr, mask, index, i)
# cv2.imwrite('./output/%s.png' % q[i], dst)
# cv2.imshow('1', dst)
# cv2.waitKey()

# # 寻找轮廓
# mask = cv2.imread('NDWI_mask1.png', 0)
# kernel = np.ones((5, 5), np.float64)
# opening = cv2.erode(mask, np.ones((3, 3), np.float64), 2)
# image = opening
# contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 找到最大外轮廓
# max_contour = max(contours, key=cv2.contourArea)
#
# # 创建一个带有最大外轮廓的空白图像
# contour_image = np.zeros_like(image)
# cv2.drawContours(contour_image, [max_contour], -1, 255, 2)
# # 创建一个与原始图像尺寸相同的空白图像
# filled_image = np.zeros_like(image)
#
# # 绘制最大外轮廓并填充
# cv2.drawContours(filled_image, [max_contour], -1, 255, cv2.FILLED)
#
# # 显示原始图像和填充后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Filled Image', filled_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imwrite('end_mask.png', filled_image)