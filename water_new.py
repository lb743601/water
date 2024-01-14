import numpy as np
from osgeo import gdal, gdal_array  # 导入读取遥感影像的库
import cv2
from numba import jit
import math
import read_data
import os
from contours import draw_contours
from buildshp import raster2shp, raster2vector
from water_quality_new import water_quality_test
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter


def find_max_region(mask_sel):
    contours, img2 = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 找到最大区域并填充
    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)

    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)

    max_idx = np.argmax(np.array(area))
    # cv2.fillContexPoly(mask[i], contours[max_idx], 0)
    # 填充最大的轮廓
    cv2.drawContours(mask_sel, contours, max_idx, 255, cv2.FILLED)


    return mask_sel



def water_extract_NDWI(data):
    in_ds = data  # 打开样本文件
    # 读取tif影像信息
    xsize = in_ds.shape[0]  # 获取行列数
    ysize = in_ds.shape[1]
    bands = in_ds.shape[2]   # 获取波段数
    # geotransform = in_ds.GetGeoTransform() # 获取仿射矩阵信息
    # projection = in_ds.GetProjectionRef()   # 获取投影信息
    # block_data = in_ds.ReadAsArray(0,0,xsize,ysize).astype(np.float32) # 获取影像信息
    # block_data = in_ds.ReadAsArray()
    # band1 = in_ds.GetRasterBand(1)
    # datatype = band1.DataType
    print("波段数为：", bands)
    print("行数为：", xsize)
    print("列数为：", ysize)

    # 获取GF-2号多个波段
    B = in_ds[:, :, 24]
    G = in_ds[:, :, 71]
    R = in_ds[:, :, 135]
    NIR = in_ds[:, :, 253]

    np.seterr(divide='ignore', invalid='ignore')    #忽略除法中的NaN值


    # 计算归一化差异水体指数NDWI
    NDWI = (G - NIR) * 1.0 / (G + NIR)
    print(NDWI[300, 300])
    # 生成tif遥感影像
    # writetif(1, xsize, ysize, projection, geotransform, NDWI, "NDWI.tif")
    # print("NDWI图像生成成功！")

    # 根据阈值划分水体与分水体
    threshold = 0.83
    NDWI = go_fast_NDWI(NDWI, threshold)     #使用numba加快for循环运行速度
    # for row in range(NDWI.shape[0]):
    #     for col in range(NDWI.shape[1]):
    #         if NDWI[row, col] >= threshold:
    #             NDWI[row, col] = 1
    #         else:
    #             NDWI[row, col] = 0

    print(NDWI.dtype, NDWI.shape)
    # 最后对图像进行开运算进行去噪，即先腐蚀后膨胀
    kernel = np.ones((5, 5), np.float64)
    opening = cv2.morphologyEx(NDWI, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    # opening = cv2.erode(NDWI, kernel, iterations=1)


    # 生成掩膜二值图——这种保存方法没有地理信息
    # gdal_array.SaveArray(opening.astype(gdal_array.numpy.uint32),
    #                      'NDWI_mask.tif', format="GTIFF", prototype='')
    # print("NDWI二值化掩膜图像生成成功！")

    # 生成掩膜二值图——采用gdal保，包含地理信息等
    # writetif(1, xsize, ysize, projection, geotransform, opening, "NDWI_mask1.tif")
    cv2.imwrite("NDWI_mask1.png", opening*255)
    print("NDWI二值化掩膜图像生成成功！")


    mask_1 = (opening*255).astype(np.uint8)

    # # 生成掩膜二值图——采用opencv简单划分
    retval, NDWI_mask = cv2.threshold(mask_1, 0, 255, cv2.THRESH_BINARY)
    print(NDWI_mask.dtype)
    mask = find_max_region(NDWI_mask)
    cv2.imshow('1', mask)
    cv2.waitKey()
    cv2.imwrite('end_mask.png', mask)
    # 保存为jpg图片
    # cv2.imwrite('NDWI_mask.jpg', NDWI)
    print("NDWI二值化掩膜图像生成成功！")


def writetif(im_bands, xsize, ysize, projection, geotransform, img, outimgname):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(outimgname, xsize, ysize, im_bands, gdal.GDT_Float32)  # 1为波段数量
    dataset.SetProjection(projection)  # 写入投影
    dataset.SetGeoTransform(geotransform)  # 写入仿射变换参数
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(img)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dataset


@jit(nopython=True)
def go_fast_NDWI(NDWI, threshold):
    for row in range(NDWI.shape[0]):
        for col in range(NDWI.shape[1]):
            if NDWI[row, col] >= threshold:
                NDWI[row, col] = 1
            else:
                NDWI[row, col] = 0
    return NDWI


def normalize(x):
    y = (x - x.min()) / (x.max() - x.min())
    return y


if __name__ == '__main__':
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    # plt.plot(x, pp1, 'ro:', label='fit')
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }


    path = r'C:\1\5'
    white_path = r'E:\cal\1\newrawfile20230618141514.raw'
    cube_white = read_data.read_raw(white_path)
    white = cube_white[263, 114, :]
    white = white*(65535/white.max())
    img_list, img_list_all = read_data.getFileName2(path, '.raw')
    print(img_list)
    print(len(img_list))
    hdr_path = img_list_all[0].split('.raw')[0] + '.hdr'
    cube = read_data.read_raw(img_list_all[2])
    print(cube.shape, white.shape)
    cube_new = (cube_white-cube_white.min())/(white - cube_white.min())
    wavethlength = read_data.read_hdr(hdr_path)
    wavethlength = np.array(wavethlength)
    # plt.plot(wavethlength, cube_new[239, 559, :], label="water", color="blue")
    # plt.plot(wavethlength, cube_new[376, 165, :], label="vegetation",color="green")
    # plt.plot(wavethlength, savgol_filter(cube_new[239, 559, :], 7, 3, mode='nearest'), label="water", color="blue")
    # plt.plot(wavethlength, savgol_filter(cube_new[376, 165, :], 7, 3, mode='nearest'), label="vegetation",color="green")
    # plt.plot(wavethlength, cube_white[239, 559, :], label="water", color="blue")
    # plt.plot(wavethlength, cube_white[376, 165, :], label="vegetation",color="green")
    # plt.plot(wavethlength, cube_white[412, 192, :], label="white", color="red")
    # plt.xlabel('Wavelength(nm)', font2)
    # plt.ylabel('Ref', font2)
    # # plt.ylabel('Intensity(a.u.)', font2)
    # plt.legend(loc='best')
    # plt.show()



    print(np.where(wavethlength == 470.7))
    print(np.where(wavethlength == 545.6))
    print(np.where(wavethlength == 700.2))
    print(np.where(wavethlength == 1002.))
    cube = normalize(cube)*255
    cube = cube.astype(np.uint8)
    bgr = cube[:, :, [37, 71, 135]]
    # bgr = bgr*255
    print(bgr.max())
    cv2.imshow('1', bgr)
    cv2.waitKey()
    # bgr = bgr.astype(np.uint8)
    # water_extract_NDWI(cube)
    # draw_contours(bgr, "NDWI_mask1.png", 'NDWI_water.jpg', 'NDWI_river.jpg', 'NDWI_end.jpg', 'NDWI_river_end.jpg')
    # raster2shp('NDWI_river_end.jpg', 'NDWI.shp')
    mask = cv2.imread('mask.png', 0)
    water_quality_test(cube, bgr, mask)
