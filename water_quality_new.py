import numpy as np
import math
from osgeo import gdal, gdal_array  # 导入读取遥感影像的库
from numba import jit
import math
import cv2


def apply_mask(bgr, mask, index):
    # 创建一个与图像尺寸相同的全零数组，作为输出图像
    z = np.zeros_like(bgr)

    mask_inv = cv2.bitwise_not(mask)
    # 将mask区域内的像素复制到输出图像中
    output = cv2.bitwise_and(index,index,mask=mask)
    output = cv2.applyColorMap(output, cv2.COLORMAP_HOT)
    bgr_ = cv2.bitwise_and(bgr,bgr,mask=mask_inv)
    bgr = cv2.bitwise_and(output,output,mask=mask)
    bgr = bgr + bgr_

    return bgr


def normalize(x):
    x = x.astype(np.float64)
    y = (x - x.min()) / (x.max() - x.min())
    return y



def water_quality_test(data, bgr, mask):
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
    B = in_ds[:, :, 37]
    G = in_ds[:, :, 71]
    R = in_ds[:, :, 135]
    NIR = in_ds[:, :, 253]

    np.seterr(divide='ignore', invalid='ignore')    #忽略除法中的NaN值

    # 化学水质参数计算
    # from 广州市黑臭水体评价模型构建及污染染溯源研究
    COD = (R / G - 0.5777) / 0.007  # 化学需氧量浓度，单位是mg/L

    # from 广州市黑臭水体评价模型构建及污染染溯源研究
    TP = 4.9703 * np.power((R / G), 11.618)  # 总磷的浓度，单位为mg/L

    # from 基于实测数据的鄱阳湖总氮、 总磷遥感反演模型研究
    TN = 2.2632 * (NIR / G) * (NIR / G) + 1.1392 * (NIR / G) + 0.7451

    # from 广州市黑臭水体评价模型构建及污染染溯源研究
    # 计算结果多为负数，不合理
    # NH3N = (R / G - 0.661) / 0.07       # 氨氮的浓度，单位为mg/L

    # from 基于Landsat-8 OLI影像的术河临沂段氮磷污染物反演
    # 取值范围为：-0.09到1.48 不合理
    # NH3N = 3.1749 * R / G - 1.9909  # 氨氮的浓度，单位为mg/L
    #
    NH3N = 4.519 * ((G - R) / (G + R)) * ((G - R) / (G + R))\
           - 2.1 * (G - R) / (G + R) + 0.47      # 氨氮的浓度，单位为mg/L

    # from 平原水库微污染水溶解氧含量模型反演与验证
    DO = 15.73229 - 30.80257 * (R + G) / 10000    # 溶解氧的浓度，单位为mg/L
    # from 基于自动监测和Sentinel-2影像的钦州湾溶解氧反演模型研究
    # DO = 771.854 * (1 / R) - 1.476 * (R * R) / (B * B) + 6.435

    # 保存影像
    print(COD.dtype, COD.max(), COD.min(), COD[300, 300])
    COD = abs(COD).astype(np.uint8)
    print(COD.dtype, COD.max(), COD.min(), COD[300, 300])
    cv2.imwrite('./quality/COD.tif', COD)
    print("COD1图像生成成功！")
    TP = normalize(TP)*255
    TP = TP.astype(np.uint8)
    cv2.imwrite('./quality/TP.tif', TP)
    print("TP图像生成成功！")

    TN = TN.astype(np.uint8)
    cv2.imwrite('./quality/TN.tif', TN)
    print("TN图像生成成功！")

    NH3N = NH3N.astype(np.uint8)
    cv2.imwrite('./quality/NH3N.tif', NH3N)
    # NH3N = apply_mask(bgr, mask, NH3N)
    # cv2.imwrite('./quality/NH3N.tif', NH3N)
    print("NH3N图像生成成功！")

    DO = DO.astype(np.uint8)
    cv2.imwrite('./quality/DO.tif', DO)
    print("DO图像生成成功！")


    # 水质等级划分
    watergrades_all = R / R    # 随便给定一个初值
    watergrades_all = go_fast_waterquality_all(COD, TP, NH3N, DO, watergrades_all)
    # 保存影像
    watergrades_all = watergrades_all.astype(np.uint8)
    cv2.imwrite('./quality/watergrades_all.tif', watergrades_all)
    print("watergrades_all图像生成成功！")



    # 营养状态指数计算
    # from 基于GF-1影像的洞庭湖区水体水质遥感监测
    chla = 4.089 * (NIR / R) * (NIR / R) - 0.746 * (NIR / R) + 29.7331  # chla 为叶绿素a浓度，单位为mg/m3
    TSS = 119.62 * np.power((R / G), 6.0823)  # tss为总悬浮物浓度，单位为mg / L
    sd = 284.15 * np.power(TSS, -0.67)  # sd 为透明度，单位是cm

    # 保存影像
    print(chla.max(), chla.min(), chla.dtype)
    chla = chla.astype(np.uint8)
    cv2.imwrite('./quality/chla.tif', chla)
    print("chla图像生成成功！")

    TSS = TSS.astype(np.uint8)
    cv2.imwrite('./quality/TSS.tif', TSS)
    print("TSS图像生成成功！")

    sd = sd.astype(np.uint8)
    cv2.imwrite('./quality/sd.tif', sd)
    print("sd图像生成成功！")

    TLI_chla = 25 + 10.86 * np.log(chla)
    TLI_sd = 51.18 - 19.4 * np.log(sd)
    TLI_TP = 94.36 + 16.24 * np.log(TP)
    TLI_TN = 54.53 + 16.94 + np.log(TN)
    # TLI_CODMn = 1.09 + 26.61 * np.log(CODMn)

    TLI = 0.3261 * TLI_chla + 0.2301 * TLI_TP + 0.2192 * TLI_TN + 0.2246 * TLI_sd
    TLIgrades = G / G #随便给定一个初值
    go_fast_TLI(TLI, TLIgrades)
    # 保存影像
    TLI = TLI.astype(np.uint8)
    cv2.imwrite('./quality/TLI.tif', TLI)
    print("TLI图像生成成功！")


    TLIgrades = TLIgrades.astype(np.uint8)
    cv2.imwrite('./quality/TLIgrades.tif', TLIgrades)
    print("TLIgrades图像生成成功！")

    NDBWI = (G - R) / (G + R)   # 黑臭水体指数
    BOI = (G - R) / (B + G + R)
    # 保存影像
    NDBWI = NDBWI.astype(np.uint8)
    cv2.imwrite('./quality/NDBWI.tif', NDBWI)
    print("NDBWI像生成成功！")

    BOI = BOI.astype(np.uint8)
    cv2.imwrite('./quality/BOI.tif', BOI)
    print("BOI图像生成成功！")

    NDBWI_grades = NDBWI
    NDBWI_grades = go_fast_NDBWI(NDBWI, NDBWI_grades)
    BOIgrades = BOI
    BOIgrades = go_fast_BOI(BOI, BOIgrades)
    # 保存影像
    NDBWI_grades = NDBWI_grades.astype(np.uint8)
    cv2.imwrite('./quality/NDBWI_grades.tif', NDBWI_grades)
    print("NDBWI_grades像生成成功！")

    BOIgrades = BOIgrades.astype(np.uint8)
    cv2.imwrite('./quality/BOI_grades.tif', BOIgrades)
    print("BOI_grades图像生成成功！")


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



# 水质等级划分
@jit(nopython=True)
def go_fast_waterquality_all(COD, TP, NH3N, DO, watergrades_all):
    for row in range(COD.shape[0]):
        for col in range(COD.shape[1]):
            if COD[row, col] > 40 or TP[row, col] > 0.4 or NH3N[row, col] > 2.0 or DO[row, col] < 2:
                watergrades_all[row, col] = 6
            elif COD[row, col] > 30 or TP[row, col] > 0.3 or NH3N[row, col] > 1.5 or DO[row, col] < 3:
                watergrades_all[row, col] = 5
            elif COD[row, col] > 20 or TP[row, col] > 0.2 or NH3N[row, col] > 1.0 or DO[row, col] < 5:
                watergrades_all[row, col] = 4
            elif COD[row, col] > 15 or TP[row, col] > 0.1 or NH3N[row, col] > 0.5 or DO[row, col] < 6:
                watergrades_all[row, col] = 3
            elif COD[row, col] > 15 or TP[row, col] > 0.02 or NH3N[row, col] > 0.15 or DO[row, col] < 7.5:
                watergrades_all[row, col] = 2
            elif COD[row, col] <= 15 and TP[row, col] <= 0.02 and NH3N[row, col] <= 0.15 and DO[row, col] >= 7.5:
                watergrades_all[row, col] = 1
            # else:
            #     watergrades_all[row, col] = 0

    return watergrades_all


# 水质营养状态指数划分
# @jit(nopython=True)
def go_fast_TLI(TLI, TLIgrades):
    for row in range(TLI.shape[0]):
        for col in range(TLI.shape[1]):
            if TLI[row, col] < 30:
                TLIgrades[row, col] = 1
            elif (TLI[row, col] >= 30) and (TLI[row, col] < 50):
                TLIgrades[row, col] = 2
            elif (TLI[row, col] >= 50) and (TLI[row, col] < 60):
                TLIgrades[row, col] = 3
            elif (TLI[row, col] >= 60) and (TLI[row, col] < 70):
                TLIgrades[row, col] = 4
            elif TLI[row, col] >= 70:
                TLIgrades[row, col] = 5
            else:
                TLIgrades[row, col] = 0
    return TLI, TLIgrades



# 黑臭水体划分
@jit(nopython=True)
def go_fast_NDBWI(NDBWI, NDBWI_grades):
    for row in range(NDBWI.shape[0]):
        for col in range(NDBWI.shape[1]):
            if NDBWI[row, col] >= 0.115 or NDBWI[row, col] <= -0.05:
                NDBWI_grades[row, col] = 1
                # print(NDBWI_grades[row, col])
            elif NDBWI[row, col] < 0.115 and NDBWI[row, col] > -0.05:
                NDBWI_grades[row, col] = 2
            else:
                NDBWI_grades[row, col] = 0
    return NDBWI_grades

# 黑臭水体划分
@jit(nopython=True)
def go_fast_BOI(BOI, BOIgrades):
    for row in range(BOI.shape[0]):
        for col in range(BOI.shape[1]):
            if BOI[row, col] >= 0.065:
                BOIgrades[row, col] = 1
                # print(BOIgrades[row, col])
            elif BOI[row, col] < 0.065:
                BOIgrades[row, col] = 2
            else:
                BOIgrades[row, col] = 0
    return BOIgrades


