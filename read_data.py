import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.signal import argrelextrema, savgol_filter
from tqdm import trange, tqdm


def read_raw(file_path):
    rawImage = np.fromfile(file_path, dtype = np.uint16)
    src = np.zeros((697, 696, 256))
    for row in range(697):
        for dim in range(256):
            src[row, :, dim] = rawImage[(dim+row*256)*696:(dim+1+row*256)*696]
    return src


def read_hdr(file_path):
    # 读取HDR文件信息,返回波段
    hdr = open(file_path).read()
    hdr = hdr.split()
    wavelength = []
    flag = 0
    for i in hdr:
        if i == '381.60,':
            flag = 1
        if flag == 1:
            if i[-1] == ',':
                i = i[:-2]
            wavelength.append(float(i))
        if i == '1007.30':
            flag = 0
    return wavelength


def text_save(content, filename, mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()


def text_read(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content


def read_hdo2():
    txt_path = './hb_curve.txt'
    txt = text_read(txt_path)[2:]
    spec_x = []
    hb02 = []
    hb = []
    for i in txt:
        s = i.split('\t')
        spec_x.append(float(s[0]))
        hb02.append(float(s[1]))
        hb.append(float(s[2]))
    return spec_x, hb02, hb


def read_caldata():
    txt_path = './hb_curve.txt'
    txt = text_read(txt_path)[2:]
    spec_x = []
    hb02 = []
    hb = []
    for i in txt:
        s = i.split('\t')
        spec_x.append(float(s[0]))
        hb02.append(float(s[1]))
        hb.append(float(s[2]))
    return spec_x, hb, hb02


def read_melanindata():
    txt_path = './eumelanin.txt'
    txt = text_read(txt_path)[1:]
    spec_x = []
    melanin = []
    for i in txt:
        s = i.split('\t')
        spec_x.append(float(s[0]))
        melanin.append(float(s[2]))
    return spec_x, melanin


def getFileName2(path, suffix):
    input_template_All = []
    input_template_All_Path = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            # print(os.path.join(root, name))
            if os.path.splitext(name)[1] == suffix:
                input_template_All.append(name)
                input_template_All_Path.append(os.path.join(root, name))

    return input_template_All, input_template_All_Path


if __name__ == '__main__':
    save = True
    out_dir = r'E:\nothing\polarization\caiseban_696'
    # 读取RAW数据
    path = r'I:\dataset\our_7\caiseban_696'
    img_list, img_list_all = getFileName2(path, '.raw')

    hdr_path = img_list_all[0].split('.raw')[0] + '.hdr'

    wavelength = read_hdr(hdr_path) # 波长

    cube = []
    for i in range(len(img_list_all)):
        src_path = img_list_all[i]
        src = read_raw(src_path) # 光谱立方体
        src = src / 256
        # src = cv2.normalize(src, src, 0, 256, cv2.NORM_MINMAX) # 归一化
        src = np.rot90(src, axes=(1, 0)).astype(np.float64) # 旋转
        src = src.astype(np.uint8) # 转为8位
        cube.append(src)
        if save:
            out_path = os.path.join(out_dir, str(i+1))
            os.makedirs(out_path, exist_ok=True)
            for j in range(256):
                out_img_path = os.path.join(out_path, '%d.png' % j)
                cv2.imwrite(out_img_path, src[:, :, j])

    cube = np.array(cube)
    print(cube.shape)
    out_dolp_path = os.path.join(out_dir, 'dolp')
    out_aop_path = os.path.join(out_dir, 'aop')
    out_s0_path = os.path.join(out_dir, 's0')
    os.makedirs(out_dolp_path, exist_ok=True)
    os.makedirs(out_s0_path, exist_ok=True)
    for j in trange(256):
        R0 = cube[0, :, :, j].astype(float) / 255
        R45 = cube[1, :, :, j].astype(float) / 255
        R90 = cube[2, :, :, j].astype(float) / 255
        R135 = cube[3, :, :, j].astype(float) / 255

        S0 = (R0 + R90 + R45 + R135) / 2
        S1 = R0 - R90
        S2 = R45 - R135

        Dolp = np.sqrt(S1 * S1 + S2 * S2) / S0
        Aop = (1 / 2) * np.arctan2(S2, S1)
        out_img_path = os.path.join(out_dolp_path, '%d.png' % j)
        aop_path = os.path.join(out_aop_path, '%d.png' % j)
        s0_path = os.path.join(out_s0_path, '%d.png' % j)
        cv2.imwrite(out_img_path, Dolp*255)
        cv2.imwrite(aop_path, Aop * 255)
        cv2.imwrite(s0_path, S0 * 255)

