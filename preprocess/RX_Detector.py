# done
from sklearn import svm
from spectral import *
import cv2
import numpy as np
import os
from scipy.stats import chi2
# import spectral.io.envi as envi
# import time


def classify(file_dir, out_dir):
    # time_start = time.time()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # time_end = time.time()
    # print('totally cost', time_end-time_start)

    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith('.hdr'):  # 判断是否以.hdr结尾
            file = file_dir + '\\' + files
            img = open_image(file).load()
            rxvals = rx(img)
            nbands = img.shape[-1]
            P = chi2.ppf(0.999, nbands)
            res = 1 * (rxvals > P)
            imshow(res)
            (filename, extension) = os.path.splitext(files)
            outputway = out_dir + r'\{a}.jpg'.format(a=filename)
            result = res * 255
            cv2.imwrite(outputway, result)


if __name__ == '__main__':
    imgWay = r'D:\PreproEasy'  # 需要分类的图像路径
    outWay = r'D:\PreproEasy\res'  # 分类后的输出路径
    classify(imgWay, outWay)

