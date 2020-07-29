from spectral import *
import cv2
import numpy as np
import os


def classify():
    img = open_image(r'E:\HE+CAM5\PreproEasy\HE_CAM52_mono_E_roi1_prePro.hdr').load()
    gt1 = cv2.imread(r'E:\HE+CAM5\PreproEasy\rgb-down\roi1_SVM.jpg')
    #  gt2 = cv2.imread(r'E:\HE+CAM5\PreproEasy\rgb-down\roi1_SVMcvt_1.jpg')
    gt = np.asarray(gt1[:, :, 0])
    (m, n) = gt.shape
    # print(gt)
    for i in range(m):
        for j in range(n):
            if gt[i, j] < 127:
                gt[i, j] = 10
            else:
                gt[i, j] = 255

    classes = create_training_classes(img, gt)
    gmlc = MahalanobisDistanceClassifier(classes)
    return gmlc

# 针对随机取得p，q两个数的素性检测
def miller_rabin_test(n):  # n为要检验得数
    p = n - 1
    r = 0
    # 寻找满足n-1 = 2^s  * m 的s,m两个数
    #  n -1 = 2^r * p
    while p % 2 == 0:  # 最后得到为奇数的p(即m)
        r += 1
        p /= 2
    b = random.randint(2, n - 2)  # 随机取b=（0.n）
    # 如果情况1    b得p次方  与1  同余  mod n
    if fastExpMod(b, int(p), n) == 1:
        return True  # 通过测试,可能为素数
    # 情况2  b得（2^r  *p）次方  与-1 (n-1) 同余  mod n
    for i in range(0,7):  # 检验六次
        if fastExpMod(b, (2 ** i) * p, n) == n - 1:
            return True  # 如果该数可能为素数，
    return False  # 不可能是素数


def imgClass(file_dir):
    gmlc = classify()
    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith('.hdr'):  # 判断是否以.hdr结尾
            print('doing img:' + files)
            file = file_dir + '\\' + files
            img2 = open_image(file).load()
            clmap = gmlc.classify_image(img2)
            (filename, extension) = os.path.splitext(files)
            outputway = r'E:\HE+CAM5\PreproEasy\rgb-down\testb\{a}.jpg'.format(a=filename)
            save_rgb(outputway, clmap)


if __name__ == '__main__':
    imgWay = r'E:\HE+CAM5\PreproEasy'
    imgClass(imgWay)

