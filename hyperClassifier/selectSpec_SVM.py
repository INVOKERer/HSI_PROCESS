# done
from sklearn import svm
from spectral import *
import cv2
import numpy as np
import os
# import spectral.io.envi as envi
# import time


def classify(file_dir, hdr_dir, roi_dir1, roi_dir2, out_dir):
    # time_start = time.time()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    img = open_image(hdr_dir).load()
    m, n, l = img.shape
    k = m * n
    img_reshape = img.reshape((k, l))
    gt1 = cv2.imread(roi_dir1)
    gt = np.asarray(gt1[:, :, 0])
    gt1_flatten = gt.flatten()
    gt2 = cv2.imread(roi_dir2)
    gt2 = np.asarray(gt2[:, :, 0])
    gt2_flatten = gt2.flatten()
    label1 = 255
    label2 = 0
    count = 0
    for i in range(k):
        if gt1_flatten[i] > 127:
            if count == 0:
                label = np.array([label1])
                data = img_reshape[i, 5:35]
                count = count + 1
            else:
                label = np.append(label, label1)
                data = np.vstack((data, img_reshape[i, 5:35]))
                # data = np.concatenate((data, img_reshape[i, :]), axis=0)
                count = count + 1
            # print(count)
        elif gt2_flatten[i] > 127:
            if count == 0:
                label = np.array([label2])
                data = img_reshape[i, 5:35]
                count = count + 1
            else:
                label = np.append(label, label2)
                data = np.vstack((data, img_reshape[i, 5:35]))
                # data = np.concatenate((data, img_reshape[i, :]), axis=0)
                count = count + 1
    p, q = data.shape
    for i in range(p-1):
        for j in range(q-1):
            if data[i, j] == float("inf"):
                data = np.delete(data, i, axis=0)
                label = np.delete(label, i, axis=0)
    print('collect samples: ' + str(count))
    clf = svm.SVC(C=1, kernel='rbf', gamma='auto', decision_function_shape='ovr')
    clf.fit(data, label)
    # time_end = time.time()
    # print('totally cost', time_end-time_start)

    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith('.hdr'):  # 判断是否以.hdr结尾
            file = file_dir + '\\' + files
            print('doing img:' + files)
            img2 = open_image(file).load()
            m, n, l = img2.shape
            k = m * n
            img2_reshape = img2.reshape((k, l))
            for i in range(k-1):
                for j in range(l-1):
                    if img2_reshape[i, j] == float("inf"):
                        img2_reshape[i, j] = 0
            img2_tr = img2_reshape[:, 5:35]
            clmap = clf.predict(img2_tr)
            res = clmap.reshape((m, n))
            (filename, extension) = os.path.splitext(files)
            outputway = out_dir + r'\{a}.jpg'.format(a=filename)
            cv2.imwrite(outputway, res)


if __name__ == '__main__':
    imgWay = r'D:\PreproEasy'                                  # 需要分类的图像路径
    hdrWay = r'D:\PreproEasy\HE_CAM52_mono_E_roi1_prePro.hdr'  # 用于分类的图像hdr路径
    roiWay1 = r'D:\PreproEasy\roia.jpg'                     # 用于分类的图像掩膜路径
    roiWay2 = r'D:\PreproEasy\roiz.jpg'
    outWay = r'D:\PreproEasy\res'                   # 分类后的输出路径
    classify(imgWay, hdrWay, roiWay1, roiWay2, outWay)

