# done
from sklearn import svm
from spectral import *
import cv2
import numpy as np
import os
from sklearn.decomposition import NMF
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import validation_curve


def inf_flt(data, label):
    p, q = data.shape
    for i in range(p-1):
        for j in range(q-1):
            if data[i, j] == float("inf"):
                data = np.delete(data, i, axis=0)
                label = np.delete(label, i, axis=0)


def inf_flt(data):
    p, q = data.shape
    for i in range(p-1):
        for j in range(q-1):
            if data[i, j] == float("inf"):
                data = np.delete(data, i, axis=0)


def classify(file_dir, hdr_dir, roi_dir1, roi_dir2, out_dir, inf_need=False):
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
                data = img_reshape[i, :]
                count = count + 1
            else:
                label = np.append(label, label1)
                data = np.vstack((data, img_reshape[i, :]))
                count = count + 1
            # print(count)
        elif gt2_flatten[i] > 127:
            if count == 0:
                label = np.array([label2])
                data = img_reshape[i, :]
                count = count + 1
            else:
                label = np.append(label, label2)
                data = np.vstack((data, img_reshape[i, :]))
                count = count + 1

    if inf_need:
        inf_flt(data, label)

    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    print('collect samples: ' + str(count))
    clf = svm.SVC(C=1, kernel='rbf', gamma='auto', decision_function_shape='ovr')
    clf.fit(data, label)
    '''
    param_range = np.logspace(-6, -2.3, 5)
    train_loss, test_loss = validation_curve(
        SVC(), data, label, param_name='gamma', param_range=param_range, cv=10,
        scoring='accuracy')
    train_loss_mean = np.mean(train_loss, axis=1)
    test_loss_mean = np.mean(test_loss, axis=1)

    plt.plot(param_range, train_loss_mean, 'o-', color="r",
             label="Training")
    plt.plot(param_range, test_loss_mean, 'o-', color="g",
             label="Cross-validation")

    plt.xlabel("gamma")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()
    '''
    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith('.hdr'):  # 判断是否以.hdr结尾
            time_start = time.time()
            print('doing img:' + files)
            file = file_dir + '\\' + files
            img2 = open_image(file).load()
            m, n, l = img2.shape
            k = m * n
            img2_reshape = img2.reshape((k, l))
            scaler = StandardScaler()
            scaler.fit(img2_reshape)
            img2_reshape = scaler.transform(img2_reshape)

            if inf_need:
                inf_flt(img2_reshape)

            clmap = clf.predict(img2_reshape)
            res = clmap.reshape((m, n))
            (filename, extension) = os.path.splitext(files)
            outputway = out_dir + r'\{a}_scaler.jpg'.format(a=filename)
            cv2.imwrite(outputway, res)
            time_end = time.time()
            print('time cost: ', time_end-time_start)


if __name__ == '__main__':
    imgWay = r'E:\HE+CAM5\PreproEasy'  # 需要分类的图像路径
    hdrWay = r'E:\HE+CAM5\PreproEasy\HE_CAM52_mono_E_roi1_prePro.hdr'  # 用于分类的图像hdr路径
    roiWay1 = r'E:\HE+CAM5\PreproEasy\roia.jpg'  # 用于分类的图像掩膜路径
    roiWay2 = r'E:\HE+CAM5\PreproEasy\roiz.jpg'
    outWay = r'E:\HE+CAM5\PreproEasy\res'  # 分类后的输出路径
    classify(imgWay, hdrWay, roiWay1, roiWay2, outWay)

