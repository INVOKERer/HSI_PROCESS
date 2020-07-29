# failed
from sklearn import svm
from spectral import *
import cv2
import numpy as np
import os
from sklearn.decomposition import NMF
from scipy.optimize import nnls
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time


def inf_flt(data, label):
    p, q = data.shape
    for i in range(p-1):
        for j in range(q-1):
            if data[i, j] == float("inf"):
                data = np.delete(data, i, axis=0)
                label = np.delete(label, i, axis=0)
    return data, label


def inf_flt(data):
    p, q = data.shape
    for i in range(p-1):
        for j in range(q-1):
            if data[i, j] == float("inf"):
                data = np.delete(data, i, axis=0)
    return data


def get_data_label(hdr_dir, roi_dir1, roi_dir2):
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
    print('collect samples: ' + str(count))
    return data, label


def classify(file_dir, hdr_dir, roi_dir1, roi_dir2, out_dir, r=20, inf_need=False):
    time_start = time.time()
    data, label = get_data_label(hdr_dir, roi_dir1, roi_dir2)

    if inf_need:
        data, label = inf_flt(data, label)

    data = np.array(data)
    label = np.array(label)
    data = data.T
    label = label.T
    nmf = NMF(n_components=r, init='random')
    data_W = nmf.fit_transform(data)
    train_H = []
    for i in range(len(data.T)):
        train_H.append(nnls(data_W, data.T[i])[0])
    data = np.array(train_H)
    time_end = time.time()
    print('Collecting samples time cost: ', time_end - time_start)
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    clf = svm.SVC(C=1, kernel='rbf', gamma='auto', decision_function_shape='ovr')
    clf.fit(data, label)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
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
            data2 = np.array(img2_reshape)

            if inf_need:
                data2 = inf_flt(data2)

            data2 = data2.T
            nmf = NMF(n_components=r, init='random')
            data2_W = nmf.fit_transform(data2)
            train2_H = []
            for i in range(len(data2.T)):
                train2_H.append(nnls(data2_W, data2.T[i])[0])
            data2 = np.array(train2_H)
            scaler = StandardScaler()
            scaler.fit(data2)
            data2 = scaler.transform(data2)
            clmap = clf.predict(data2)
            res = clmap.reshape((m, n))
            (filename, extension) = os.path.splitext(files)
            outputway = out_dir + r'\{a}_nmf5.jpg'.format(a=filename)
            cv2.imwrite(outputway, res)
            time_end = time.time()
            print('time cost: ', time_end-time_start)


if __name__ == '__main__':
    imgWay = r'D:\PreproEasy'  # 需要分类的图像路径
    hdrWay = r'D:\PreproEasy\HE_CAM52_mono_E_roi1_prePro.hdr'  # 用于分类的图像hdr路径
    roiWay1 = r'D:\PreproEasy\roia.jpg'  # 用于分类的图像掩膜路径
    roiWay2 = r'D:\PreproEasy\roiz.jpg'
    outWay = r'D:\PreproEasy\res'  # 分类后的输出路径
    classify(imgWay, hdrWay, roiWay1, roiWay2, outWay)

