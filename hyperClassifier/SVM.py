# done
from sklearn import svm
from spectral import *
import cv2
import numpy as np
import os
# import time
from sklearn.impute import SimpleImputer


def inf_flt(p, q, data, label):
    for i in range(p-1):
        for j in range(q-1):
            if data[i, j] == float("inf"):
                data = np.delete(data, i, axis=0)
                label = np.delete(label, i, axis=0)


def inf_flt(p, q, data):
    for i in range(p-1):
        for j in range(q-1):
            if data[i, j] == float("inf"):
                data = np.delete(data, i, axis=0)


def read_traindata(hdr_dir, roi_dir1, roi_dir2):
    img = open_image(hdr_dir).load()
    m, n, l = img.shape
    k = m * n
    img_reshape = img.reshape((k, l))
    gt1 = cv2.imread(roi_dir1)
    gt1 = np.asarray(gt1[:, :, 0])
    gt2 = cv2.imread(roi_dir2)
    gt2 = np.asarray(gt2[:, :, 0])
    label1_value = 255
    label2_value = 0
    gt1_reshape = gt1.reshape((k, 1))
    img1_gt1_list = list(np.hstack((gt1_reshape, img_reshape)))
    data1_gt1_list = list(filter(lambda number: number[0] > 127, img1_gt1_list))
    data1_gt1_array = np.asarray(data1_gt1_list)
    data1 = data1_gt1_array[:, 1:]
    count1 = data1.shape[0]
    label1 = np.ones(count1)*label1_value
    gt2_reshape = gt2.reshape((k, 1))
    img2_gt2_list = list(np.hstack((gt2_reshape, img_reshape)))
    data2_gt2_list = list(filter(lambda number: number[0] > 127, img2_gt2_list))
    data2_gt2_array = np.asarray(data2_gt2_list)
    data2 = data2_gt2_array[:, 1:]
    count2 = data2.shape[0]
    label2 = np.ones(count2)*label2_value
    data = np.vstack((data1, data2))
    label = np.hstack((label1, label2))
    return data, label
    

def classify(file_dir, hdr_dir, roi_dir1, roi_dir2, out_dir):
    # time_start = time.time()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    data, label = read_traindata(hdr_dir, roi_dir1, roi_dir2)
    key_nan = np.isnan(data).any()
    key_inf = np.isinf(data).any()
    if key_nan:
        my_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        data = my_imputer.fit_transform(data)
    if key_inf:
        data_inf = np.isinf(data)
        data[data_inf] = 1.2

    count = data.shape[0]
    print('collect samples: ' + str(count))
    clf = svm.SVC(C=1, kernel='rbf', gamma='auto', decision_function_shape='ovr')
    clf.fit(data, label)
    # time_end = time.time()
    # print('totally cost', time_end-time_start)

    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith('.hdr'):  # 判断是否以.hdr结尾
            print('doing img:' + files)
            file = file_dir + '\\' + files
            img2 = open_image(file).load()
            m, n, l = img2.shape
            k = m * n
            img2_reshape = img2.reshape((k, l))
            key_nan2 = np.isnan(img2_reshape).any()
            key_inf2 = np.isinf(img2_reshape).any()
            if key_nan2:
                img2_reshape = SimpleImputer(missing_values=np.nan, strategy="mean")
                img2_reshape = my_imputer.fit_transform(img2_reshape)
            if key_inf2:
                img2_reshape_inf = np.isinf(img2_reshape)
                img2_reshape[img2_reshape_inf] = 1.2
            clmap = clf.predict(img2_reshape)
            res = clmap.reshape((m, n))
            (filename, extension) = os.path.splitext(files)
            outputway = out_dir + r'\{a}_SVM.jpg'.format(a=filename)
            cv2.imwrite(outputway, res)


if __name__ == '__main__':
    imgWay = r'E:\HE+CAM5\2020\cam\preprocess'  # 需要分类的图像路径
    hdrWay = r'E:\HE+CAM5\2020\cam\preprocess\cam-0.hdr'  # 用于分类的图像hdr路径
    roiWay1 = r'E:\HE+CAM5\2020\cam\preprocess\roix.jpg'  # 用于分类的图像掩膜路径
    roiWay2 = r'E:\HE+CAM5\2020\cam\preprocess\roiy.jpg'
    outWay = r'E:\HE+CAM5\2020\cam\preprocess\GBDT'  # 分类后的输出路径
    classify(imgWay, hdrWay, roiWay1, roiWay2, outWay)

