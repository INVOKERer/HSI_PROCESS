from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
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
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def inf_flt(data, label):
    p, q = data.shape
    for i in range(p-1):
        for j in range(q-1):
            if data[i, j] == float("inf"):
                print('has NaN')
                data = np.delete(data, i, axis=0)
                label = np.delete(label, i, axis=0)
    return data, label


def inf_flt1(data):
    p, q = data.shape
    for i in range(p-1):
        for j in range(q-1):
            if data[i, j] == float("inf"):
                data = np.delete(data, i, axis=0)
    return data


def classify(file_dir, hdr_dir, roi_dir1, roi_dir2, out_dir, key_scale=False):
    # time_start = time.time()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
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

    key_inf = np.isinf(data).any()
    if key_inf:
        data[np.isinf(data)] = np.nan
        # data_inf = np.isinf(data)
        # data[data_inf] = 1.2
    # print(np.isinf(data).any())
    key_nan2 = np.isnan(data).any()
    if key_nan2:
        my_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        data = my_imputer.fit_transform(data)
    count = data.shape[0]
    print('collect samples: ' + str(count))
    if key_scale is True:
        scaler1 = StandardScaler()
        scaler1.fit(data)
        data = scaler1.transform(data)
    '''
    gbdt = GradientBoostingClassifier()
    cross_val_score(gbdt, data, label, cv=20, scoring='roc_auc').mean()

    def gbdt_cv(n_estimators, min_samples_split, max_features, max_depth):
        res = cross_val_score(
            GradientBoostingClassifier(n_estimators=int(n_estimators),
                                       min_samples_split=int(min_samples_split),
                                       max_features=min(max_features, 0.999),  # float
                                       max_depth=int(max_depth),
                                       random_state=2
                                       ),
            data, label, scoring='roc_auc', cv=20
        ).mean()
        return res

    gbdt_op = BayesianOptimization(
        gbdt_cv,
        {'n_estimators': (10, 250),
         'min_samples_split': (2, 25),
         'max_features': (0.1, 0.999),
         'max_depth': (5, 15)}
    )
    gbdt_op.maximize()
    print(gbdt_op.max)
    n_estimators = int(gbdt_op.max['params']['n_estimators'])
    min_samples_split = int(gbdt_op.max['params']['min_samples_split'])
    max_features = min(gbdt_op.max['params']['max_features'], 0.999)
    max_depth = int(gbdt_op.max['params']['max_depth'])
    clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                     min_samples_split=min_samples_split,
                                     max_features=max_features,  # float
                                     max_depth=max_depth,
                                     )
    '''
    clf = GradientBoostingClassifier()

    clf.fit(data, label)

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

            key_inf2 = np.isinf(img2_reshape).any()
            if key_inf2:
                img2_reshape[np.isinf(img2_reshape)] = np.nan

            key_nan2 = np.isnan(img2_reshape).any()
            if key_nan2:
                my_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
                img2_reshape = my_imputer.fit_transform(img2_reshape)
                # img2_reshape_inf = np.isinf(img2_reshape)
                # img2_reshape[img2_reshape_inf] = 1.2
            # img2_reshape_new = np.delete(img2_reshape, np.where(np.isnan(img2_reshape))[0], axis=0)
            if key_scale is True:
                scaler = StandardScaler()
                scaler.fit(img2_reshape)
                img2_reshape = scaler.transform(img2_reshape)
            clmap = clf.predict(img2_reshape)
            res = clmap.reshape((m, n))
            (filename, extension) = os.path.splitext(files)
            outputway = out_dir + r'\{a}_GBDT_default_3.jpg'.format(a=filename)
            cv2.imwrite(outputway, res)
            time_end = time.time()
            print('time cost: ', time_end-time_start)


if __name__ == '__main__':
    imgWay = r'E:\HE+CAM5\Cutresult'  # 需要分类的图像路径
    hdrWay = r'E:\HE+CAM5\Cutresult\HE_CAM52_mono_E_roi1_prePro.hdr'  # 用于分类的图像hdr路径
    roiWay1 = r'E:\HE+CAM5\Cutresult\roia.jpg'  # 用于分类的图像掩膜路径
    roiWay2 = r'E:\HE+CAM5\Cutresult\roiz.jpg'
    outWay = r'E:\HE+CAM5\Cutresult\gbdtresulta'  # 分类后的输出路径
    classify(imgWay, hdrWay, roiWay1, roiWay2, outWay)
    '''
    from bayes_opt import BayesianOptimization

    x, y = make_classification(n_samples=2500, n_features=10, n_classes=2)
    gbdt = GradientBoostingClassifier()
    cross_val_score(gbdt, x, y, cv=20, scoring='roc_auc').mean()


    def gbdt_cv(n_estimators, min_samples_split, max_features, max_depth):
        res = cross_val_score(
            GradientBoostingClassifier(n_estimators=int(n_estimators),
                                       min_samples_split=int(min_samples_split),
                                       max_features=min(max_features, 0.999),  # float
                                       max_depth=int(max_depth),
                                       random_state=2
                                       ),
            x, y, scoring='roc_auc', cv=20
        ).mean()
        return res

    gbdt_op = BayesianOptimization(
        gbdt_cv,
        {'n_estimators': (10, 250),
         'min_samples_split': (2, 25),
         'max_features': (0.1, 0.999),
         'max_depth': (5, 15)}
    )
    gbdt_op.maximize()
    print(gbdt_op.max)
    n_estimators = int(gbdt_op.max['params']['n_estimators'])
    min_samples_split = int(gbdt_op.max['params']['min_samples_split'])
    max_features = min(gbdt_op.max['params']['max_features'], 0.999)
    max_depth = int(gbdt_op.max['params']['max_depth'])
    clf1 = GradientBoostingClassifier(n_estimators=n_estimators,
                                      min_samples_split=min_samples_split,
                                      max_features=max_features,  # float
                                      max_depth=max_depth,
                                      )
    clf2 = GradientBoostingClassifier()
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    # clf = GradientBoostingClassifier(random_state=0)
    clf1.fit(X_train, y_train)
    result1 = clf1.predict(X_test)
    print('Bayes: ', clf1.score(X_test, y_test))
    clf2.fit(X_train, y_train)
    result2 = clf2.predict(X_test)
    print('default: ', clf2.score(X_test, y_test))
    # 创建新的figure
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.scatter(x[:, 0], x[:, 1], c=y)
    ax = fig.add_subplot(222)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    ax = fig.add_subplot(223)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=result1)
    ax = fig.add_subplot(224)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=result2)
    plt.show()
    '''



