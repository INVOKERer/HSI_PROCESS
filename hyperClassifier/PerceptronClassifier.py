from spectral import *
import cv2
import numpy as np
import os


def classify(file_dir, hdr_dir, roi_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    img = open_image(hdr_dir).load()
    gt1 = cv2.imread(roi_dir)
    gt = np.asarray(gt1[:, :, 0])
    (m, n) = gt.shape
    for i in range(m):
        for j in range(n):
            if gt[i, j] < 127:
                gt[i, j] = 10
            else:
                gt[i, j] = 255
    pc = principal_components(img)
    pc_0999 = pc.reduce(fraction=0.999)
    img_pc = pc_0999.transform(img)
    classes = create_training_classes(img_pc, gt)
    # fld = linear_discriminant(classes)
    # xdata = fld.transform(img)
    # xdata = np.asarray(xdata)
    # classes = create_training_classes(xdata, gt)
    nfeatures = img_pc.shape[-1]
    nclasses = len(classes)
    p = PerceptronClassifier([nfeatures, 20, 8, nclasses])
    p.train(classes, 20, clip=0., accuracy=100., batch=1, momentum=0.3, rate=0.3)
    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith('.hdr'):  # 判断是否以.hdr结尾
            print('doing img:' + files)
            file = file_dir + '\\' + files
            img2 = open_image(file).load()
            img_pc2 = pc_0999.transform(img2)
            clmap = p.classify_image(img_pc2)
            (filename, extension) = os.path.splitext(files)
            outputway = out_dir + r'\{a}.jpg'.format(a=filename)
            save_rgb(outputway, clmap)


if __name__ == '__main__':
    imgWay = r'E:\HE+CAM5\PreproEasy'                                  # 需要分类的图像路径
    hdrWay = r'E:\HE+CAM5\PreproEasy\HE_CAM52_mono_E_roi1_prePro.hdr'  # 用于分类的图像hdr路径
    roiWay = r'E:\HE+CAM5\PreproEasy\roi1_SVM.jpg'                     # 用于分类的图像掩膜路径
    outWay = r'E:\HE+CAM5\PreproEasy\rgb-down\testd'                   # 分类后的输出路径
    classify(imgWay, hdrWay, roiWay, outWay)
