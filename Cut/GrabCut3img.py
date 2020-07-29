import cv2
import numpy as np
import os
import math


'''
mask_canny = cv2.Canny(mask2, 127, 255)
I, contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i, cont in enumerate(contours):
    ares = cv2.contourArea(cont)   # 计算包围性状的面积
    length1 = cv2.arcLength(cont, True)
    length2 = cv2.arcLength(cont, False)
    M = cv2.moments(cont)
    print(M['m00'])
    if M['m00'] < 5000 or ares < 5000 or length1 < 40 or length2 < 40:                  # 过滤面积小于10的形状
        del contours[i]
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
img = cv2.resize(img, (900, 650))
cv2.imshow("img", img)
cv2.waitKey(0)
'''


def imgGrabCut(file_dir, mask_dir, HE_dir, out_dir):

    splits = os.listdir(file_dir)
    for sp in splits:
        img_fold_A = os.path.join(file_dir)
        img_fold_B = os.path.join(mask_dir)
        img_fold_C = os.path.join(HE_dir)
        img_list_A = os.listdir(img_fold_A)
        img_list_B = os.listdir(img_fold_B)
        img_list_C = os.listdir(img_fold_C)
        img_list_A.sort()
        img_list_B.sort()
        img_list_C.sort()
        num_imgs = min(len(img_list_A), len(img_list_B), len(img_list_C))
        img_fold_result = os.path.join(out_dir)
        if not os.path.isdir(img_fold_result):
            os.makedirs(img_fold_result)
        print('split = %s, number of images = %d' % (sp, num_imgs))
        for n in range(num_imgs):
            name_A = img_list_A[n]
            path_A = os.path.join(img_fold_A, name_A)
            name_B = img_list_B[n]
            path_B = os.path.join(img_fold_B, name_B)
            name_C = img_list_C[n]
            path_C = os.path.join(img_fold_C, name_C)
            print(name_A, name_B)
            if os.path.isfile(path_A) and os.path.isfile(path_B):
                name_B = name_B
                path_B_re = os.path.join(img_fold_result, name_B)
                name_A = name_A
                path_A_re = os.path.join(img_fold_result, name_A)
                img = cv2.imread(path_A, 1)   # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                HE_img = cv2.imread(path_C, 1)
                mask_3ch = cv2.imread(path_B, 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                mask = np.zeros((img.shape[:2]), np.uint8)
                ret, mask_3ch_thr = cv2.threshold(mask_3ch, 127, 255, cv2.THRESH_BINARY)
                mask_1 = mask_3ch_thr[:, :, 0]
                mask_3ch_erode = cv2.erode(mask_3ch_thr, None, iterations=3)
                mask_3ch_dilate = cv2.dilate(mask_3ch_erode, None, iterations=3)
                mask_1ch_dilate = mask_3ch_dilate[:, :, 0]

                mask[mask_1 == 0] = 2
                mask[mask_1 == 255] = 3
                # 这里计算了5次
                mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel=None, fgdModel=None, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)
                # 关于where函数第一个参数是条件，满足条件的话赋值为0，否则是1。如果只有第一个参数的话返回满足条件元素的坐标。
                mask2 = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
                # mask2就是这样固定的
                cv2.imwrite(path_B_re, mask2)

                I, contours, hierarchy = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for i, cont in enumerate(contours):
                    ares = cv2.contourArea(cont)   # 计算包围性状的面积
                    length1 = cv2.arcLength(cont, True)
                    length2 = cv2.arcLength(cont, False)
                    M = cv2.moments(cont)
                    print('M: ', M['m00'])

                    if M['m00'] < 500 or ares < 500 or length1 < 50 or length2 < 50:           # 过滤
                        del contours[i]
                    '''else:
                        ellipse = cv2.fitEllipse(cont)
                        a, b = ellipse[1]
                        area = a * b
                        print('ellipse: ', area, ' a: ', a, ' b: ', b)
                        if area < 1800:
                            del contours[i]'''

                cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
                HE_img = cv2.drawContours(HE_img, contours, -1, (0, 0, 255), 3)
                cv2.imwrite(path_A_re, HE_img)


if __name__ == '__main__':
    imgWay = r'E:\HE+CAM5\2020\cam\preprocess\GBDT\CAM'      # 原图所在文件夹
    maskway = r'E:\HE+CAM5\2020\cam\preprocess\GBDT\gbdt5'
    outWay = r'E:\HE+CAM5\2020\cam\preprocess\GBDT\GrabCutresult5'
    HE_dir = r'E:\HE+CAM5\2020\cam\preprocess\GBDT\HE'
    imgGrabCut(imgWay, maskway, HE_dir, outWay)
