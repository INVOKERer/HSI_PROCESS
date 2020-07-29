from spectral import *
import os
import cv2


def imgLoad(file_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith(('jpg', 'png', 'jpeg', 'bmp')):  # 判断是否以.hdr结尾
            file = file_dir + '\\' + files
            imgOri = cv2.imread(file)
            gray = cv2.cvtColor(imgOri, cv2.COLOR_BGR2GRAY)
            (filename, extension) = os.path.splitext(files)
            outputway = out_dir + r'\{a}'.format(a=files)
            cv2.imwrite(outputway, gray)


if __name__ == '__main__':
    imgWay = r'E:\HE+CAM5\2020\HE2\deblurgray\trainA'      # 原图所在文件夹
    outWay = r'E:\HE+CAM5\2020\HE2\deblurgray\trainB'
    imgLoad(imgWay, outWay)


