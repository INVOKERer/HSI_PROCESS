from spectral import *
import os
import numpy as np
import time
from PIL import Image
import cv2


def imggray2hyper(file_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    image = None
    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith(('jpg', 'png', 'jpeg', 'bmp')):  # 判断是否以.hdr结尾
            file = file_dir + '\\' + files
            imgOri = cv2.imread(file)
            gray = cv2.cvtColor(imgOri, cv2.COLOR_BGR2GRAY)
            # (filename, extension) = os.path.splitext(files)
            if image is None:
                image = gray
            else:
                image = np.dstack((image, gray))
            # cv2.imwrite(outputway, gray)
    filepath, tmpfilename = os.path.split(file_dir)
    outputway = out_dir + '\\' + tmpfilename + '.hdr'
    envi.save_image(outputway, image, interleave='bsq', dtype=np.float32)


if __name__ == '__main__':
    fileway = r'E:\HE+CAM5\2020\cam\test\1-2'
    # im = Image.open(fileway)
    # img = cv2.imread(fileway)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outfile = r'E:\HE+CAM5\2020\cam\test'
    imggray2hyper(fileway, outfile)
