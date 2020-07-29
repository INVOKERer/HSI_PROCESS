from spectral import *
import spectral.io.envi as envi
import os


def imgLoad(file_dir,out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith('.hdr'):  # 判断是否以.hdr结尾
            file = file_dir + '\\' + files
            imgOri = envi.open(file)
            R, G, B = 28, 18, 9  # 从1开始
            rgb = get_rgb(imgOri, bands=(R-1, G-1, B-1), stretch=(0.02, 0.98))  # ((0.02, 0.98), (0.02, 0.98), (0.02, 0.98)))
            (filename, extension) = os.path.splitext(files)
            outputway = out_dir + r'\{a}.jpg'.format(a=filename)
            save_rgb(outputway, rgb)


if __name__ == '__main__':
    imgWay = r'E:\HE+CAM5\2020\cam\preprocess'      # 原图所在文件夹
    outWay = r'E:\HE+CAM5\2020\cam\preprocess\rgb'
    imgLoad(imgWay, outWay)


