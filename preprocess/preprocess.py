from spectral import *
import os
import numpy as np
import time


def math(s1, s2):
    # time_start = time.time()
    # sx = s1/s2
    # time_end = time.time()
    # print('time cost: ', time_end-time_start)
    time_start = time.time()
    s = np.divide(s1, s2)
    time_end = time.time()
    print('time cost: ', time_end-time_start)
    return s


def imgprocess(file_dir, out_dir, blankend):       # hdr中data type = 5 interleave = bip
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith(blankend):   # 判断是否以*结尾
            file = file_dir + '\\' + files
            blank = open_image(file)
            blank = blank[:, :, :]
            imgblank = np.asarray(blank, dtype=float)

    count = 1
    for files in os.listdir(file_dir):
        if files.endswith('.hdr'):  # 判断是否以.hdr结尾
            file = file_dir + '\\' + files
            ori = open_image(file)
            img = ori[:, :, :]
            img = np.asarray(img, dtype=float)
            (filename, extension) = os.path.splitext(files)
            pcdata = math(img, imgblank)
            rawfile = out_dir + '\\' + filename + '.raw'
            pcdata.tofile(rawfile)
            print('pre image ' + str(count))
            count = count + 1


def generateHDR(file_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    rename_file(file_dir, '.hdr', '.txt')
    for files in os.listdir(file_dir):
        if files.endswith('.txt'):
            file = file_dir + '\\' + files
            f = open(file, "r+", encoding='UTF-8')   # 设置文件对象
            new_stus = ''
            for i in f:
                if i == 'data type = 12\n':
                    i = 'data type = 5\n'
                    # new_stus.append(i)
                    new_stus = new_stus + i
                elif i == 'interleave = bsq\n':
                    i = 'interleave = bip\n'
                    # new_stus.append(i)
                    new_stus = new_stus + i
                elif i == 'wavelength units = Unknown\n':
                    i = 'wavelength units = Nanometers\n'
                    # new_stus.append(i)
                    new_stus = new_stus + i
                else:
                    # new_stus.append(i)
                    new_stus = new_stus + i
            # f.truncate()   # 清空txt
            f.close()
            full_path = out_dir + '\\' + files
            file = open(full_path, 'w')
            file.write(new_stus)
            file.close()
    rename_file(out_dir, ".txt", ".hdr")
    rename_file(file_dir, ".txt", ".hdr")


def rename_file(file_dir, x, y):   # change x to y
    config = 0
    for files in os.listdir(file_dir):
        portion = os.path.splitext(files)
        if files.endswith('config.txt'):
            config = config + 1
        elif portion[1] == x:   # 如果后缀是x
            # 重新组合文件名和后缀名
            file = file_dir + '\\' + files
            newname = file_dir + '\\' + portion[0] + y
            os.rename(file, newname)


if __name__ == '__main__':
    imgWay = r'E:\HE+CAM5\2020\cam'             # 原图所在文件夹
    outdir = r'E:\HE+CAM5\2020\cam\preprocess'      # 输出路径
    blank_file_end = 'endblank-7681-6304.hdr'               # 空白图像的末尾
    imgprocess(imgWay, outdir, blank_file_end)
    generateHDR(imgWay, outdir)

