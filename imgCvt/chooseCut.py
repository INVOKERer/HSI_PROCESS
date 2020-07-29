import cv2
import os
from PIL import Image


# 切割图片
def splitimage(src, rownum, colnum, walk, x, y, dstpath):
    img = cv2.imread(src)
    w, h, l = img.shape
    # x, y = 0, 100  # 起始坐标
    if rownum <= h and colnum <= w:
        print('Original image info: %sx%sx%s' % (w, h, l))
        print('图片切割')

        s = os.path.split(src)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename = fn[0]
        ext = fn[-1]

        num = 0
        rowheight = walk
        colwidth = walk
        for r in range(rownum):
            for c in range(colnum):
                # box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                box = img[r * colwidth + x:(r + 1) * rowheight + x, c * colwidth + y:(c + 1) * rowheight + y]
                way = os.path.join(dstpath, basename + '_' + str(num) + '.' + ext)
                cv2.imwrite(way, box)
                num = num + 1

        print('共生成 %s 张小图片。' % num)
    else:
        print('error')


# 创建文件夹
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False


if __name__ == '__main__':
    folder = r'E:\HE+CAM5\tiff\2019-12-11_HE+IHC\resultHE'  # 存放图片的文件夹
    mkpath = r'E:\HE+CAM5\tiff\2019-12-11_HE+IHC\resultHE\trainA'
    path = os.listdir(folder)
    mkdir(mkpath)
    row = int(2)  # 切割行数
    col = int(2)  # 切割列数
    k = int(512)  # 步长
    a, b = 0, 0   # 起始坐标点
    for each_bmp in path:  # 批量操作
        first_name, second_name = os.path.splitext(each_bmp)
        each_bmp = os.path.join(folder, each_bmp)
        src = each_bmp
        print(src)
        print(first_name)
        if os.path.isfile(src):
            dstpath = mkpath
            if (dstpath == '') or os.path.exists(dstpath):
                # row = int(2)  # 切割行数
                # col = int(3)  # 切割列数
                # k = int(512)  # 步长
                if row > 0 and col > 0:
                    splitimage(src, row, col, k, a, b, dstpath)
                else:
                    print('无效的')
            else:
                print('图片保存目录 %s 不存在！' % dstpath)
        else:
            print('图片文件 %s 不存在！' % src)
