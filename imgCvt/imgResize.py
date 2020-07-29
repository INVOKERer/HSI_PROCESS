import cv2
import os


def imgResize(file_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith('.jpg'):  # 判断是否以.jpg结尾
            file = file_dir + '\\' + files
            imgOri = cv2.imread(file)
            size = (1024, 1024)
            result = cv2.resize(imgOri, size, interpolation=cv2.INTER_AREA)
            (filename, extension) = os.path.splitext(files)
            outputway = out_dir + '//' + filename + '.jpg'
            cv2.imwrite(outputway, result)


if __name__ == '__main__':
    folder = r'E:\HE+CAM5\PreproEasy\rgb-down1'       # 存放图片的文件夹
    outWay = 'E:/HE+CAM5/PreproEasy/rgb-down1/testA'  # 输出路径
    imgResize(folder, outWay)

