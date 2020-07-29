import cv2
import os


def judge_white(Img):
    import cv2
    import numpy as np
    HSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    Lower = np.array([0, 0, 221])
    Upper = np.array([180, 30, 255])
    mask = cv2.inRange(HSV, Lower, Upper)
    # cont = 0
    mask = mask.reshape(-1)
    mask = mask.tolist()
    k = float(len(mask))
    '''
    def is_zero(n):
        return n == 0
    '''
    newmask = list(filter(lambda number: number != 0, mask))
    '''
    for i in mask:
        if i == 255:
            cont += 1
        else:
            continue
    '''
    cont = float(len(newmask))  # 白色个数
    # p = round(cont / k, 4)
    p = cont / k
    return p


def main(file_dir, pe):
    for files in os.listdir(file_dir):
        # 当前文件夹所有文件
        if files.endswith('.jpg'):
            file = file_dir + '\\' + files
            Img = cv2.imread(file)
            # print(file)
            p = judge_white(Img)
            print(p)
            if p > pe:
                os.remove(file)


if __name__ == '__main__':
    filename = r'D:\datasets\ndpi\test\result1'
    pe = 0.66
    main(filename, pe)
