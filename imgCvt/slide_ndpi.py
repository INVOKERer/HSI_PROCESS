import openslide as op
from openslide.deepzoom import DeepZoomGenerator
import os
import numpy as np
import judge_white
# import time


def ndpi_cut(ndpi_way, x, k, outway, pe):            # x = 分割后图片边长  k = ndpi图片的第几个level
    # time_start = time.time()
    if not os.path.isdir(outway):
        os.makedirs(outway)
    for files in os.listdir(ndpi_way):
        # 当前文件夹所有文件
        if files.endswith('.tiff'):  # 判断是否以.ndpi结尾 也可以为tiff
            slide = op.OpenSlide(ndpi_way + '\\' + files)
            # print(slide.level_dimensions)
            bigImg = DeepZoomGenerator(slide, tile_size=x-2, overlap=1, limit_bounds=False)
            count = bigImg.level_count - 1
            z = count - k                  # 倒着读取
            raw, col = bigImg.level_tiles[z]
            # print(raw, col)
            (filename, extension) = os.path.splitext(files)
            for i in range(raw-1):
                for j in range(col-1):
                    try:
                        img = bigImg.get_tile(z, (j, i))
                        Img = np.asarray(img)
                        if Img.shape == (a, a, Img.shape[2]):
                            p = judge_white.judge_white(Img)
                            print(p, Img.shape, (a, a, Img.shape[2]))
                            if p < pe:
                                outputway = outway + '\\' + filename + r'_{a}_{b}.png'.format(a=i, b=j)
                                img.save(outputway)
                            else:
                                pass
                    except ValueError:
                        break
            print(files + ' done')
            # time_end = time.time()
            # print('totally cost', time_end-time_start)


if __name__ == '__main__':
    ndpi_way = r'E:\HE+CAM5\tiff\tiff\CAM+HE'
    out_dir = r'E:\HE+CAM5\tiff\tiff\CAM+HE_result'
    a, n = 512, 0       # a = 分割后图片边长  n = tif图片的第几个level（从0开始）
    pe = 0.77
    ndpi_cut(ndpi_way, a, n, out_dir, pe)
