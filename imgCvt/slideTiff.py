import openslide as op
from openslide.deepzoom import DeepZoomGenerator
import os


def tif_cut(tifway, x, k, outway):            # x = 分割后图片边长  k = tif图片的第几个level
    if not os.path.isdir(outway):
        os.makedirs(outway)
    slide = op.OpenSlide(tifway)
    print(slide.level_dimensions)
    bigImg = DeepZoomGenerator(slide, tile_size=x-2, overlap=1, limit_bounds=False)
    count = bigImg.level_count - 1
    z = count - k                  # 倒着读取
    raw, col = bigImg.level_tiles[z]
    print(raw, col)
    if not os.path.isdir(outway):
        os.makedirs(outway)
    for i in range(raw-1):
        for j in range(col-1):
            try:
                img = bigImg.get_tile(z, (j, i))
                outputway = outway + r'\rgb_{a}_{b}.png'.format(a=i, b=j)
                img.save(outputway)
            except ValueError:
                break
    print('done')


if __name__ == '__main__':
    tif_way = r'E:\HE+CAM5\tiff\HE'
    out_dir = r'E:\HE+CAM5\tiff\HE\result2'
    a, n = 1024, 0       # a = 分割后图片边长  n = tif图片的第几个level（从0开始）
    tif_cut(tif_way, a, n, out_dir)
