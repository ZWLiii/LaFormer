import glob
import os
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import random

SEED = 42

# CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest',
#            'agricultural')
#
# PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
#            [159, 129, 183], [0, 255, 0], [255, 195, 128]]

#配置适用于 num_classes=2、使用 CrossEntropyLoss 的多类语义分割
CLASSES = ('background', 'agricultural')

PALETTE = [
    [255, 255, 255],  # 背景白色
    [255, 195, 128]      # 耕地橙色
]


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


#用于解析命令行参数，允许用户在运行脚本时自定义输入和输出目录
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", default="data/LoveDA/Train/Rural/masks_png")
    parser.add_argument("--output-mask-dir", default="data/LoveDA/Train/Rural/masks_png_convert")
    return parser.parse_args()

#原来用于转换标签掩码
# def convert_label(mask):
#     mask[mask == 0] = 8  #将掩码中所有值为 0 的像素改为 8
#     mask -= 1
#
#     return mask

#把原掩码中 类别 6（agricultural）设为 1，其它设为 0，生成新的二分类标签图。
def convert_label(mask):
    # 生成二分类掩码：农业类设为 1，其他为 0
    bin_mask = np.zeros_like(mask, dtype=np.uint8)
    bin_mask[mask == 7] = 1
    return bin_mask




# 将标签掩码转换为彩色RGB图像
def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)  #创建一个与输入掩码相同大小的空RGB图像，初始化为全黑

    # mask_convert = mask[np.newaxis, :, :]
    # mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]   #背景白色
    # mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    # mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    # mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]
    # mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]
    # mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]
    # mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]     #农田

    mask_rgb[mask == 1] = [255, 195, 128]      # 耕地橙色
    mask_rgb[mask == 0] = [255, 255, 255]  # 非农业类：白色
    return mask_rgb



# 多线程并行处理每个掩码  读取原始掩码图像，转换标签（使用 convert_label 函数）生成彩色可视化图像（使用 label2rgb 函数）。保存转换后的标签掩码和彩色可视化图像。
def patch_format(inp):
    (mask_path, masks_output_dir) = inp
    # print(mask_path, masks_output_dir)
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    label = convert_label(mask)
    rgb_label = label2rgb(label.copy())
    rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_RGB2BGR)
    out_mask_path_rgb = os.path.join(masks_output_dir + '_rgb', "{}.png".format(mask_filename))
    cv2.imwrite(out_mask_path_rgb, rgb_label)

    out_mask_path = os.path.join(masks_output_dir, "{}.png".format(mask_filename))
    cv2.imwrite(out_mask_path, label)

    print(np.unique(mask))


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    masks_dir = args.mask_dir
    masks_output_dir = args.output_mask_dir
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))

    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        os.makedirs(masks_output_dir + '_rgb')

    inp = [(mask_path, masks_output_dir) for mask_path in mask_paths]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


