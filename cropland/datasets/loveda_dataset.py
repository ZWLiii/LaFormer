from .transform import *
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import matplotlib.patches as mpatches
from PIL import Image, ImageOps
import random

# 用于处理和可视化LoveDA数据集中的遥感图像和语义分割掩码


# CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest',
#            'agricultural')
#
# PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
#            [159, 129, 183], [0, 255, 0], [255, 195, 128]]

# 配置适用于 num_classes=2、使用 CrossEntropyLoss 的多类语义分割
CLASSES = ('background', 'agricultural')

PALETTE = [
    [255, 255, 255],  # 背景白色
    [255, 195, 128]  # 耕地橙色
]

# 二分类任务（num_classes=1）+ Sigmoid + BCE loss
# CLASSES = ('agricultural',)  # 只有一个正类
# PALETTE = [
#     [255, 195, 128],      # 耕地橙色
# ]


ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)


# 定义了训练数据的增强变换，包括水平翻转、垂直翻转、随机亮度对比度调整和归一化
def get_training_transform():
    train_transform = [
        # albu.Resize(height=1024, width=1024),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        # albu.RandomRotate90(p=0.5),
        # albu.OneOf([
        #     albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
        #     albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25)
        # ], p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


# 对输入图像和掩码进行多尺度训练和裁剪，然后应用数据增强
def train_aug(img, mask):
    # multi-scale training and crop
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False)])
    img, mask = crop_aug(img, mask)

    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


# 定义了验证数据的增强变换，主要是归一化
def get_val_transform():
    val_transform = [
        # albu.Resize(height=1024, width=1024, interpolation=cv2.INTER_CUBIC),
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


# 用于加载和处理训练数据集。支持普通加载和马赛克数据增强加载
class LoveDATrainDataset(Dataset):
    def __init__(self, data_root='data/LoveDA/Train', img_dir='images_png', mosaic_ratio=0.25,
                 mask_dir='masks_png_convert', img_suffix='.png', mask_suffix='.png',
                 transform=train_aug, img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mosaic_ratio = mosaic_ratio

        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    # 根据索引加载图像和掩码，并应用数据增强
    def __getitem__(self, index):
        p_ratio = random.random()
        img, mask = self.load_img_and_mask(index)
        if p_ratio < self.mosaic_ratio:
            img, mask = self.load_mosaic_img_and_mask(index)
        if self.transform:
            img, mask = self.transform(img, mask)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id, img_type = self.img_ids[index]
        results = {'img': img, 'gt_semantic_seg': mask, 'img_id': img_id, 'img_type': img_type}

        return results

    def __len__(self):
        length = len(self.img_ids)
        return length

    #     #获取数据集中所有图像的ID,解压压缩包可能会传进空文件，加入了筛选文件代码
    def get_img_ids(self, data_root, img_dir, mask_dir):
        def collect_ids(split_name):
            img_dir_path = osp.join(data_root, split_name, img_dir)
            mask_dir_path = osp.join(data_root, split_name, mask_dir)

            img_filename_list = [
                f for f in os.listdir(img_dir_path)
                if f.endswith('.png') and len(f.split('.')[0]) > 0 and not f.startswith('.')]
            mask_filename_list = [
                f for f in os.listdir(mask_dir_path)
                if f.endswith('.png') and len(f.split('.')[0]) > 0 and not f.startswith('.')]

            assert len(img_filename_list) == len(mask_filename_list), \
                f"{split_name} 图像和掩码数量不一致！图像 {len(img_filename_list)} vs 掩码{len(mask_filename_list)}"

            img_ids = [(f.split('.')[0], split_name) for f in img_filename_list]
            return img_ids  # 这个 return 应该在函数内部

        urban_img_ids = collect_ids('Urban')
        rural_img_ids = collect_ids('Rural')
        img_ids = urban_img_ids + rural_img_ids

        return img_ids  # ✅ 这个 return 必须缩进在 get_img_ids 函数体中

    # 加载单个图像和对应的掩码。
    def load_img_and_mask(self, index):
        img_id, img_type = self.img_ids[index]
        img_name = osp.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, img_type, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        return img, mask

    # 加载马赛克增强的图像和掩码。
    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask


loveda_val_dataset = LoveDATrainDataset(data_root='data/LoveDA/Val', mosaic_ratio=0.0,
                                        transform=val_aug)


# 用于加载和处理测试数据集
class LoveDATestDataset(Dataset):
    def __init__(self, data_root='data/LoveDA/Test', img_dir='images_png',
                 img_suffix='.png', mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir

        self.img_suffix = img_suffix
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir)

    # 根据索引加载图像，并应用归一化。
    def __getitem__(self, index):
        img = self.load_img(index)

        img = np.array(img)
        aug = albu.Normalize()(image=img)
        img = aug['image']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img_id, img_type = self.img_ids[index]

        results = {'img': img, 'img_id': img_id, 'img_type': img_type}

        return results

    def __len__(self):
        length = len(self.img_ids)

        return length

    # 获取测试集中所有图像的ID。
    def get_img_ids(self, data_root, img_dir):
        def valid_ids(file_list, type_str):
            return [
                (str(fname.split('.')[0]), type_str)
                for fname in file_list
                if fname.endswith('.png') and fname.split('.')[0].strip()  # 非空名过滤
            ]

        urban_path = osp.join(data_root, 'Urban', img_dir)
        rural_path = osp.join(data_root, 'Rural', img_dir)

        urban_img_ids = valid_ids(os.listdir(urban_path), 'Urban')
        rural_img_ids = valid_ids(os.listdir(rural_path), 'Rural')

        return urban_img_ids + rural_img_ids
        # urban_img_filename_list = os.listdir(osp.join(data_root, 'Urban', img_dir))
        # urban_img_ids = [(str(id.split('.')[0]), 'Urban') for id in urban_img_filename_list]
        # rural_img_filename_list = os.listdir(osp.join(data_root, 'Rural', img_dir))
        # rural_img_ids = [(str(id.split('.')[0]), 'Rural') for id in rural_img_filename_list]
        # img_ids = urban_img_ids + rural_img_ids
        #
        # return img_ids

    # 加载单一图像。
    def load_img(self, index):
        img_id, img_type = self.img_ids[index]
        img_name = osp.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)
        img = Image.open(img_name).convert('RGB')

        return img


# 显示遥感图像、真实掩码和预测掩码
def show_img_mask_seg(seg_path, img_path, mask_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    seg_list = seg_list[start_seg_index:start_seg_index + 2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i]) / 255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        mask = cv2.imread(f'{mask_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).convert('P')
        mask.putpalette(np.array(PALETTE, dtype=np.uint8))
        mask = np.array(mask.convert('RGB'))
        img_id = str(seg_id.split('.')[0]) + '.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE ' + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title('Mask True ' + seg_id)
        ax[i, 2].set_axis_off()
        ax[i, 2].imshow(img_seg)
        ax[i, 2].set_title('Mask Predict ' + seg_id)
        ax[i, 2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


# 显示遥感图像和分割结果
def show_seg(seg_path, img_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    seg_list = seg_list[start_seg_index:start_seg_index + 2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i]) / 255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        img_id = str(seg_id.split('.')[0]) + '.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE ' + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(img_seg)
        ax[i, 1].set_title('Seg IMAGE ' + seg_id)
        ax[i, 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


# 显示遥感图像和对应的掩码
def show_mask(img, mask, img_id):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    patches = [mpatches.Patch(color=np.array(PALETTE[i]) / 255., label=CLASSES[i]) for i in range(len(CLASSES))]
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert('RGB'))
    ax1.imshow(img)
    ax1.set_title('RS IMAGE ' + str(img_id) + '.png')
    ax2.imshow(mask)
    ax2.set_title('Mask ' + str(img_id) + '.png')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')
