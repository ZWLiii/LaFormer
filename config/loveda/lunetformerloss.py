from torch.utils.data import DataLoader
from cropseg.losses.edge_aware_loss import EdgeAwareBinaryLoss
from cropseg.datasets.loveda_dataset import *
from cropseg.models.LaEageUNetFormer import LaUNetFormer
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 100   #训练的最大轮数，2的倍数
ignore_index = len(CLASSES)
train_batch_size = 16  #训练的批量大小，每次迭代处理8张
val_batch_size = 16    #验证的批量大小，每次迭代处理8张
lr = 2e-3     #
weight_decay = 0.01   #权重衰减（L2正则化）系数，用于减少过拟合。
#骨干网络的学习率和权重衰减
backbone_lr = 6e-5
backbone_weight_decay = 0.01

num_classes = 1
classes = ('agricultural',)  # 只保留前景类作为评估关注对象

weights_name = "lunetformerloss"
weights_path = "model_weights/loveda/{}".format(weights_name)
test_weights_name = "lunetformerloss"
log_name = 'loveda/{}'.format(weights_name)
monitor = 'val_Agriculture_IoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None # whether continue training with the checkpoint, default None

#定义了一个UNetFormer模型，类别数为num_classes
net = LaUNetFormer(num_classes=num_classes)

loss = EdgeAwareBinaryLoss(edge_factor=1.5, aux_weight=0.3)

use_aux_loss = True #是否使用辅助损失

# define the dataloader
#数据增强
#定义了训练数据的增强变换，包括水平翻转和归一化。
def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)

#对输入图像和掩码进行数据增强，包括随机缩放和智能裁剪
def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


train_dataset = LoveDATrainDataset(transform=train_aug, data_root='data/LoveDA/train_val')

val_dataset = loveda_val_dataset

test_dataset = LoveDATestDataset()


#分别为训练和验证数据集创建数据加载器，设置了批量大小、工作线程数、是否使用GPU加速、是否打乱数据等参数
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,  # os.cpu_count()根据CPU核心数量设置4
                          pin_memory=True,#是否用GPU加速
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

#优化器和学习率调度器
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}  #为模型的不同部分设置不同的学习率和权重衰减
net_params = process_model_params(net, layerwise_params=layerwise_params)   #处理模型参数以应用分层学习率
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

