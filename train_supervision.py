import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from tools.cfg import py2cfg
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"    #设置 Hugging Face 的镜像服务器地址的环境变量
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
import sys
sys.path.append('config/loveda')  #解决找不到unetformer模块问题


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

#允许用户在运行脚本时指定一个配置文件的路径。
def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()

class BinaryEvaluatorWrapper:
    def __init__(self):
        self.evaluator = Evaluator(num_class=2)  # 明确是二分类（0:背景, 1:前景）

    def add_batch(self, gt_image, pred_image):
        # 确保输入是 0/1，且类型正确
        gt_image = (gt_image > 0).astype(np.uint8)
        pred_image = (pred_image > 0).astype(np.uint8)
        self.evaluator.add_batch(gt_image, pred_image)

    def reset(self):
        self.evaluator.reset()

    def evaluate(self):
        return {
            "IoU": self.evaluator.Intersection_over_Union()[1],
            "F1": self.evaluator.F1()[1],
            "Precision": self.evaluator.Precision()[1],
            "Recall": self.evaluator.Recall()[1]
        }



class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.use_aux = config.use_aux_loss
        self.num_classes = config.num_classes  # 👈 关键：用于判断结构

        self.metrics_train = Evaluator(num_class=2)
        self.metrics_val = Evaluator(num_class=2)

    def forward(self, x):
        return self.net(x)

    def post_process(self, logits):
        """自动根据输出通道判断预测方式"""
        if logits.shape[1] == 1:
            prob = torch.sigmoid(logits)                   # [B, 1, H, W]
            pred = (prob > 0.5).long().squeeze(1)          # [B, H, W]
        elif logits.shape[1] >= 2:
            prob = torch.softmax(logits, dim=1)            # [B, C, H, W]
            pred = prob.argmax(dim=1).long()               # [B, H, W]
        else:
            raise ValueError(f"Unsupported output shape: {logits.shape}")
        return pred

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.net(img)
        loss = self.loss(prediction, mask)

        # 取主输出
        if isinstance(prediction, (tuple, list)) and self.use_aux:
            logits = prediction[0]
        else:
            logits = prediction

        pred_mask = self.post_process(logits)  # 👈 自动处理预测方式

        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pred_mask[i].cpu().numpy())

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        iou = self.metrics_train.Intersection_over_Union()
        f1 = self.metrics_train.F1()[1]
        pre = self.metrics_train.Precision()[1]
        recall = self.metrics_train.Recall()[1]
        oa = self.metrics_train.OA()
        agri_iou = iou[1]
        miou = np.mean(iou)

        self.metrics_train.reset()
        self.log_dict({
            'train_mIoU': miou,
            'train_Agriculture_IoU': agri_iou,
            'train_F1': f1,
            'train_Pre': pre,
            'train_Recall': recall,
            'train_OA': oa
        }, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        loss_val = self.loss(prediction, mask)

        if isinstance(prediction, (tuple, list)) and self.use_aux:
            logits = prediction[0]
        else:
            logits = prediction

        pred_mask = self.post_process(logits)  # 👈 自动处理预测方式

        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pred_mask[i].cpu().numpy())

        self.log("val_loss", loss_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        iou = self.metrics_val.Intersection_over_Union()
        f1 = self.metrics_val.F1()[1]
        pre = self.metrics_val.Precision()[1]
        recall = self.metrics_val.Recall()[1]
        oa = self.metrics_val.OA()
        agri_iou = iou[1]
        miou = np.mean(iou)

        self.metrics_val.reset()
        self.log_dict({
            'val_mIoU': miou,
            'val_Agriculture_IoU': agri_iou,
            'val_F1': f1,
            'val_Pre': pre,
            'val_Recall': recall,
            'val_OA': oa
        }, prog_bar=True)

    def configure_optimizers(self):
        return [self.config.optimizer], [self.config.lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader


# training
def main():
    args = get_args()       #获取命令行参数
    config = py2cfg(args.config_path)   #这一步会执行 odunetformer.py
    seed_everything(42)

    # 定义早停回调（关键添加部分）
    early_stop = EarlyStopping(
        monitor=config.monitor,  # 监控指标（如 val_mIoU）
        patience=6,  # 连续6个epoch无提升则停止
        mode=config.monitor_mode,  # 根据监控指标方向设置（max/min）
        verbose=True  # 打印提示信息
    )

    # 配置模型保存的回调函数
    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    # 配置日志记录器
    logger = CSVLogger('lightning_logs', name=config.log_name)

    # 创建模型并指定模型路径、检查点路径、权重名称等参数
    model = Supervision_Train(config)

    # 如果指定了预训练的检查点路径，则加载预训练的模型
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    # 配置 PyTorch Lightning 的 Trainer
    trainer = pl.Trainer(devices=config.gpus,
                         max_epochs=config.max_epoch,
                         accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         #val_check_interval=1.0,  # 每个 epoch 结束后验证一次
                         callbacks=[checkpoint_callback, early_stop],
                         strategy='auto',
                         logger=logger)    #设置训练器

    # 启动训练
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
   main()
