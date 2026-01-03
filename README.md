## Version 2.0 (stable)

[Welcome to my homepage!](https://github.com/ZWLiii/LaFormer)

## Introduction

**LaFormer** is an open-source  semantic segmentation toolbox based on PyTorch, [pytorch lightning](https://www.pytorchlightning.ai/) and [timm](https://github.com/rwightman/pytorch-image-models), 
which mainly focuses on developing advanced Vision Transformers for agricultural remote sensing image segmentation.


## Major Features

- Unified Benchmark

  we provide a unified training script for various segmentation methods.
  
- Simple and Effective

  Thanks to **pytorch lightning** and **timm** , the code is easy for further development.
  
- Supported Remote Sensing Datasets

  - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)
  - [FGFD](https://github.com/Henryjiepanli/DBBANet)
  - More datasets will be supported in the future.
  
- Multi-scale Training and Testing
- Inference on Huge Remote Sensing Images
  
## Folder Structure

Prepare the following folders to organize this repo:
```none
LaFormer (code)
├── pretrain_weights (pretrained weights of backbones, such as vit, swin, etc)
├── model_weights (save the model weights trained on ISPRS vaihingen, LoveDA, etc)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)

```

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r GeoSeg/requirements.txt
```

## Pretrained Weights of Backbones

[Baidu Disk](https://pan.baidu.com/s/1foJkxeUZwVi5SnKNpn6hfg) : 1234 

[Google Drive](https://drive.google.com/drive/folders/1ELpFKONJZbXmwB5WCXG7w42eHtrXzyPn?usp=sharing)

## Data Preprocessing

Download the datasets from the official website and split them yourself.


Generate the training set.

**LoveDA**
```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```
**FGFD**
```
python tools/fgfd_mask_convert.py --mask-dir data/FGFD/train/label --output-mask-dir data/FGFD/train/label_convert
python tools/fgfd_mask_convert.py --mask-dir data/FGFD/val/label --output-mask-dir data/FGFD/val/label_convert
python tools/fgfd_mask_convert.py --mask-dir data/FGFD/train_val/label --output-mask-dir data/FGFD/train_val/label_convert
python tools/fgfd_mask_convert.py --mask-dir data/FGFD/test/label --output-mask-dir data/FGFD/test/label_convert
```
## Training

"-c" means the path of the config, use different **config** to train different models.

```
python train_supervision.py -c config/loveda/lunetformerloss.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format

**LoveDA**
```
python loveda_test.py -c config/loveda/lunetformerloss.py -o fig_results/loveda/lunetformerloss --rgb -t 'd4' --val

```

**FGFD** 
```
python fgfd_test.py -c config/fgfd/lunetformerloss.py -o results/fgfd/lunetformerloss --rgb -t 'd4' --val
```

## License 
-------
Code is released for non-commercial and research purposes only. For commercial purposes, please contact the authors.

## Citation

If you find this project useful in your research, please consider citing：

- [LaFormer: a Laplacian edge-guided transformer for farmland segmentation in remote sensing images](https://www.spiedigitallibrary.org/journals/journal-of-applied-remote-sensing/volume-19/issue-4/044515/LaFormer--a-Laplacian-edge-guided-transformer-for-farmland-segmentation/10.1117/1.JRS.19.044515.short)



## Acknowledgement

We wish **GeoSeg** could serve the growing research of remote sensing by providing a unified benchmark 
and inspiring researchers to develop their own segmentation networks. Many thanks the following projects's contributions to **GeoSeg**.
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
