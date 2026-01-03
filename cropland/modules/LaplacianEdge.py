import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

class LaplacianEdge(nn.Module):
    def __init__(self):
        super().__init__()
        # 高斯平滑卷积
        self.gaussian = nn.Conv2d(3, 3, kernel_size=5, padding=2, groups=3, bias=False)
        self.gaussian.weight.data[:] = self.get_gaussian_kernel()
        self.gaussian.weight.requires_grad = False

        # Laplacian 卷积核
        laplacian_kernel = torch.tensor([[-1, -1, -1],
                                         [-1,  8, -1],
                                         [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.laplacian = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.laplacian.weight.data[:] = laplacian_kernel
        self.laplacian.weight.requires_grad = False

    def get_gaussian_kernel(self, ksize=5, sigma=1.0):
        """生成高斯平滑卷积核"""
        ax = torch.arange(-ksize // 2 + 1., ksize // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / torch.sum(kernel)
        kernel3 = kernel.expand(3, 1, ksize, ksize)
        return kernel3

    def forward(self, x):
        x = self.gaussian(x)
        x_gray = TF.rgb_to_grayscale(x)
        edge = self.laplacian(x_gray)
        return edge

class EdgeFusionEncoder(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        self.edge_detector = LaplacianEdge()

        self.edge_convs = nn.ModuleList([
            nn.Conv2d(1, encoder_channels[0], 1),
            nn.Conv2d(1, encoder_channels[1], 1),
            nn.Conv2d(1, encoder_channels[2], 1),
            nn.Conv2d(1, encoder_channels[3], 1),
        ])

    def forward(self, x, res1, res2, res3, res4):
        edge = self.edge_detector(x)

        edge1 = F.interpolate(edge, size=res1.shape[2:], mode='bilinear', align_corners=False)
        edge2 = F.interpolate(edge, size=res2.shape[2:], mode='bilinear', align_corners=False)
        edge3 = F.interpolate(edge, size=res3.shape[2:], mode='bilinear', align_corners=False)
        edge4 = F.interpolate(edge, size=res4.shape[2:], mode='bilinear', align_corners=False)

        res1 = res1 + self.edge_convs[0](edge1)
        res2 = res2 + self.edge_convs[1](edge2)
        res3 = res3 + self.edge_convs[2](edge3)
        res4 = res4 + self.edge_convs[3](edge4)

        return res1, res2, res3, res4
