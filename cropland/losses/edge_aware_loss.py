import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAwareBinaryLoss(nn.Module):
    def __init__(self, edge_factor=1.0, aux_weight=0.4):
        super(EdgeAwareBinaryLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.edge_factor = edge_factor
        self.aux_weight = aux_weight

    def get_boundary(self, x):
        laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32, device=x.device
        ).reshape(1, 1, 3, 3).requires_grad_(False)

        x = x.unsqueeze(1).float()  # [B, 1, H, W]
        x = F.conv2d(x, laplacian_kernel, padding=1)
        x = x.clamp(min=0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0
        return x

    def compute_edge_loss(self, pred, target):
        # pred: [B, 1, H, W], logits
        # target: [B, H, W]
        with torch.no_grad():
            pred_mask = (torch.sigmoid(pred) > 0.5).long().squeeze(1)
        boundary_preds = self.get_boundary(pred_mask).view(pred.size(0), -1)
        boundary_targets = self.get_boundary(target).view(pred.size(0), -1)

        edge_loss = F.binary_cross_entropy(boundary_preds, boundary_targets)
        return edge_loss

    def forward(self, logits, targets):
        if isinstance(logits, (tuple, list)) and len(logits) == 2:
            logit_main, logit_aux = logits
        else:
            logit_main = logits
            logit_aux = None

        targets = targets.float()  # BCE要求float标签

        main_loss = self.bce(logit_main, targets.unsqueeze(1))  # [B, 1, H, W]
        edge_loss = self.compute_edge_loss(logit_main, targets)

        total_loss = main_loss + self.edge_factor * edge_loss

        if self.training and logit_aux is not None:
            aux_loss = self.bce(logit_aux, targets.unsqueeze(1))
            total_loss += self.aux_weight * aux_loss

        return total_loss
