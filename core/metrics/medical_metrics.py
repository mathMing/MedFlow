import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def compute_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5
) -> float:
    """
    计算二分类或多通道的真实 Dice 相似度系数 (DSC)
    pred: [B, C, (D), H, W] 模型原始 logits 或概率
    target: [B, C, (D), H, W] 对应通道的 0/1 标注
    """
    with torch.no_grad():
        if pred.shape != target.shape:
            # 兼容 target 为 [B, 1, ...] 或 [B, ...]
            if target.dim() == pred.dim() - 1:
                target = target.unsqueeze(1)

        # 概率化二值化
        if pred.min() < 0 or pred.max() > 1:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred

        pred_bin = (pred_prob > threshold).float()
        target_f = target.float()

        dims = tuple(range(2, pred.dim())) # 沿空间维度累加
        intersection = torch.sum(pred_bin * target_f, dim=dims)
        union = torch.sum(pred_bin, dim=dims) + torch.sum(target_f, dim=dims)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean().item()

def compute_iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5
) -> float:
    """计算 Jaccard / IoU 系数"""
    with torch.no_grad():
        if pred.shape != target.shape and target.dim() == pred.dim() - 1:
            target = target.unsqueeze(1)

        if pred.min() < 0 or pred.max() > 1:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred

        pred_bin = (pred_prob > threshold).float()
        target_f = target.float()

        dims = tuple(range(2, pred.dim()))
        intersection = torch.sum(pred_bin * target_f, dim=dims)
        total = torch.sum(pred_bin, dim=dims) + torch.sum(target_f, dim=dims)
        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)
        return iou.mean().item()

class DiceLoss(nn.Module):
    """可导的 Soft Dice Loss，专用于医疗图像分割"""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == pred.dim() - 1:
            target = target.unsqueeze(1)
        target = target.float()

        pred_prob = torch.sigmoid(pred)
        dims = tuple(range(2, pred.dim()))

        intersection = torch.sum(pred_prob * target, dim=dims)
        cardinality = torch.sum(pred_prob.pow(2) + target.pow(2), dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    """Dice Loss + BCE Loss (医疗分割最经典鲁棒的联合损失)"""
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == pred.dim() - 1:
            target = target.unsqueeze(1)
        target = target.float()

        l_dice = self.dice_loss(pred, target)
        l_bce = self.bce_loss(pred, target)
        return self.dice_weight * l_dice + self.bce_weight * l_bce
