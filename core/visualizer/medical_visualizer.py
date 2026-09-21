import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无图形界面服务器安全渲染
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import Optional, Tuple

class MedicalVisualizer:
    """
    医疗影像可视化与战报渲染器：
    1. 2D/3D 切片掩膜叠加图 (Overlay): 绿色=GT, 红色=Pred, 重合=黄色
    2. 训练指标动态收敛曲线 (Loss + Dice Curve)
    """

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().float().numpy()
        return np.asarray(tensor, dtype=np.float32)

    @classmethod
    def plot_slice_comparison(
        cls,
        image: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_mask: torch.Tensor,
        save_path: str,
        title: str = "Validation Sample Comparison"
    ) -> str:
        """
        自动解析 2D 或 3D 样本，生成三栏对比图 (原图, 标注与预测叠加对比, 预测细节)
        image: [C, (D), H, W]
        gt_mask: [C, (D), H, W]
        pred_mask: [C, (D), H, W]
        """
        img_np = cls._to_numpy(image)
        gt_np = cls._to_numpy(gt_mask)
        pred_np = cls._to_numpy(pred_mask)

        # 二值化预测
        if pred_np.min() < 0 or pred_np.max() > 1:
            pred_np = 1.0 / (1.0 + np.exp(-pred_np))
        pred_bin = (pred_np > 0.5).astype(np.float32)
        gt_bin = (gt_np > 0.5).astype(np.float32)

        # 判断是否为 3D 体素数据 (判断维度)
        if img_np.ndim == 4:  # [C, D, H, W]
            c, d, h, w = img_np.shape
            # 自动搜寻病灶/目标器官截面积最大的切片 index
            gt_area_per_slice = gt_bin[0].sum(axis=(1, 2))
            if gt_area_per_slice.max() > 0:
                best_slice_idx = int(np.argmax(gt_area_per_slice))
            else:
                best_slice_idx = d // 2

            slice_img = img_np[0, best_slice_idx]
            slice_gt = gt_bin[0, best_slice_idx]
            slice_pred = pred_bin[0, best_slice_idx]
            slice_info = f"Axial Slice #{best_slice_idx}/{d}"
        else:  # 2D 数据 [C, H, W]
            slice_img = img_np[0]
            slice_gt = gt_bin[0]
            slice_pred = pred_bin[0]
            slice_info = "2D Single Frame"

        # 归一化图像灰度
        p_min, p_max = slice_img.min(), slice_img.max()
        if p_max > p_min:
            norm_img = (slice_img - p_min) / (p_max - p_min)
        else:
            norm_img = slice_img

        # 构造 RGBA 掩膜叠加
        # GT = 绿色 (0, 1, 0), Pred = 红色 (1, 0, 0), 重合 = 黄色
        h, w = norm_img.shape
        overlay_rgb = np.stack([norm_img, norm_img, norm_img], axis=-1)

        gt_mask_bool = slice_gt > 0
        pred_mask_bool = slice_pred > 0

        # 半透明叠加 (Alpha Blending)
        alpha = 0.45
        # 绿色: GT
        overlay_rgb[gt_mask_bool, 1] = np.clip(overlay_rgb[gt_mask_bool, 1] * (1 - alpha) + 1.0 * alpha, 0, 1)
        # 红色: Pred
        overlay_rgb[pred_mask_bool, 0] = np.clip(overlay_rgb[pred_mask_bool, 0] * (1 - alpha) + 1.0 * alpha, 0, 1)

        # 画布绘制
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=150)

        # 1. 原始影像
        axes[0].imshow(norm_img, cmap='gray')
        axes[0].set_title(f"Original ({slice_info})", fontsize=11)
        axes[0].axis('off')

        # 2. GT 真实标注
        axes[1].imshow(norm_img, cmap='gray')
        masked_gt = np.ma.masked_where(slice_gt == 0, slice_gt)
        axes[1].imshow(masked_gt, cmap='Greens', alpha=0.6, vmin=0, vmax=1)
        axes[1].set_title("Ground Truth (Green)", fontsize=11)
        axes[1].axis('off')

        # 3. 预测叠加对比
        axes[2].imshow(overlay_rgb)
        axes[2].set_title("Overlay (Green=GT, Red=Pred, Yellow=Match)", fontsize=11)
        axes[2].axis('off')

        plt.suptitle(title, fontsize=13, weight='bold', y=0.98)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path

    @classmethod
    def plot_convergence_curve(
        cls,
        history: dict,
        save_path: str,
        title: str = "Training Convergence Curves"
    ) -> str:
        """绘制 Loss 和 Dice 随 Epoch 演进曲线"""
        epochs = range(1, len(history.get("train_loss", [])) + 1)
        if not epochs:
            return save_path

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

        # Loss 曲线
        ax1.plot(epochs, history.get("train_loss", []), 'b-', label='Train Loss', lw=1.8)
        if "val_loss" in history and history["val_loss"]:
            ax1.plot(epochs, history.get("val_loss", []), 'r--', label='Val Loss', lw=1.8)
        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.set_title('Loss Curve', fontsize=11, weight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()

        # Dice 曲线
        if "val_dice" in history and history["val_dice"]:
            ax2.plot(epochs, history.get("val_dice", []), 'g-', label='Val Dice (DSC)', lw=2.0)
            ax2.set_xlabel('Epoch', fontsize=10)
            ax2.set_ylabel('Dice Score', fontsize=10)
            ax2.set_title('Validation Dice (Higher is better)', fontsize=11, weight='bold')
            ax2.grid(True, linestyle=':', alpha=0.6)
            ax2.set_ylim([0.0, 1.02])
            ax2.legend()

        plt.suptitle(title, fontsize=12, weight='bold')
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path
