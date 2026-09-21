import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Dict, Any, Optional

class SyntheticMedicalDataset(Dataset):
    """
    轻量级 2D/3D 模拟医疗影像数据集：
    用于即时环境自测与流水线调优，无需立即下载数 10GB 真实数据。
    自动生成模拟组织背景及随机分布的圆形/椭球形病灶 (Lesion / Organ)。
    """
    def __init__(
        self,
        num_samples: int = 30,
        spatial_dims: int = 2,
        size: Tuple[int, ...] = (128, 128),
        in_channels: int = 1,
        out_channels: int = 1
    ):
        self.num_samples = num_samples
        self.spatial_dims = spatial_dims
        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        np.random.seed(idx)
        if self.spatial_dims == 2:
            h, w = self.size
            # 基础背景噪音 (类似超声或 X 光)
            img = np.random.normal(loc=0.3, scale=0.1, size=(self.in_channels, h, w)).astype(np.float32)
            mask = np.zeros((self.out_channels, h, w), dtype=np.float32)

            # 生成随机圆形病灶
            cx, cy = np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4)
            radius = np.random.randint(min(h, w) // 10, min(h, w) // 4)
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            lesion_region = dist_from_center <= radius

            mask[0, lesion_region] = 1.0
            # 提高病灶区域灰度对比度
            img[0, lesion_region] += np.random.uniform(0.4, 0.7)
            img = np.clip(img, 0.0, 1.0)

        elif self.spatial_dims == 3:
            d, h, w = self.size
            # 模拟 CT/MRI 体素
            img = np.random.normal(loc=0.2, scale=0.08, size=(self.in_channels, d, h, w)).astype(np.float32)
            mask = np.zeros((self.out_channels, d, h, w), dtype=np.float32)

            # 生成 3D 椭球体病灶
            cd = np.random.randint(d // 4, 3 * d // 4)
            ch = np.random.randint(h // 4, 3 * h // 4)
            cw = np.random.randint(w // 4, 3 * w // 4)
            rd, rh, rw = np.random.randint(4, 10), np.random.randint(8, 20), np.random.randint(8, 20)

            z, y, x = np.ogrid[:d, :h, :w]
            dist = ((z - cd) / rd) ** 2 + ((y - ch) / rh) ** 2 + ((x - cw) / rw) ** 2
            lesion_region = dist <= 1.0

            mask[0, lesion_region] = 1.0
            img[0, lesion_region] += np.random.uniform(0.35, 0.6)
            img = np.clip(img, 0.0, 1.0)
        else:
            raise ValueError(f"不支持的维度: {self.spatial_dims}")

        return torch.from_numpy(img), torch.from_numpy(mask)

def get_dataloaders(config: Dict[str, Any]):
    """数据加载器工厂函数"""
    spatial_dims = config.get("spatial_dims", 2)
    batch_size = config.get("batch_size", 4)
    data_shape = tuple(config.get("data_shape", [128, 128]))

    # 目前使用模拟医疗影像数据集作为自测与演示
    train_ds = SyntheticMedicalDataset(
        num_samples=config.get("train_samples", 40),
        spatial_dims=spatial_dims,
        size=data_shape,
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1)
    )
    val_ds = SyntheticMedicalDataset(
        num_samples=config.get("val_samples", 10),
        spatial_dims=spatial_dims,
        size=data_shape,
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1)
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0 # Windows 兼容防死锁
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1, # 验证集逐个切片精细评估
        shuffle=False,
        num_workers=0
    )
    return train_loader, val_loader
