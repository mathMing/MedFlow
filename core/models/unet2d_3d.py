import torch
import torch.nn as nn
from typing import Optional, Dict, Any

# ======================= 2D UNet 实现 =======================
class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 32):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock2D(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock2D(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock2D(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock2D(f * 4, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock2D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock2D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock2D(f * 2, f)

        self.final = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# ======================= 3D UNet 实现 =======================
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 16):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock3D(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(f, f * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(f * 4, f * 8)

        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)

        self.final = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# ======================= 模型工厂 (支持 MONAI 拓展) =======================
def get_model(config: Dict[str, Any]) -> nn.Module:
    """根据配置动态实例化 2D/3D 模型，无缝兼容 MONAI"""
    spatial_dims = config.get("spatial_dims", 2)
    in_channels = config.get("in_channels", 1)
    out_channels = config.get("out_channels", 1)
    model_name = config.get("model_name", "unet").lower()
    base_filters = config.get("base_filters", 32 if spatial_dims == 2 else 16)

    # 尝试加载 MONAI 模型 (如果用户安装了 monai 并且配置指定)
    if config.get("use_monai", False):
        try:
            import monai.networks.nets as monai_nets
            if hasattr(monai_nets, model_name.upper()):
                print(f"[ModelFactory] 使用 MONAI 原生模型: {model_name.upper()}")
                cls = getattr(monai_nets, model_name.upper())
                return cls(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels)
        except ImportError:
            print("[ModelFactory] 未检测到 monai 安装，降级使用内置优化版 PyTorch 模型")

    # 原生 PyTorch 2D / 3D 实现
    if spatial_dims == 3:
        print(f"[ModelFactory] 实例化 3D UNet (in={in_channels}, out={out_channels}, base={base_filters})")
        return UNet3D(in_channels=in_channels, out_channels=out_channels, base_filters=base_filters)
    else:
        print(f"[ModelFactory] 实例化 2D UNet (in={in_channels}, out={out_channels}, base={base_filters})")
        return UNet2D(in_channels=in_channels, out_channels=out_channels, base_filters=base_filters)
