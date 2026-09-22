---
title: "论文标题"
venue: "会议/期刊 (如 MICCAI 2024)"
task: "3D 医学图像分割"
code_url: "https://github.com/..."
---

### 1. 核心创新点 (Key Innovation)
- 用一句话总结这篇文章的最大卖点（例如：结合了 Mamba 的全局感受野和 UNet 的局部特征）。

### 2. 对比基线与数据集 (Baseline & Datasets)
- **数据集**：BraTS 2021, MSD Liver
- **对比模型**：SwinUNETR, nnUNet, TransUNet

### 3. 性能表现 (Performance / SOTA)
- 在 MSD 肝脏肿瘤数据集上，Dice 达到了 0.96，HD95 降低到了 3.12 mm。

### 4. 局限性与启发 (Limitations & My Opportunities)
- **致命弱点**：作者在 Discussion 中提到推理显存占用过大，无法处理大于 128x128x128 的 Patch。
- **我的切入点**：我可以尝试引入轻量化注意力或者滑动窗口级联预测，来解决显存问题并超越它！
