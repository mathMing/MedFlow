# MedVision 科研流水线与飞书移动端控制系统

专为 **计算机视觉与医疗影像处理（2D & 3D）** 科研人员打造的自动化无人值守实验平台。

---

## 🌟 核心特性

1. **📱 手机飞书随时掌控**：
   - 支持移动端查看实时显卡占用（`/gpu`）与实验进程（`/status`）。
   - 手机端远程下达启动任务指令（`/run exp_3d_demo`）与急停指令（`/stop`）。
   - 自动推送结构化富文本科研战报与切片图。
2. **🧠 医疗影像 2D/3D 通用解耦设计**：
   - 3D CT/MRI（如 `.nii.gz`）与 2D（内窥镜/病理/X光）全面支持。
   - 智能切片提取引擎：自动定位 3D 病灶最大截面，并生成高精度三色半透明对比图（绿色=GT，红色=Pred，黄色=重合）。
   - 医疗金标准评估：Dice 相似度系数（DSC）、IoU、Dice+BCE 联合可导损失函数。
3. **💻 Windows & Ubuntu 跨平台原生支持**：
   - 统一路径抽象，自动适应单卡/多卡/CPU 调试环境。
   - 免公网 IP：支持飞书长连接（WebSocket）直接穿透校园网/医院内网防火墙。
4. **🔌 PyTorch & MONAI 兼容**：
   - 基础自带极速高效的 PyTorch 2D/3D UNet。
   - 预留 MONAI 工厂接口，随时一键升级为 SwinUNETR、SegResNet 等前沿骨干网络。

---

## 🚀 快速上手 (1 分钟极速体验)

### 1. 运行自检与绘图测试
在终端中进入项目目录运行测试脚本：
```bash
python scripts/test_feishu.py
```
该脚本会自动：
- 生成模拟 3D 医疗影像数据。
- 自动提取病灶切片并渲染高分辨率三栏对比图（保存至 `experiments/test_slice_preview.png`）。
- 绘制收敛曲线（保存至 `experiments/test_curve_preview.png`）。

### 2. 跑一个 2D 或 3D 训练实验
```bash
# 启动 2D 任务演示
python scripts/train.py --config configs/exp_2d_demo.yaml

# 启动 3D 任务演示 (3D CT/MRI 体素训练)
python scripts/train.py --config configs/exp_3d_demo.yaml
```

---

## 📲 配置手机飞书通知（只需 1 分钟）

### 极速模式：仅接收通知与战报 (Webhook 方式)
1. 打开电脑端或手机端飞书，进入任意个人群聊（或新建一个专属科研群“我的实验助手”）。
2. 点击群设置 ➔ **群机器人** ➔ **添加机器人** ➔ 选择 **自定义机器人**。
3. 给机器人起名为“实验监控助手”，复制生成的 **Webhook 地址**。
4. 打开 `configs/base.yaml`，将地址填入：
   ```yaml
   feishu:
     enable: true
     webhook_url: "https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxx"
   ```
5. 再次运行 `python scripts/test_feishu.py`，手机即可立即收到第一张带切片图的科研战报！

### 双向交互模式：手机发送指令调度实验 (免公网 IP)
如需从手机直接发送 `/gpu`, `/status`, `/run exp_3d_demo`：
1. 访问 [飞书开放平台](https://open.feishu.cn/)，点击“创建企业自建应用”。
2. 在“凭证与基础信息”中获取 `App ID` 和 `App Secret`。
3. 在“事件与回调”中开启 **长连接模式 (WebSocket)**，并添加权限 `im:message:send_as_bot`（接收并回复消息）。
4. 将 `app_id` 与 `app_secret` 填入 `configs/base.yaml`。
5. 在主机后台启动常驻监听：
   ```bash
   python scripts/feishu_bot.py
   ```
   此时无论你在宿舍、食堂还是路上，用手机在飞书中给机器人发消息，实验主机都会秒级响应！

---

## 📂 目录结构与规范

```text
medvision_pipeline/
├── configs/                   # 实验配置文件 (改参数只需改 YAML)
│   ├── base.yaml              # 全局通用配置 (设备/飞书凭据/学习率)
│   ├── exp_2d_demo.yaml       # 2D 任务配置
│   └── exp_3d_demo.yaml       # 3D 任务配置
├── core/                      # 核心模块
│   ├── datasets/              # 数据集定义 (Synthetic / NIfTI)
│   ├── metrics/               # 医疗金标准评估 (Dice, IoU, CombinedLoss)
│   ├── models/                # 2D/3D UNet (支持 MONAI 拓展)
│   ├── notifier/              # 飞书卡片与图文推送引擎
│   └── visualizer/            # 智能病灶切片抽取与收敛曲线
├── scripts/                   # 顶层执行脚本
│   ├── train.py               # 统一训练主入口 (自动保存最佳模型与切片)
│   ├── feishu_bot.py          # 手机端指令守护进程
│   └── test_feishu.py         # 连通性测试工具
└── requirements.txt           # 运行依赖
```
