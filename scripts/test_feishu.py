import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.notifier.feishu_notifier import FeishuNotifier
from core.visualizer.medical_visualizer import MedicalVisualizer
from core.datasets.medical_dataset import SyntheticMedicalDataset

def main():
    print("==========================================")
    print("🔬 正在测试 MedVision 飞书推送与渲染环境...")
    print("==========================================")

    # 1. 模拟生成一个 3D CT/MRI 样本并自动截取切片渲染
    print("[1/3] 模拟生成 3D 医疗影像并测试自动切片提取...")
    ds_3d = SyntheticMedicalDataset(num_samples=1, spatial_dims=3, size=(24, 64, 64))
    img, gt = ds_3d[0]
    
    # 模拟一个预测 mask (加轻微噪声)
    pred = gt.clone()
    pred[:, :, :10, :10] = 0.0

    test_img_path = str(PROJECT_ROOT / "experiments" / "test_slice_preview.png")
    MedicalVisualizer.plot_slice_comparison(
        image=img,
        gt_mask=gt,
        pred_mask=pred,
        save_path=test_img_path,
        title="3D CT Slice Auto-Extraction Preview"
    )
    print(f"  -> 切片对比图已成功生成: {test_img_path}")

    # 2. 测试曲线绘制
    print("[2/3] 测试训练收敛曲线绘制...")
    dummy_history = {
        "train_loss": [0.85, 0.62, 0.45, 0.31, 0.22],
        "val_loss": [0.89, 0.65, 0.48, 0.35, 0.25],
        "val_dice": [0.35, 0.58, 0.72, 0.81, 0.88]
    }
    curve_path = str(PROJECT_ROOT / "experiments" / "test_curve_preview.png")
    MedicalVisualizer.plot_convergence_curve(dummy_history, curve_path)
    print(f"  -> 收敛曲线图已成功生成: {curve_path}")

    # 3. 飞书连通性检查
    print("[3/3] 检查飞书机器人连通性...")
    from scripts.train import load_config
    cfg = load_config(str(PROJECT_ROOT / "configs" / "base.yaml"))
    feishu_cfg = cfg.get("feishu", {})
    webhook = feishu_cfg.get("webhook_url")

    notifier = FeishuNotifier(webhook_url=webhook)
    if webhook:
        print(f"  -> 检测到已配置 Webhook，正在发送测试卡片...")
        success = notifier.send_epoch_report_card(
            exp_name="Self-Test-Job",
            epoch=5,
            total_epochs=5,
            train_loss=0.22,
            val_loss=0.25,
            val_dice=0.88,
            best_dice=0.88,
            eta_str="已完成",
            image_path=test_img_path
        )
        if success:
            print("  -> 🎉 飞书测试卡片发送成功！请检查手机飞书群。")
        else:
            print("  -> ⚠️ 飞书测试卡片发送失败，请检查 Webhook 地址是否正确。")
    else:
        print("  -> ℹ️ 提示: configs/base.yaml 中未填写 webhook_url。")
        print("     您可以随时在 configs/base.yaml 中填入飞书机器人 Webhook 地址后重新运行本脚本进行验证。")

    print("\n✅ 环境与绘图自检通过！随时可启动 train.py 开启科研实验。")

if __name__ == "__main__":
    main()
