import os
import sys
import time
import yaml
import argparse
import traceback
from pathlib import Path
import torch

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.models.unet2d_3d import get_model
from core.datasets.medical_dataset import get_dataloaders
from core.metrics.medical_metrics import CombinedLoss, compute_dice_score, compute_iou_score
from core.visualizer.medical_visualizer import MedicalVisualizer
from core.notifier.feishu_notifier import FeishuNotifier

def load_config(config_path: str) -> dict:
    """加载配置并与 base.yaml 深度合并"""
    base_cfg_path = PROJECT_ROOT / "configs" / "base.yaml"
    cfg = {}
    if base_cfg_path.exists():
        with open(base_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    with open(config_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f) or {}
    
    cfg.update(exp_cfg)
    return cfg

def main():
    parser = argparse.ArgumentParser(description="MedVision 统一科研训练流水线")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "exp_2d_demo.yaml"),
                        help="配置文件路径 (YAML)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg.get("exp_name", "experiment")
    exp_dir = PROJECT_ROOT / cfg.get("save_dir", "experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. 硬件设备检测
    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"\n==========================================")
    print(f"[MedVision] 任务启动: {exp_name}")
    print(f"[MedVision] 计算设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Mode'})")
    print(f"[MedVision] 任务空间维度: {cfg.get('spatial_dims')}D | 模型: {cfg.get('model_name')}")
    print(f"==========================================\n")

    # 2. 初始化飞书推送器
    feishu_cfg = cfg.get("feishu", {})
    notifier = FeishuNotifier(
        webhook_url=feishu_cfg.get("webhook_url"),
        app_id=feishu_cfg.get("app_id"),
        app_secret=feishu_cfg.get("app_secret")
    )

    # 推送启动卡片
    summary_for_card = {
        "任务名称": exp_name,
        "数据空间": f"{cfg.get('spatial_dims')}D 医疗影像",
        "网络模型": cfg.get("model_name"),
        "总轮次 (Epochs)": cfg.get("epochs", 10),
        "硬件环境": str(device)
    }
    notifier.send_training_start_card(exp_name, summary_for_card)

    try:
        # 3. 构建数据与模型
        train_loader, val_loader = get_dataloaders(cfg)
        model = get_model(cfg).to(device)

        criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))

        epochs = cfg.get("epochs", 10)
        best_dice = 0.0
        history = {"train_loss": [], "val_loss": [], "val_dice": []}

        start_time = time.time()

        # 4. 训练主循环
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            model.train()
            total_train_loss = 0.0

            for batch_idx, (images, masks) in enumerate(train_loader):
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_loader))
            history["train_loss"].append(avg_train_loss)

            # 5. 验证与指标评估
            if epoch % cfg.get("val_interval", 1) == 0:
                model.eval()
                total_val_loss = 0.0
                total_dice = 0.0
                last_sample = None

                with torch.no_grad():
                    for val_images, val_masks in val_loader:
                        val_images, val_masks = val_images.to(device), val_masks.to(device)
                        val_outputs = model(val_images)
                        v_loss = criterion(val_outputs, val_masks)
                        total_val_loss += v_loss.item()
                        total_dice += compute_dice_score(val_outputs, val_masks)
                        last_sample = (val_images[0], val_masks[0], val_outputs[0])

                avg_val_loss = total_val_loss / max(1, len(val_loader))
                avg_val_dice = total_dice / max(1, len(val_loader))
                history["val_loss"].append(avg_val_loss)
                history["val_dice"].append(avg_val_dice)

                # 计算剩余预估时间 (ETA)
                elapsed = time.time() - start_time
                avg_epoch_time = elapsed / epoch
                remaining_epochs = epochs - epoch
                eta_seconds = int(avg_epoch_time * remaining_epochs)
                eta_str = f"{eta_seconds // 60}分{eta_seconds % 60}秒" if eta_seconds > 0 else "即将完成"

                print(f"[Epoch {epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f} | ETA: {eta_str}")

                # 6. 生成切片可视化与收敛曲线
                slice_img_path = str(exp_dir / f"val_slice_epoch_{epoch}.png")
                curve_img_path = str(exp_dir / "convergence_curve.png")

                if last_sample:
                    MedicalVisualizer.plot_slice_comparison(
                        image=last_sample[0],
                        gt_mask=last_sample[1],
                        pred_mask=last_sample[2],
                        save_path=slice_img_path,
                        title=f"{exp_name} - Epoch {epoch} Slice Validation"
                    )

                MedicalVisualizer.plot_convergence_curve(history, curve_img_path)

                # 7. 保存最优 Checkpoint
                if avg_val_dice > best_dice:
                    best_dice = avg_val_dice
                    torch.save(model.state_dict(), exp_dir / "best_model.pth")
                    print(f"  --> 🌟 刷新最佳 Dice: {best_dice:.4f}，已保存权重")

                # 8. 实时向手机飞书发送汇报卡片
                notifier.send_epoch_report_card(
                    exp_name=exp_name,
                    epoch=epoch,
                    total_epochs=epochs,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    val_dice=avg_val_dice,
                    best_dice=best_dice,
                    eta_str=eta_str,
                    image_path=slice_img_path
                )

        # 训练结束保存
        torch.save(model.state_dict(), exp_dir / "last_model.pth")
        print(f"\n✅ 实验 {exp_name} 顺利完成！最佳验证集 Dice: {best_dice:.4f}")
        notifier.send_text(f"🎉 实验 {exp_name} 已全流程训练完成！最优 Dice 得分: {best_dice:.4f}。权重与图表已归档至工作站。")

    except Exception as e:
        err_msg = str(e)
        tb_str = traceback.format_exc()
        print(f"\n🚨 [ERROR] 训练异常中断: {err_msg}")
        print(tb_str)
        notifier.send_error_alert_card(exp_name, err_msg, tb_str)
        sys.exit(1)

if __name__ == "__main__":
    main()
