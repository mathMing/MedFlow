import os
import json
import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List

class FeishuNotifier:
    """
    飞书消息与战报推送器：
    支持：
    1. 飞书自定义群机器人 Webhook (单向极速推送文本、富文本卡片)
    2. 飞书开放平台应用 API (使用 app_id 和 app_secret 上传图片并在卡片中展示高清医疗切片)
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
    ):
        self.webhook_url = webhook_url or os.getenv("FEISHU_WEBHOOK_URL", "")
        self.app_id = app_id or os.getenv("FEISHU_APP_ID", "")
        self.app_secret = app_secret or os.getenv("FEISHU_APP_SECRET", "")
        self._tenant_access_token = None

    def _get_tenant_access_token(self) -> Optional[str]:
        """获取飞书应用凭据 tenant_access_token，用于上传图片"""
        if not self.app_id or not self.app_secret:
            return None
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        headers = {"Content-Type": "application/json; charset=utf-8"}
        payload = {"app_id": self.app_id, "app_secret": self.app_secret}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            data = resp.json()
            if data.get("code") == 0:
                self._tenant_access_token = data.get("tenant_access_token")
                return self._tenant_access_token
            else:
                print(f"[FeishuNotifier] 获取 Token 失败: {data.get('msg')}")
        except Exception as e:
            print(f"[FeishuNotifier] 获取 Token 异常: {e}")
        return None

    def upload_image(self, image_path: str) -> Optional[str]:
        """上传本地生成的 2D/3D 切片或 Loss 曲线图片到飞书，返回 image_key"""
        token = self._get_tenant_access_token()
        if not token:
            return None
        
        path = Path(image_path)
        if not path.exists():
            print(f"[FeishuNotifier] 图片文件不存在: {image_path}")
            return None

        url = "https://open.feishu.cn/open-apis/im/v1/images"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            with open(path, "rb") as f:
                files = {"image": f}
                data = {"image_type": "message"}
                resp = requests.post(url, headers=headers, files=files, data=data, timeout=30)
                res = resp.json()
                if res.get("code") == 0:
                    return res["data"]["image_key"]
                else:
                    print(f"[FeishuNotifier] 上传图片失败: {res.get('msg')}")
        except Exception as e:
            print(f"[FeishuNotifier] 上传图片异常: {e}")
        return None

    def send_card(self, card_dict: Dict[str, Any]) -> bool:
        """发送飞书互动卡片"""
        if not self.webhook_url:
            print("[FeishuNotifier] 未配置 FEISHU_WEBHOOK_URL，跳过飞书推送")
            return False

        payload = {
            "msg_type": "interactive",
            "card": card_dict
        }
        headers = {"Content-Type": "application/json; charset=utf-8"}
        try:
            resp = requests.post(self.webhook_url, headers=headers, json=payload, timeout=10)
            res = resp.json()
            if res.get("code") == 0 or res.get("StatusCode") == 0:
                return True
            else:
                print(f"[FeishuNotifier] 推送卡片失败: {res}")
        except Exception as e:
            print(f"[FeishuNotifier] 推送网络异常: {e}")
        return False

    def send_text(self, text: str) -> bool:
        """发送简单纯文本消息"""
        if not self.webhook_url:
            print(f"[Feishu Local Log] {text}")
            return False
        payload = {"msg_type": "text", "content": {"text": text}}
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"[FeishuNotifier] 发送文本异常: {e}")
            return False

    def send_training_start_card(self, exp_name: str, config_summary: Dict[str, Any]):
        """实验启动通知卡片"""
        fields = []
        for k, v in config_summary.items():
            fields.append({
                "is_short": True,
                "text": {"tag": "lark_md", "content": f"**{k}**: {v}"}
            })

        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "blue",
                "title": {"tag": "plain_text", "content": f"🚀 实验已启动: {exp_name}"}
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": "实验室工作站已成功拉起训练任务，详细配置如下："}
                },
                {"tag": "hr"},
                {"tag": "div", "fields": fields},
                {
                    "tag": "note",
                    "elements": [{"tag": "plain_text", "content": "手机端可随时发送 /status 查看实时指标"}]
                }
            ]
        }
        return self.send_card(card)

    def send_epoch_report_card(
        self,
        exp_name: str,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        val_dice: float,
        best_dice: float,
        eta_str: str = "计算中...",
        image_path: Optional[str] = None
    ):
        """Epoch 验证汇报卡片（包含指标与 3D/2D 切片图）"""
        # 判断卡片主题颜色
        is_best = (val_dice >= best_dice and val_dice > 0)
        template_color = "green" if is_best else "indigo"
        title_prefix = "🌟 刷新最优指标!" if is_best else "📊 训练阶段汇报"

        fields = [
            {"is_short": True, "text": {"tag": "lark_md", "content": f"**当前轮次**: {epoch}/{total_epochs}"}},
            {"is_short": True, "text": {"tag": "lark_md", "content": f"**预估剩余时间**: {eta_str}"}},
            {"is_short": True, "text": {"tag": "lark_md", "content": f"**Train Loss**: `{train_loss:.4f}`"}},
            {"is_short": True, "text": {"tag": "lark_md", "content": f"**Val Loss**: `{val_loss:.4f}`"}},
            {"is_short": True, "text": {"tag": "lark_md", "content": f"**Val Dice (当前)**: `{val_dice:.4f}`"}},
            {"is_short": True, "text": {"tag": "lark_md", "content": f"**Best Dice (历史)**: `{best_dice:.4f}`"}},
        ]

        elements = [
            {"tag": "div", "fields": fields},
            {"tag": "hr"}
        ]

        # 如果有图片且配置了 App 凭据，则尝试上传
        if image_path and Path(image_path).exists():
            img_key = self.upload_image(image_path)
            if img_key:
                elements.append({
                    "tag": "img",
                    "img_key": img_key,
                    "alt": {"tag": "plain_text", "content": "医疗影像切片验证结果与分割对比"}
                })
            else:
                elements.append({
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": f"*(验证切片已生成并保存在主机: `{Path(image_path).name}`)*"}
                })

        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": template_color,
                "title": {"tag": "plain_text", "content": f"{title_prefix} [{exp_name}]"}
            },
            "elements": elements
        }
        return self.send_card(card)

    def send_error_alert_card(self, exp_name: str, error_msg: str, traceback_str: str):
        """训练异常或 OOM 紧急报警卡片"""
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "red",
                "title": {"tag": "plain_text", "content": f"🚨 实验异常中断告警 [{exp_name}]"}
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": f"**错误描述**: <font color='red'>{error_msg}</font>"}
                },
                {"tag": "hr"},
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": f"**堆栈定位**:\n```text\n{traceback_str[:800]}\n```"}
                },
                {
                    "tag": "note",
                    "elements": [{"tag": "plain_text", "content": "建议检查显存配置 (Batch Size / Crop Size) 或数据格式"}]
                }
            ]
        }
        return self.send_card(card)

    def send_gpu_status_card(self, gpu_info_list: List[Dict[str, Any]]):
        """显卡状态实时汇报卡片"""
        elements = []
        for gpu in gpu_info_list:
            gpu_idx = gpu.get("index", 0)
            gpu_name = gpu.get("name", "Unknown GPU")
            mem_used = gpu.get("memory_used_mb", 0)
            mem_total = gpu.get("memory_total_mb", 1)
            util = gpu.get("gpu_util_percent", 0)
            mem_percent = (mem_used / mem_total) * 100

            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        f"**GPU {gpu_idx}: {gpu_name}**\n"
                        f"• 核心利用率: `{util}%`\n"
                        f"• 显存占用: `{mem_used:.0f} MB / {mem_total:.0f} MB ({mem_percent:.1f}%)`"
                    )
                }
            })
            elements.append({"tag": "hr"})

        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "turquoise",
                "title": {"tag": "plain_text", "content": "🖥️ 实验室显卡状态看板"}
            },
            "elements": elements
        }
        return self.send_card(card)
