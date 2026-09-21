import os
import sys
import time
import json
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.notifier.feishu_notifier import FeishuNotifier

class SystemMonitor:
    """跨平台 (Windows & Ubuntu) 显卡与实验进程监控工具"""

    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """获取主机所有显卡状态"""
        gpu_list = []
        try:
            # 优先尝试 pynvml
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_list.append({
                    "index": i,
                    "name": name,
                    "memory_used_mb": mem_info.used / (1024 ** 2),
                    "memory_total_mb": mem_info.total / (1024 ** 2),
                    "gpu_util_percent": util.gpu
                })
            pynvml.nvmlShutdown()
            return gpu_list
        except Exception:
            pass

        # 降级尝试 nvidia-smi 命令行
        try:
            cmd = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
            res = subprocess.check_output(cmd, encoding="utf-8", timeout=5)
            for line in res.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpu_list.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_used_mb": float(parts[2]),
                        "memory_total_mb": float(parts[3]),
                        "gpu_util_percent": float(parts[4])
                    })
            return gpu_list
        except Exception:
            pass

        # 无显卡或非 NVIDIA 环境 (如 CPU 调试模式)
        return [{
            "index": 0,
            "name": "CPU / Virtual Mode",
            "memory_used_mb": 0,
            "memory_total_mb": 16384,
            "gpu_util_percent": 0
        }]

class LabExperimentDispatcher:
    """管理在本地跑的后台实验进程"""
    def __init__(self, notifier: FeishuNotifier):
        self.notifier = notifier
        self.current_process: Optional[subprocess.Popen] = None
        self.current_exp_name: Optional[str] = None
        self.start_time: Optional[float] = None

    def execute_command(self, cmd_text: str) -> str:
        """解析手机端发来的指令"""
        cmd = cmd_text.strip()
        tokens = cmd.split()
        if not tokens:
            return "请输入指令，例如: /gpu, /status, /run exp_2d_demo, /stop"

        action = tokens[0].lower()

        if action == "/gpu":
            gpus = SystemMonitor.get_gpu_info()
            self.notifier.send_gpu_status_card(gpus)
            return "已向飞书推送最新显卡状态卡片！"

        elif action == "/status":
            if self.current_process and self.current_process.poll() is None:
                elapsed = int(time.time() - self.start_time)
                msg = f"⏳ 实验【{self.current_exp_name}】正在运行中...\n• 已运行时间: {elapsed // 60}分{elapsed % 60}秒\n• PID: {self.current_process.pid}"
            else:
                msg = "💤 当前无正在运行的实验，主机处于空闲就绪状态。\n可使用 /run <配置名> 启动新实验。"
            self.notifier.send_text(msg)
            return msg

        elif action == "/run":
            if len(tokens) < 2:
                return "请指定实验配置文件，例如: /run exp_2d_demo 或 /run exp_3d_demo"
            exp_cfg_name = tokens[1]
            if not exp_cfg_name.endswith(".yaml"):
                exp_cfg_name += ".yaml"
            cfg_file = PROJECT_ROOT / "configs" / exp_cfg_name
            if not cfg_file.exists():
                return f"❌ 找不到配置文件: {exp_cfg_name}，请检查 configs/ 目录"

            if self.current_process and self.current_process.poll() is None:
                return f"⚠️ 当前已有实验【{self.current_exp_name}】正在运行，请等待其完成或先发送 /stop 终止。"

            # 启动实验子进程
            python_exe = sys.executable
            train_script = str(PROJECT_ROOT / "scripts" / "train.py")
            cmd_args = [python_exe, train_script, "--config", str(cfg_file)]
            
            self.current_process = subprocess.Popen(cmd_args, cwd=str(PROJECT_ROOT))
            self.current_exp_name = exp_cfg_name.replace(".yaml", "")
            self.start_time = time.time()
            msg = f"🚀 正在后台拉起实验: {self.current_exp_name} (PID: {self.current_process.pid})"
            self.notifier.send_text(msg)
            return msg

        elif action == "/stop":
            if self.current_process and self.current_process.poll() is None:
                self.current_process.terminate()
                msg = f"🛑 已成功向实验【{self.current_exp_name}】发送终止信号。"
                self.current_process = None
                self.current_exp_name = None
            else:
                msg = "当前无正在运行的实验。"
            self.notifier.send_text(msg)
            return msg

        else:
            return f"未知指令: {action}。支持的指令包括: /gpu, /status, /run <exp>, /stop"

def start_bot_service():
    """启动飞书机器人监听服务"""
    base_cfg_path = PROJECT_ROOT / "configs" / "base.yaml"
    cfg = {}
    if base_cfg_path.exists():
        with open(base_cfg_path, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f) or {}

    feishu_cfg = cfg.get("feishu", {})
    notifier = FeishuNotifier(
        webhook_url=feishu_cfg.get("webhook_url"),
        app_id=feishu_cfg.get("app_id"),
        app_secret=feishu_cfg.get("app_secret")
    )
    dispatcher = LabExperimentDispatcher(notifier)

    print("\n=======================================================")
    print("🤖 MedVision 飞书移动端控制守护进程已启动")
    print("=======================================================")
    print("支持指令:")
    print("  /gpu              -> 查询主机显卡与显存占用")
    print("  /status           -> 查询当前实验运行状态")
    print("  /run <config>     -> 远程启动新实验 (如 /run exp_2d_demo)")
    print("  /stop             -> 终止正在运行的实验")
    print("=======================================================\n")

    # 检查是否已配置官方 lark-oapi WebSocket 长连接
    app_id = feishu_cfg.get("app_id")
    app_secret = feishu_cfg.get("app_secret")

    if app_id and app_secret:
        try:
            import lark_oapi as lark
            from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

            def on_message(data: P2ImMessageReceiveV1):
                try:
                    content_json = json.loads(data.event.message.content)
                    text = content_json.get("text", "").strip()
                    print(f"[Mobile Command Received] {text}")
                    reply = dispatcher.execute_command(text)
                except Exception as e:
                    print(f"处理手机消息异常: {e}")

            print("[FeishuBot] 正在建立飞书免公网IP长连接 (WebSocket)...")
            event_handler = lark.EventDispatcherHandler.builder("", "") \
                .register_p2_im_message_receive_v1(on_message) \
                .build()

            client = lark.ws.Client(
                app_id=app_id,
                app_secret=app_secret,
                event_handler=event_handler,
                log_level=lark.LogLevel.INFO
            )
            client.start()
            return
        except ImportError:
            print("[FeishuBot] 提示: 未安装 lark-oapi。如需开启长连接收消息，请运行: pip install lark-oapi")
        except Exception as e:
            print(f"[FeishuBot] 建立长连接异常: {e}")

    # 本地交互模拟（或仅推送模式）
    print("[FeishuBot] 处于本地控制台待命模式。您可以直接在此终端输入手机同款指令进行测试，或通过 Webhook 推送汇报。")
    while True:
        try:
            line = input("MedVision-Shell> ")
            if line.strip().lower() in ["exit", "quit"]:
                break
            res = dispatcher.execute_command(line)
            print(f">> {res}")
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    start_bot_service()
