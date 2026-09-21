import os
import sys
import yaml
import subprocess
import streamlit as st
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "configs"
EXP_DIR = PROJECT_ROOT / "experiments"

st.set_page_config(page_title="MedVision 科研流水线", page_icon="🔬", layout="wide")

# ================= 状态管理 =================
if "process" not in st.session_state:
    st.session_state.process = None

def get_config_files():
    if not CONFIG_DIR.exists():
        return []
    return [f.name for f in CONFIG_DIR.glob("*.yaml")]

# ================= 侧边栏导航 =================
st.sidebar.title("🔬 MedVision Web")
st.sidebar.markdown("轻量级医疗影像科研控制台")
menu = st.sidebar.radio("功能导航", ["📖 快速引导", "⚙️ 实验配置与执行", "📊 实时战报看板"])

# ================= 页面 1: 快速引导 =================
if menu == "📖 快速引导":
    st.title("欢迎使用 MedVision 医疗影像流水线")
    st.markdown("""
    ### 🎯 设计初衷
    本系统旨在让科研人员彻底告别繁琐的调参记录和服务器值守。您可以直接在此 Web 端修改参数并启动实验，或者使用手机飞书下达指令。
    
    ### 🛠️ 工作流指南
    1. **准备配置**: 在 `⚙️ 实验配置与执行` 面板选择一个 YAML 配置，例如 `exp_2d_demo.yaml`，按需修改并保存。
    2. **启动训练**: 点击【启动实验】，后台将自动加载模型并执行训练。
    3. **查看结果**: 切换到 `📊 实时战报看板`，随时刷新查看 Loss 曲线和 2D/3D 切片的预测可视化。
    4. **手机接管**: 您随时可以在 `base.yaml` 中配置飞书 Webhook，训练过程将同步推送至您的手机。

    ---
    *Powered by Streamlit & PyTorch.* 
    *GitHub 准备就绪: 代码结构已完全模块化，可直接 Push 开源！*
    """)

# ================= 页面 2: 实验配置与执行 =================
elif menu == "⚙️ 实验配置与执行":
    st.title("实验配置与调度中心")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. 选择并编辑配置")
        cfg_files = get_config_files()
        selected_cfg = st.selectbox("选择 YAML 配置文件", cfg_files)
        
        if selected_cfg:
            cfg_path = CONFIG_DIR / selected_cfg
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_content = f.read()
                
            new_content = st.text_area("编辑 YAML", cfg_content, height=350)
            if st.button("💾 保存配置"):
                with open(cfg_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                st.success(f"已保存: {selected_cfg}")

    with col2:
        st.subheader("2. 进程控制")
        st.info("💡 点击下方按钮，相当于在终端执行 `python scripts/train.py`")
        
        if st.session_state.process and st.session_state.process.poll() is None:
            st.warning(f"⚠️ 当前正在运行: {selected_cfg} (PID: {st.session_state.process.pid})")
            if st.button("⏹ 紧急停止实验", type="primary"):
                st.session_state.process.terminate()
                st.session_state.process = None
                st.success("已发送终止信号！")
        else:
            st.success("🟢 实验机当前处于空闲状态，随时可启动。")
            if st.button("▶ 启动实验"):
                if selected_cfg:
                    cfg_path = CONFIG_DIR / selected_cfg
                    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"), "--config", str(cfg_path)]
                    st.session_state.process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
                    st.rerun()

# ================= 页面 3: 实时战报看板 =================
elif menu == "📊 实时战报看板":
    st.title("实验监控与可视化")
    st.markdown("在实验运行过程中，可以点击刷新查看最新的指标曲线与预测切片。")
    
    if st.button("🔄 刷新图像"):
        st.rerun()
        
    col1, col2 = st.columns(2)
    
    def display_image(path_str, title, col):
        path = Path(path_str)
        if path.exists():
            col.subheader(title)
            img = Image.open(path)
            col.image(img, use_container_width=True)
        else:
            col.info(f"暂无生成图像，需等待验证阶段...\n(路径: {path_str})")

    # 尝试寻找生成的实验图像 (自动匹配最新的 demo 文件夹)
    # 因为 demo 生成的图片路径如 experiments/exp_2d_demo/...
    # 这里我们遍历所有实验目录下最新的图片
    
    latest_curve = None
    latest_slice = None
    
    if EXP_DIR.exists():
        # 寻找最近修改的 convergence_curve.png
        curves = list(EXP_DIR.rglob("convergence_curve.png"))
        if curves:
            latest_curve = max(curves, key=os.path.getmtime)
            
        # 寻找最近修改的 slice png
        slices = list(EXP_DIR.rglob("val_slice_epoch_*.png"))
        if not slices:
            # 兼容自检脚本生成的图片
            slices = list(EXP_DIR.rglob("test_slice_preview.png"))
        if slices:
            latest_slice = max(slices, key=os.path.getmtime)

    with col1:
        if latest_slice:
            display_image(latest_slice, "🔍 最新验证切片预测 (GT vs Pred)", col1)
        else:
            col1.info("暂未发现生成的切片图像，请先启动一次实验。")

    with col2:
        if latest_curve:
            display_image(latest_curve, "📈 训练收敛曲线", col2)
        else:
            col2.info("暂未发现生成的收敛曲线。")
