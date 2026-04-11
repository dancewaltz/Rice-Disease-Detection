import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd

# --- 1. 核心路径配置 (确保 best.pt, results.png 等在 GitHub 根目录) ---
MODEL_PATH = "best.pt" 
TRAIN_LOG_DIR = "." 

# 初始化历史记录
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 2. 网页样式美化 (CSS 注入) ---
st.set_page_config(page_title="水稻病害检测平台", layout="wide", page_icon="🌾")

st.markdown("""
    <style>
    .main { background-color: #f9fbf9; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #2e7d32; color: white; }
    .diagnosis-card { padding: 20px; border-radius: 15px; background-color: white; border-left: 5px solid #2e7d32; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 导航与标题 (左侧和中间均加入“水稻病害检测”) ---
with st.sidebar:
    st.markdown("# 🌾 水稻病害检测") # 左侧导航标题
    st.divider()
    option = st.radio("功能切换", ["智能单图检测", "专业批量分析", "实时监控模式", "详细检测记录", "模型性能评估"])
    st.divider()
    conf_val = st.slider("置信度 (Conf)", 0.05, 1.0, 0.45)
    iou_val = st.slider("重叠过滤 (IoU)", 0.1, 0.9, 0.35)

# 页面中间标题
st.markdown("<h1 style='text-align: center; color: #1b5e20;'>🌾 水稻病害检测系统</h1>", unsafe_allow_html=True)

# --- 4. 知识库与核心逻辑 ---
DISEASE_WIKI = {
    "BACTERIAL LEAF BLIGHT": {"name": "白叶枯病", "desc": "边缘黄白色条斑...", "advice": "喷施叶枯唑...", "url": "https://baike.baidu.com/item/水稻白叶枯病"},
    "BROWN SPOT": {"name": "褐斑病", "desc": "芝麻粒褐色斑点...", "advice": "喷施拿敌稳...", "url": "https://baike.baidu.com/item/水稻褐斑病"},
    "HEALTHY": {"name": "健康植株", "desc": "颜色翠绿...", "advice": "保持常规管理。", "url": "https://baike.baidu.com/item/水稻/133543"}
}

@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

model = load_yolo_model()

def render_diagnosis(found_classes):
    """通用诊断报告渲染"""
    if not found_classes:
        st.info("💡 诊断结果：植株目前表现健康。")
        return
    st.markdown("### 📋 专家处方建议")
    for c in set(found_classes):
        info = DISEASE_WIKI.get(c, {"name": c, "desc": "暂无描述", "advice": "咨询农技人员", "url": "#"})
        st.markdown(f"""
            <div class="diagnosis-card">
                <h4>病症名称：{info['name']}</h4>
                <p><b>特征：</b>{info['desc']}</p>
                <p style='color: #2e7d32;'><b>建议：</b>{info['advice']}</p>
            </div><br>
        """, unsafe_allow_html=True)
        st.link_button(f"🔗 查看 {info['name']} 百科详情", info['url'])

def add_record(mode, filename, found_classes, result_img):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    res_text = ", ".join([DISEASE_WIKI.get(c, {"name":c})["name"] for c in set(found_classes)]) or "健康"
    thumb = result_img.copy()
    thumb.thumbnail((180, 180))
    st.session_state['history'].insert(0, {"时间": now, "模式": mode, "文件名": filename, "结果": res_text, "预览": thumb, "详情": result_img, "标签": list(set(found_classes))})

# --- 5. 模块逻辑 ---

if option == "智能单图检测":
    file = st.file_uploader("上传照片", type=['jpg','png','jpeg'])
    if file:
        img = Image.open(file)
        res = model.predict(img, conf=conf_val, iou=iou_val)[0]
        res_plotted = Image.fromarray(res.plot()[..., ::-1])
        c1, c2 = st.columns(2)
        with c1: st.image(img, caption="原始图片")
        with c2: st.image(res_plotted, caption="识别结果")
        found = [res.names[int(b.cls[0])] for b in res.boxes]
        add_record("单图", file.name, found, res_plotted)
        render_diagnosis(found)

elif option == "专业批量分析":
    files = st.file_uploader("多选上传", accept_multiple_files=True)
    if files and st.button("开始分析"):
        for f in files:
            img = Image.open(f)
            res = model.predict(img, conf=conf_val, iou=iou_val)[0]
            res_plotted = Image.fromarray(res.plot()[..., ::-1])
            found = [res.names[int(b.cls[0])] for b in res.boxes]
            add_record("批量", f.name, found, res_plotted)
            with st.expander(f"🔍 查看详情: {f.name}"):
                ca, cb = st.columns([1, 2])
                with ca: st.image(res_plotted)
                with cb: render_diagnosis(found)

elif option == "实时监控模式":
    st.info("💡 提示：请点击浏览器地址栏左侧的'锁头'图标，允许网页访问摄像头。")
    cam_image = st.camera_input("拍照实时检测")
    if cam_image:
        img = Image.open(cam_image)
        res = model.predict(img, conf=conf_val)[0]
        res_plotted = Image.fromarray(res.plot()[..., ::-1])
        st.image(res_plotted, caption="实时识别结果")
        render_diagnosis([res.names[int(b.cls[0])] for b in res.boxes])

elif option == "详细检测记录":
    if not st.session_state['history']: st.warning("暂无记录")
    for i, rec in enumerate(st.session_state['history']):
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 3, 1])
            col1.image(rec["预览"])
            col2.write(f"📅 {rec['时间']} | {rec['文件名']}\n\n**结果：{rec['结果']}**")
            if col3.button("详情", key=f"rec_{i}"):
                st.image(rec["详情"])
                render_diagnosis(rec["标签"])

elif option == "模型性能评估":
    st.subheader("📊 模型训练关键指标曲线")
    res_path = "results.png" # 确保该文件在GitHub根目录
    cm_path = "confusion_matrix.png"
    if os.path.exists(res_path):
        st.image(res_path, caption="训练 Loss 与 mAP 指标曲线", use_container_width=True)
    else:
        st.error("未在 GitHub 根目录找到 results.png，请上传该文件。")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="混淆矩阵", use_container_width=True)
