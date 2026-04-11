import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd

# --- 1. 核心路径配置 (适配云端相对路径) ---
# 请确保 best.pt, results.png, confusion_matrix.png 都在 GitHub 根目录下
MODEL_PATH = "best.pt" 
TRAIN_LOG_DIR = "." # 指向当前文件夹

# 初始化增强型历史记录
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 2. 农艺知识百科库 (保持不变) ---
DISEASE_WIKI = {
    "BACTERIAL LEAF BLIGHT": {"name": "白叶枯病", "desc": "叶缘黄白色条斑...", "advice": "喷施叶枯唑...", "url": "https://baike.baidu.com/item/水稻白叶枯病"},
    "BROWN SPOT": {"name": "褐斑病", "desc": "芝麻粒褐色斑点...", "advice": "喷施拿敌稳...", "url": "https://baike.baidu.com/item/水稻褐斑病"},
    "DEFICIENCY- NITROGEN": {"name": "缺氮症", "desc": "叶片发黄...", "advice": "追施尿素...", "url": "https://baike.baidu.com/item/植物缺氮"},
    "DEFICIENCY- POTASSIUM": {"name": "缺钾症", "desc": "红褐色焦枯...", "advice": "喷施磷酸二氢钾...", "url": "https://baike.baidu.com/item/植物缺钾"},
    "LEAF BLAST": {"name": "稻瘟病", "desc": "梭形斑...", "advice": "三环唑防治...", "url": "https://baike.baidu.com/item/稻瘟病"},
    "DEFICIENCY- NITROGEN MANGANESE POTASSIUM MAGNESIUM": {"name": "复合营养缺乏", "desc": "氮锰钾镁匮乏...", "advice": "施用三元复合肥...", "url": "https://baike.baidu.com/item/科学施肥"},
    "HEALTHY": {"name": "健康植株", "desc": "颜色翠绿...", "advice": "保持常规管理。", "url": "https://baike.baidu.com/item/水稻/133543"}
}

# --- 3. 核心功能组件 ---
@st.cache_resource
def load_yolo_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

model = load_yolo_model()

def add_record(mode, filename, found_classes, result_img):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cn_names = [DISEASE_WIKI.get(c, {"name": c})["name"] for c in set(found_classes)]
    res_text = ", ".join(cn_names) if cn_names else "健康/未识别"
    thumb = result_img.copy()
    thumb.thumbnail((180, 180))
    st.session_state['history'].insert(0, {
        "时间": now, "模式": mode, "文件名": filename, "结果": res_text,
        "预览": thumb, "详情图": result_img, "原始标签": list(set(found_classes))
    })

def render_diagnosis_report(found_classes):
    if not found_classes:
        st.info("💡 诊断结果：植株目前表现健康。")
        return
    st.subheader("📋 专家诊断处方")
    for c in set(found_classes):
        info = DISEASE_WIKI.get(c, {"name": c, "desc": "暂无描述", "advice": "咨询农技人员", "url": "#"})
        with st.container(border=True):
            st.markdown(f"### **{info['name']}**")
            st.write(f"**【病症特征】**：{info['desc']}")
            st.success(f"**【防治建议】**：{info['advice']}")
            st.link_button(f"🔗 查看详细百科", info['url'])

# --- 4. UI 界面与侧边栏 (更新标题) ---
st.set_page_config(page_title="水稻病害检测平台", layout="wide", page_icon="🌾")

# 侧边栏增加标题
with st.sidebar:
    st.title("🌾 水稻病害检测系统") # 修改左侧标题
    option = st.radio("导航菜单", ["智能单图检测", "专业批量分析", "实时监控模式", "历史数据中心", "模型性能评估"])
    st.divider()
    conf_val = st.slider("置信度 (Conf)", 0.05, 1.0, 0.45)
    iou_val = st.slider("重叠过滤 (IoU)", 0.1, 0.9, 0.35)
    if st.button("🗑️ 清空历史记录"):
        st.session_state['history'] = []
        st.rerun()

# 页面中间增加标题
st.title("🌾 水稻病害检测") # 修改中间标题

if model is None:
    st.error(f"⚠️ 无法加载模型！请确保 best.pt 已上传至 GitHub 根目录。")
    st.stop()

# --- 5. 功能逻辑 ---

if option == "智能单图检测":
    st.header("🖼️ 单张图像精细化诊断")
    file = st.file_uploader("上传叶片照片", type=['jpg','jpeg','png'])
    if file:
        img = Image.open(file)
        results = model(img, conf=conf_val, iou=iou_val, agnostic_nms=True)[0]
        res_img = Image.fromarray(results.plot()[..., ::-1])
        c1, c2 = st.columns(2)
        with c1: st.image(img, caption="输入图片", use_container_width=True)
        with c2: st.image(res_img, caption="AI 识别结果", use_container_width=True)
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        add_record("单图", file.name, found, res_img)
        render_diagnosis_report(found)

elif option == "专业批量分析":
    st.header("📂 批量自动分析")
    files = st.file_uploader("上传多张图片", accept_multiple_files=True)
    if files and st.button("开始任务"):
        bar = st.progress(0)
        for i, f in enumerate(files):
            img = Image.open(f)
            res = model(img, conf=conf_val, iou=iou_val, agnostic_nms=True)[0]
            res_img = Image.fromarray(res.plot()[..., ::-1])
            found = [res.names[int(b.cls[0])] for b in res.boxes]
            add_record("批量", f.name, found, res_img)
            with st.expander(f"结果：{f.name}"):
                ca, cb = st.columns([1, 2])
                with ca: st.image(res_img)
                with cb: render_diagnosis_report(found)
            bar.progress((i+1)/len(files))
        st.balloons()

elif option == "实时监控模式":
    st.header("📷 网页端实时监测")
    st.write("请点击下方按钮启动您设备（手机/电脑）的摄像头：")
    # 使用 st.camera_input 替代 cv2.VideoCapture，适配网页端
    cam_image = st.camera_input("拍照检测") 
    
    if cam_image:
        img = Image.open(cam_image)
        results = model(img, conf=conf_val, iou=iou_val, agnostic_nms=True)[0]
        res_img = Image.fromarray(results.plot()[..., ::-1])
        st.image(res_img, caption="检测结果")
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        render_diagnosis_report(found)

elif option == "历史数据中心":
    st.header("📜 历史监测数据")
    if not st.session_state['history']:
        st.info("尚无检测记录")
    else:
        for i, rec in enumerate(st.session_state['history']):
            with st.container(border=True):
                L1, L2, L3, L4 = st.columns([1, 2, 3, 1])
                with L1: st.image(rec["预览"])
                with L2: st.write(f"📅 {rec['时间']}")
                with L3: st.success(f"结果: {rec['结果']}")
                with L4:
                    if st.button("查看详情", key=f"det_{i}"):
                        st.image(rec["详情图"])
                        render_diagnosis_report(rec["原始标签"])

elif option == "模型性能评估":
    st.header("📈 模型训练关键指标")
    # 云端环境下检测当前目录的文件
    res_img_path = "results.png"
    cm_path = "confusion_matrix.png"
    
    col_res, col_chart = st.columns([3, 2])
    with col_res:
        if os.path.exists(res_img_path):
            st.image(res_img_path, caption="Loss & mAP 训练曲线", use_container_width=True)
        else:
            st.error("⚠️ GitHub 仓库中未找到 results.png 文件。请将其上传到根目录。")
            
    with col_chart:
        st.markdown("### 📊 指标说明")
        st.write("- **Box Loss**: 定位误差，越低代表框画得越准。")
        st.write("- **mAP50**: 核心精度指标，代表识别的整体准确率。")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="混淆矩阵 (分类精度对比)")
        else:
            st.warning("⚠️ GitHub 根目录下缺失 confusion_matrix.png。")
