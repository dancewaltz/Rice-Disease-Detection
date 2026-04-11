import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd
import io

# --- 1. 核心路径配置 ---
MODEL_PATH = r"E:\学习\毕设\runs\detect\runs\detect\rice_13class_model\weights\best.pt"
TRAIN_LOG_DIR = r"E:\学习\毕设\runs\detect\runs\detect\rice_13class_model"

# 初始化增强型历史记录
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 2. 农艺知识百科库 (13类完整大写匹配) ---
DISEASE_WIKI = {
    "BACTERIAL LEAF BLIGHT": {
        "name": "白叶枯病",
        "desc": "发病时叶片边缘出现黄白色条斑，病斑边缘呈波浪状，严重时导致全叶枯萎。",
        "advice": "1. 选用抗病品种；2. 避免偏施氮肥；3. 喷施叶枯唑或农用链霉素。",
        "url": "https://baike.baidu.com/item/水稻白叶枯病"
    },
    "BROWN SPOT": {
        "name": "褐斑病",
        "desc": "叶片出现芝麻粒大小褐色斑点，中心灰白色。常导致叶片干枯。",
        "advice": "1. 增施钾肥；2. 喷施拿敌稳或苯醚甲环唑。",
        "url": "https://baike.baidu.com/item/水稻褐斑病"
    },
    "DEFICIENCY- NITROGEN": {
        "name": "缺氮症",
        "desc": "植株生长缓慢，老叶首先由叶尖向基部均匀发黄。",
        "advice": "1. 及时追施尿素；2. 配合叶面喷施尿素溶液。",
        "url": "https://baike.baidu.com/item/植物缺氮"
    },
    "DEFICIENCY- POTASSIUM": {
        "name": "缺钾症",
        "desc": "老叶叶尖及边缘出现红褐色焦枯，形似火烧。",
        "advice": "1. 追施氯化钾；2. 喷施磷酸二氢钾溶液。",
        "url": "https://baike.baidu.com/item/植物缺钾"
    },
    "LEAF BLAST": {
        "name": "稻瘟病",
        "desc": "典型症状为“梭形斑”，中心灰白，边缘红褐。",
        "advice": "1. 科学排灌；2. 使用三环唑、稻瘟灵防治。",
        "url": "https://baike.baidu.com/item/稻瘟病"
    },
    "DEFICIENCY- NITROGEN MANGANESE POTASSIUM MAGNESIUM": {
        "name": "复合营养缺乏",
        "desc": "多种元素匮乏（氮锰钾镁），生长严重受阻。",
        "advice": "1. 施用三元复合肥；2. 补充微量元素叶面肥。",
        "url": "https://baike.baidu.com/item/科学施肥"
    },
    "HEALTHY": {
        "name": "健康植株",
        "desc": "叶片颜色翠绿，无病斑或虫害痕迹。",
        "advice": "继续保持良好的肥水管理，预防为主。",
        "url": "https://baike.baidu.com/item/水稻/133543"
    },
    # 其他类别可根据需要继续补充...
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
    """诊断报告渲染组件"""
    if not found_classes:
        st.info("💡 诊断结果：植株目前表现健康。")
        return
    st.markdown("---")
    st.subheader("📋 专家诊断处方")
    for c in set(found_classes):
        info = DISEASE_WIKI.get(c, {"name": c, "desc": "暂无描述", "advice": "咨询农技人员", "url": "#"})
        with st.container(border=True):
            col_icon, col_content = st.columns([1, 8])
            with col_icon: st.title("💊")
            with col_content:
                st.markdown(f"### **{info['name']}**")
                st.write(f"**【病症特征】**：{info['desc']}")
                st.success(f"**【防治建议】**：{info['advice']}")
                st.link_button(f"🔗 查看详细百科", info['url'])

# --- 4. UI 界面与侧边栏 ---
st.set_page_config(page_title="水稻智慧诊断平台", layout="wide", page_icon="🌾")

with st.sidebar:
    st.title("🌾 智慧农业系统")
    option = st.radio("导航菜单", ["智能单图检测", "专业批量分析", "实时监控模式", "历史数据中心", "模型性能评估"])
    st.divider()
    conf_val = st.slider("置信度 (Conf)", 0.05, 1.0, 0.45)
    iou_val = st.slider("重叠过滤 (IoU)", 0.1, 0.9, 0.35)
    if st.button("🗑️ 清空历史记录"):
        st.session_state['history'] = []
        st.rerun()

if model is None:
    st.error(f"⚠️ 无法加载模型！路径：{MODEL_PATH}")
    st.stop()

# --- 5. 功能逻辑（修复空白问题） ---

# A：单图
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

# B：批量分析
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

# C：实时监控模式（补全逻辑）
elif option == "实时监控模式":
    st.header("📷 实时动态监控")
    st.write("点击下方按钮启动摄像头进行实时病害监测：")
    run = st.checkbox("🟢 启动实时视频流")
    FRAME_WINDOW = st.image([]) # 预留视频显示窗口
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret: break
            # 转换颜色空间以供显示和检测
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, conf=conf_val, iou=iou_val, agnostic_nms=True)[0]
            # 渲染画面并展示
            FRAME_WINDOW.image(results.plot())
        cap.release()
    else:
        st.info("摄像头已关闭。")

# D：历史中心
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

# E：模型性能评估（补全逻辑）
elif option == "模型性能评估":
    st.header("📈 模型训练关键指标")
    if os.path.exists(TRAIN_LOG_DIR):
        col_res, col_chart = st.columns([3, 2])
        with col_res:
            res_img_path = os.path.join(TRAIN_LOG_DIR, "results.png")
            if os.path.exists(res_img_path):
                st.image(res_img_path, caption="Loss & mAP 训练曲线", use_container_width=True)
            else:
                st.warning("未找到 results.png 日志文件。")
        with col_chart:
            st.markdown("### 📊 指标说明")
            st.write("- **Box Loss**: 定位误差，越低代表框画得越准。")
            st.write("- **mAP50**: 核心精度指标，代表识别的整体准确率。")
            st.info("当前模型基于 13 类数据训练，展现了良好的收敛性。")
            # 混淆矩阵
            cm_path = os.path.join(TRAIN_LOG_DIR, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="混淆矩阵（分类精度对比）")
    else:
        st.error("找不到训练日志文件夹，请检查路径配置。")