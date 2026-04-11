import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd

# --- 1. 核心路径配置 (适配云端相对路径) ---
MODEL_PATH = "best.pt" 
# 性能评估图表需放在 GitHub 根目录下
RES_PATH = "results.png"
CM_PATH = "confusion_matrix.png"

# 初始化历史记录存储
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 2. 网页 UI 样式美化 (自定义 CSS) ---
st.set_page_config(page_title="水稻病害检测平台", layout="wide", page_icon="🌾")

st.markdown("""
    <style>
    /* 全局背景与字体 */
    .main { background-color: #f9fbf9; font-family: 'Microsoft YaHei', sans-serif; }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] { background-color: #f0f7f0; border-right: 2px solid #2e7d32; }
    
    /* 隐藏 Streamlit 默认页眉页脚 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 标题样式 */
    .main-title { color: #1b5e20; text-align: center; font-weight: 800; margin-bottom: 30px; }
    
    /* 诊断处方卡片美化 */
    .diagnosis-card { 
        padding: 20px; 
        border-radius: 15px; 
        background-color: white; 
        border-left: 8px solid #2e7d32; 
        box-shadow: 3px 3px 12px rgba(0,0,0,0.08); 
        margin-bottom: 20px;
    }
    .card-header { color: #d32f2f; font-size: 1.4em; font-weight: bold; margin-bottom: 10px; }
    
    /* 按钮美化 */
    .stButton>button { 
        width: 100%; border-radius: 10px; height: 3.2em; 
        background-color: #2e7d32; color: white; font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #1b5e20; transform: translateY(-2px); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 专家级农艺知识百科库 (完全匹配模型标签) ---
DISEASE_WIKI = {
    "DISEASE- Bacterial Leaf Blight": {
        "name": "水稻白叶枯病",
        "desc": "叶片边缘出现黄白色长条斑，边缘呈波浪状。湿度大时有黄色菌脓，严重时全叶枯萎。",
        "advice": "1. 选用抗病品种；2. 科学排灌，防止淹水；3. 喷施叶枯唑或农用链霉素。",
        "url": "https://baike.baidu.com/item/水稻白叶枯病"
    },
    "NUTRIENT DEFFICIENT- Silicon": {
        "name": "水稻缺硅症",
        "desc": "叶片柔软下垂，植株抗逆性变差，极易发生倒伏及感染病害。",
        "advice": "1. 施用硅化肥（如硅酸钙）；2. 喷施叶面硅肥增强叶片硬度。",
        "url": "https://baike.baidu.com/item/植物缺硅"
    },
    "DISEASE- Leaf Blast": {
        "name": "水稻稻瘟病",
        "desc": "典型病斑为梭形，中心灰白，边缘红褐。被称为‘水稻癌症’，对产量威胁极大。",
        "advice": "1. 避免偏施氮肥；2. 及时喷施三环唑、稻瘟灵或肟菌酯·戊唑醇。",
        "url": "https://baike.baidu.com/item/稻瘟病"
    },
    "DISEASE- Brown Spot": {
        "name": "水稻胡麻叶斑病",
        "desc": "叶片散生褐色芝麻粒状病斑，中心灰白色。多发于瘦瘠稻田。",
        "advice": "1. 增施钾肥和有机肥；2. 发病初使用苯醚甲环唑或百菌清防治。",
        "url": "https://baike.baidu.com/item/水稻胡麻叶斑病"
    },
    "HEALTHY": {
        "name": "健康植株",
        "desc": "叶色翠绿，无病斑或生理性缺素症状。",
        "advice": "目前生长良好。请保持科学肥水管理，定期巡查。",
        "url": "https://baike.baidu.com/item/水稻/133543"
    }
    # 更多类别（如缺氮、缺磷等）可按此格式继续扩充
}

# --- 4. 核心功能函数 ---
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

model = load_model()

def render_diagnosis(found_classes):
    """渲染诊断报告"""
    if not found_classes:
        st.info("💡 诊断结果：未检测到明显异常。")
        return
    st.markdown("<h3 style='color: #2e7d32;'>📋 专家诊断处方</h3>", unsafe_allow_html=True)
    for c in set(found_classes):
        # 模糊匹配逻辑
        info = DISEASE_WIKI.get(c)
        if not info:
            for k, v in DISEASE_WIKI.items():
                if c.lower() in k.lower() or k.lower() in c.lower():
                    info = v; break
        
        if not info:
            info = {"name": c, "desc": "暂无描述", "advice": "咨询农技人员", "url": "#"}
            
        st.markdown(f"""
            <div class="diagnosis-card">
                <div class="card-header">识别到：{info['name']}</div>
                <p><b>【病症特征】</b>：{info['desc']}</p>
                <p style="background-color: #f1f8e9; padding: 10px; border-radius: 5px;">
                    <b>【防治建议】</b>：{info['advice']}
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.link_button(f"🔗 查看 {info['name']} 详细百科手册", info['url'])

def add_record(mode, filename, found, res_img):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    names = [DISEASE_WIKI.get(c, {"name":c})["name"] for c in set(found)]
    thumb = res_img.copy(); thumb.thumbnail((180, 180))
    st.session_state['history'].insert(0, {
        "时间": now, "模式": mode, "文件名": filename, 
        "结果": ", ".join(names) or "健康", "预览": thumb, "详情": res_img, "标签": list(set(found))
    })

# --- 5. 侧边栏与导航 ---
with st.sidebar:
    st.markdown("# 🌾 水稻病害检测")
    st.divider()
    option = st.radio("导航菜单", ["智能单图检测", "专业批量分析", "实时监控模式", "详细检测记录", "模型性能评估"])
    st.divider()
    conf_val = st.slider("置信度 (Conf)", 0.05, 1.0, 0.45)
    iou_val = st.slider("重叠过滤 (IoU)", 0.1, 0.9, 0.35)
    if st.button("🗑️ 清空历史"):
        st.session_state['history'] = []; st.rerun()

# 主页面标题
st.markdown("<h1 class='main-title'>🌾 水稻病害检测系统</h1>", unsafe_allow_html=True)

if not model:
    st.error("⚠️ 未能加载模型，请确保 best.pt 已上传。")
    st.stop()

# --- 6. 各模块逻辑 ---

if option == "智能单图检测":
    file = st.file_uploader("上传照片", type=['jpg','png','jpeg'])
    if file:
        img = Image.open(file)
        res = model.predict(img, conf=conf_val, iou=iou_val)[0]
        res_plotted = Image.fromarray(res.plot()[..., ::-1])
        c1, c2 = st.columns(2)
        with c1: st.image(img, caption="原始图片", use_container_width=True)
        with c2: st.image(res_plotted, caption="识别结果", use_container_width=True)
        found = [res.names[int(b.cls[0])] for b in res.boxes]
        add_record("单图", file.name, found, res_plotted)
        render_diagnosis(found)

elif option == "专业批量分析":
    files = st.file_uploader("多选图片", accept_multiple_files=True)
    if files and st.button("开始批量诊断"):
        for f in files:
            img = Image.open(f)
            res = model.predict(img, conf=conf_val)[0]
            res_plotted = Image.fromarray(res.plot()[..., ::-1])
            found = [res.names[int(b.cls[0])] for b in res.boxes]
            add_record("批量", f.name, found, res_plotted)
            with st.expander(f"🔍 结果预览: {f.name}"):
                ca, cb = st.columns([1, 2])
                with ca: st.image(res_plotted)
                with cb: render_diagnosis(found)

elif option == "实时监控模式":
    st.info("💡 提示：请允许浏览器访问摄像头。")
    cam_image = st.camera_input("拍照即刻识别")
    if cam_image:
        img = Image.open(cam_image)
        res = model.predict(img, conf=conf_val)[0]
        res_plotted = Image.fromarray(res.plot()[..., ::-1])
        st.image(res_plotted, caption="诊断快照")
        render_diagnosis([res.names[int(b.cls[0])] for b in res.boxes])

elif option == "详细检测记录":
    if not st.session_state['history']: st.warning("暂无记录")
    for i, rec in enumerate(st.session_state['history']):
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 4, 1])
            col1.image(rec["预览"])
            col2.markdown(f"📅 **时间**：{rec['时间']} | **文件名**：{rec['文件名']}\n\n**诊断结果**：`{rec['结果']}`")
            if col3.button("详情", key=f"btn_{i}"):
                st.image(rec["详情"])
                render_diagnosis(rec["标签"])

elif option == "模型性能评估":
    st.subheader("📊 模型训练关键指标曲线")
    if os.path.exists(RES_PATH):
        st.image(RES_PATH, caption="训练 Loss 与 mAP 指标曲线", use_container_width=True)
    else:
        st.error("未在根目录找到 results.png")
    if os.path.exists(CM_PATH):
        st.image(CM_PATH, caption="混淆矩阵", use_container_width=True)
