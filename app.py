import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd

# --- 1. 核心路径配置 ---
MODEL_PATH = "best.pt" 
RES_PATH = "results.png"
CM_PATH = "confusion_matrix.png"

# 初始化历史记录
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 2. 网页 UI 样式美化 ---
st.set_page_config(page_title="水稻病害检测平台", layout="wide", page_icon="🌾")

st.markdown("""
    <style>
    .main { background-color: #f9fbf9; }
    [data-testid="stSidebar"] { background-color: #f0f7f0; border-right: 2px solid #2e7d32; }
    #MainMenu, footer, header {visibility: hidden;}
    .main-title { color: #1b5e20; text-align: center; font-weight: 800; margin-bottom: 30px; }
    .diagnosis-card { 
        padding: 20px; border-radius: 15px; background-color: white; 
        border-left: 8px solid #2e7d32; box-shadow: 3px 3px 12px rgba(0,0,0,0.08); margin-bottom: 20px;
    }
    .card-header { color: #d32f2f; font-size: 1.4em; font-weight: bold; margin-bottom: 10px; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.2em; background-color: #2e7d32; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 增强版农艺知识百科库 (修复匹配问题) ---
# Key 只需要包含模型输出中的核心关键词即可
DISEASE_WIKI = {
    "Narrow Brown Spot": {
        "name": "窄条斑病",
        "desc": "叶片上出现与叶脉平行的深褐色窄长条斑。发病严重时叶片由顶端向下枯萎，影响灌浆。",
        "advice": "1. 选用抗病品种；2. 科学施肥，避免偏施氮肥；3. 发病初期喷施三环唑或丙环唑。",
        "url": "https://baike.baidu.com/item/%E6%B0%B4%E7%A8%BB%E7%AA%84%E6%9D%A1%E6%96%91%E7%97%85"
    },
    "Bacterial Leaf Blight": {
        "name": "白叶枯病",
        "desc": "叶片边缘出现黄白色长条斑，边缘呈波浪状。湿度大时有菌脓。",
        "advice": "1. 科学排灌，防止淹水；2. 喷施叶枯唑或农用链霉素。",
        "url": "https://baike.baidu.com/item/%E6%B0%B4%E7%A8%BB%E7%99%BD%E5%8F%B6%E6%9E%AF%E7%97%85"
    },
    "Brown Spot": {
        "name": "胡麻叶斑病",
        "desc": "叶片散生褐色芝麻粒状病斑，中心灰白色。多发于瘦瘠稻田。",
        "advice": "1. 增施钾肥和有机肥；2. 发病初期使用苯醚甲环唑。",
        "url": "https://baike.baidu.com/item/%E6%B0%B4%E7%A8%BB%E8%83%A1%E9%BA%BB%E5%8F%B6%E6%96%91%E7%97%85"
    },
    "Nitrogen": {
        "name": "缺氮症",
        "desc": "全株色泽变黄，生长迟缓，老叶由叶尖向基部均匀变黄干枯。",
        "advice": "及时追施尿素或碳铵，配合叶面喷施尿素溶液。",
        "url": "https://baike.baidu.com/item/%E6%A4%8D%E7%89%A9%E7%BC%BA%E6%B0%AE"
    },
    "Potassium": {
        "name": "缺钾症",
        "desc": "老叶叶尖及边缘焦枯，形似火烧，叶面常出现赤褐色斑点。",
        "advice": "增施氯化钾或硫酸钾，抢晴天喷施磷酸二氢钾。",
        "url": "https://baike.baidu.com/item/%E6%A4%8D%E7%89%A9%E7%BC%BA%E9%92%BE"
    },
    "Calcium": {
        "name": "缺钙症",
        "desc": "幼叶尖端卷曲发黄，严重时生长点枯死，根系短而多。",
        "advice": "施用石灰或钙镁磷肥，改善土壤酸碱度。",
        "url": "https://baike.baidu.com/item/%E6%A4%8D%E7%89%A9%E7%BC%BA%E9%92%99"
    },
    "Silicon": {
        "name": "缺硅症",
        "desc": "叶片柔软下垂，植株抗逆性变差，易倒伏并感染病害。",
        "advice": "施用硅化肥，喷施叶面硅肥增强叶片硬度。",
        "url": "https://baike.baidu.com/item/%E6%A4%8D%E7%89%A9%E7%BC%BA%E7%A1%85"
    },
    "Leaf Blast": {
        "name": "稻瘟病",
        "desc": "梭形病斑，中心灰白。被称为‘水稻癌症’，威胁极大。",
        "advice": "1. 避免偏施氮肥；2. 及时喷施三环唑或稻瘟灵。",
        "url": "https://baike.baidu.com/item/%E7%A8%BB%E7%91%9F%E7%97%85"
    },
    "HEALTHY": {
        "name": "健康植株",
        "desc": "叶色翠绿，无病斑或生理性缺素症状。",
        "advice": "目前生长良好。请保持常规肥水管理。",
        "url": "https://baike.baidu.com/item/%E6%B0%B4%E7%A8%BB/133543"
    }
}

# --- 4. 核心功能函数 ---
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

model = load_model()

def render_diagnosis(found_classes):
    """渲染诊断报告 (支持长标签子串匹配)"""
    if not found_classes:
        st.info("💡 诊断结果：未检测到明显异常。")
        return
    
    st.markdown("<h3 style='color: #2e7d32;'>📋 专家诊断处方</h3>", unsafe_allow_html=True)
    
    # 获取去重后的检测结果列表
    detected_list = list(set(found_classes))
    all_matched_info = []

    for c in detected_list:
        found_match = False
        # 遍历知识库，检查知识库的 Key 是否在模型输出的复杂标签中
        for key, info in DISEASE_WIKI.items():
            if key.lower() in c.lower():
                if info not in all_matched_info: # 避免重复展示
                    all_matched_info.append(info)
                found_match = True
        
        # 如果彻底没匹配上，显示原始标签
        if not found_match:
            all_matched_info.append({"name": c, "desc": "暂无该细分标签描述", "advice": "咨询农技人员", "url": "#"})

    # 渲染匹配到的所有信息卡片
    for info in all_matched_info:
        st.markdown(f"""
            <div class="diagnosis-card">
                <div class="card-header">识别到：{info['name']}</div>
                <p><b>【病症特征】</b>：{info['desc']}</p>
                <p style="background-color: #f1f8e9; padding: 10px; border-radius: 5px;">
                    <b>【防治建议】</b>：{info['advice']}
                </p>
            </div>
        """, unsafe_allow_html=True)
        if info['url'] != "#":
            st.link_button(f"🔗 查看 {info['name']} 详细百科手册", info['url'])

def add_record(mode, filename, found, res_img):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 历史记录简要名称
    res_names = []
    for c in set(found):
        for k, v in DISEASE_WIKI.items():
            if k.lower() in c.lower(): res_names.append(v['name'])
    
    display_res = ", ".join(list(set(res_names))) or "健康"
    thumb = res_img.copy(); thumb.thumbnail((180, 180))
    st.session_state['history'].insert(0, {
        "时间": now, "模式": mode, "文件名": filename, 
        "结果": display_res, "预览": thumb, "详情": res_img, "标签": list(set(found))
    })

# --- 5. 侧边栏与主标题 ---
with st.sidebar:
    st.markdown("# 🌾 水稻病害检测")
    st.divider()
    option = st.radio("导航菜单", ["智能单图检测", "专业批量分析", "实时监控模式", "详细检测记录", "模型性能评估"])
    st.divider()
    conf_val = st.slider("置信度 (Conf)", 0.05, 1.0, 0.45)
    iou_val = st.slider("重叠过滤 (IoU)", 0.1, 0.9, 0.35)
    if st.button("🗑️ 清空历史"):
        st.session_state['history'] = []; st.rerun()

st.markdown("<h1 class='main-title'>🌾 水稻病害检测系统</h1>", unsafe_allow_html=True)

if not model:
    st.error("⚠️ 未能加载模型。")
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
            col2.markdown(f"📅 **时间**：{rec['时间']} | **文件名**：{rec['文件名']}\n\n**结果**：`{rec['结果']}`")
            if col3.button("详情", key=f"btn_{i}"):
                st.image(rec["详情"])
                render_diagnosis(rec["标签"])

elif option == "模型性能评估":
    st.subheader("📊 模型训练关键指标曲线")
    if os.path.exists(RES_PATH): st.image(RES_PATH, use_container_width=True)
    if os.path.exists(CM_PATH): st.image(CM_PATH, use_container_width=True)
