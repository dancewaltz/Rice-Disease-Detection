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
MODEL_PATH = "best.pt"  # 必须使用相对路径，确保模型和 app.py 在同一目录
TRAIN_LOG_DIR = "logs"  # 防止云端因无法读取 E 盘路径而崩溃

# 初始化增强型历史记录
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 2. UI 页面全局配置与 CSS 网页美化 ---
st.set_page_config(page_title="水稻智慧诊断平台", layout="wide", page_icon="🌾")

# 定义自定义 CSS 样式（内嵌网页样式表）
ST_CSS = """
<style>
    /* 隐藏 Streamlit 自带的页眉、页脚和菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 网页背景与全局字体 */
    .main { background-color: #f8fafc; }
    
    /* 大标题样式 */
    h1 { 
        color: #166534; /* 深绿色 */
        font-family: 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif; 
        font-weight: 800; 
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1); 
    }
    
    /* 副标题与通用 h3 样式 */
    h2, h3 { color: #15803d; }
    h3 { border-bottom: 2px solid #a7f3d0; padding-bottom: 8px; margin-top: 25px; }

    /* 美化所有按钮（Button） */
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3.5em; 
        background-color: #166534; /* 主色调 */
        color: white; 
        font-weight: bold; 
        border: none; 
        transition: 0.3s; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover { 
        background-color: #15803d; /* 悬停颜色 */
        border: none; 
        transform: translateY(-2px); /* 悬停上移 */
    }

    /* 美化专家诊断处方卡片（Card） */
    .diagnosis-card { 
        border: 1px solid #e2e8f0; 
        border-radius: 12px; 
        padding: 20px; 
        background-color: white; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); 
        margin-bottom: 20px; 
    }
    .diagnosis-name { color: #e11d48; font-weight: bold; font-size: 1.3em; }

    /* 美化侧边栏 */
    [data-testid="stSidebar"] { background-color: #ecfdf5; border-right: 1px solid #a7f3d0;}
    [data-testid="stSidebar"] h1 { font-size: 1.5em; color: #166534; }
    
    /* 美化折叠栏 (Expander) */
    .stExpander { border: 1px solid #e2e8f0; border-radius: 8px; background-color: white; margin-bottom: 10px; }
</style>
"""
# 将 CSS 注入网页
st.markdown(ST_CSS, unsafe_allow_html=True)

# --- 3. 农艺知识百科库 (13类完整大写匹配) ---
DISEASE_WIKI = {
    "BACTERIAL LEAF BLIGHT": {
        "name": "白叶枯病",
        "desc": "主要危害叶片，产生黄白色长条斑，边缘波浪状，严重时全叶枯死。",
        "advice": "选用抗病品种；发病初期喷施叶枯唑或农用链霉素；加强排灌管理。",
        "url": "https://baike.baidu.com/item/水稻白叶枯病"
    },
    "BROWN SPOT": {
        "name": "褐斑病",
        "desc": "叶片出现褐色小点，中心灰白色，边缘褐色，形似芝麻粒。",
        "advice": "增施钾肥提高抗性；发病初期喷施 75% 肟菌·戊唑醇（拿敌稳）或苯醚甲环唑。",
        "url": "https://baike.baidu.com/item/水稻褐斑病"
    },
    "DEFICIENCY- NITROGEN": {
        "name": "缺氮症",
        "desc": "植株矮小，老叶先发黄，根系发育不良，分蘖显著减少。",
        "advice": "及时追施速效氮肥（如尿素）；改善灌溉条件促进肥料吸收。",
        "url": "https://baike.baidu.com/item/植物缺氮"
    },
    "LEAF BLAST": {
        "name": "稻瘟病",
        "desc": "被称为“水稻癌症”，病斑呈梭形，中心灰白色，边缘红褐色，有褐色“坏死线”。背面有灰绿色霉层。",
        "advice": "避免过量施用氮肥；发病期使用三环唑、稻瘟灵或肟菌酯进行药剂防治。",
        "url": "https://baike.baidu.com/item/稻瘟病"
    },
    "HEALTHY": {
        "name": "健康植株",
        "desc": "叶色翠绿，生长旺盛，无明显病斑或虫害迹象。",
        "advice": "目前状态良好，请继续保持正常的肥水管理与田间巡查。",
        "url": "https://baike.baidu.com/item/水稻/133543"
    },
    # 其他营养缺乏类别（如 DEFICIENCY- PHOSPHORUS 等）请按此格式补全外部链接
}

# --- 4. 核心功能组件与推理修复 ---
@st.cache_resource
def load_yolo_model():
    if os.path.exists(MODEL_PATH):
        # 云端 CPU 环境加载优化
        return YOLO(MODEL_PATH)
    return None

model = load_yolo_model()

def add_record(mode, filename, found_classes, result_img):
    """保存记录，包括缩略图预览"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 转换中文名称
    cn_names = [DISEASE_WIKI.get(c, {"name": c})["name"] for c in set(found_classes)]
    res_text = ", ".join(cn_names) if cn_names else "健康/未识别"
    
    # 将结果图转为缩略图存入内存
    thumb = result_img.copy()
    thumb.thumbnail((180, 180)) 
    
    st.session_state['history'].insert(0, {
        "时间": now, "模式": mode, "文件名": filename, "检测结果": res_text,
        "预览图": thumb, "原始结果": result_img, "类别列表": list(set(found_classes))
    })

def render_diagnosis_report(found_classes):
    """渲染精细化诊断报告组件"""
    if not found_classes:
        st.info("💡 诊断结果：植株目前表现健康。请继续保持良好的肥水管理。")
        return

    st.markdown("<h3>📋 专家诊断处方</h3>", unsafe_allow_html=True)
    for c in set(found_classes):
        info = DISEASE_WIKI.get(c, {
            "name": c, 
            "desc": "暂无详细描述", 
            "advice": "请联系农技站获取支持",
            "url": "https://gs.jurieo.com/gemini/official"
        })
        # 使用自定义卡片布局渲染报告
        card_html = f"""
        <div class="diagnosis-card">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 2.5em; margin-right: 15px;">💊</span>
                <div>
                    <span class="diagnosis-name">{info['name']}</span>
                    <div style="color: #64748b; font-size: 0.9em; margin-top: 3px;">{c}</div>
                </div>
            </div>
            <p><strong>【病症描述】</strong>：{info['desc']}</p>
            <p style="color: #166534; background-color: #f0fdf4; padding: 10px; border-radius: 8px;"><strong>【防治建议】**：{info['advice']}</p>
            <a href="{info['url']}" target="_blank" style="text-decoration: none;">
                <button style="width: auto; padding: 8px 15px; background-color: white; color: #166534; border: 1px solid #166534; border-radius: 6px; cursor: pointer; margin-top: 10px;">
                    🔗 查看详细百科详情
                </button>
            </a>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

# --- 5. 功能模块导航栏 ---
with st.sidebar:
    st.markdown("<h1>🌾 智慧诊断导航</h1>", unsafe_allow_html=True)
    option = st.radio("功能模块切换", ["智能单图检测", "专业批量分析", "实时监控模式", "详细检测记录", "模型性能评估"])
    st.divider()
    st.markdown("### ⚙️ 推理参数微调")
    conf_threshold = st.slider("置信度阈值 (Conf)", 0.05, 1.0, 0.45)
    iou_threshold = st.slider("重叠过滤 (IoU)", 0.1, 0.9, 0.35)
    
    if st.button("🗑️ 清空所有记录"):
        st.session_state['history'] = []
        st.rerun()

# 美化版主页标题
t1, t2 = st.columns([1, 10])
with t1:
    st.markdown("<h1 style='text-align: center; font-size: 4em;'>🌾</h1>", unsafe_allow_html=True)
with t2:
    st.markdown("<h1>水稻生理性病害与营养缺乏智慧诊断平台 V3.5</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1em;'>基于 YOLOv11n 的农学专家系统。云端 CPU 实时运行，为农业生产提供科学的诊断处方。</p>", unsafe_allow_html=True)

# --- 6. 模块业务逻辑 ---

# 模块 A：单图检测
if option == "智能单图检测":
    st.header("🖼️ 单张图像精细化诊断")
    file = st.file_uploader("选择上传一张叶片照片", type=['jpg','png','jpeg'])
    if file:
        img = Image.open(file)
        # --- 修复核心：改用 predict 方法以获得更稳定的 Results 对象 ---
        pred_list = model.predict(img, conf=conf_threshold, iou=iou_threshold, agnostic_nms=True)
        # 安全地获取 Results 对象（处理不同的 ultralytics 版本返回 tuple 的情况）
        results = pred_list[0] if isinstance(pred_list, list) else pred_list
        
        # 结果绘制与处理
        res_plot = results.plot()
        # 将BGR转RGB
        res_plotted = Image.fromarray(res_plot[..., ::-1])
        
        c1, c2 = st.columns(2)
        with c1: st.image(img, caption="原始图片", use_container_width=True)
        with c2: st.image(res_plotted, caption="识别结果", use_container_width=True)
        
        # 获取类别标签
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        add_record("单图", file.name, found, res_plotted)
        render_diagnosis_report(found)

# 模块 B：批量分析（升级：支持独立展开详细诊断）
elif option == "专业批量分析":
    st.header("📂 批量自动化处理模式")
    files = st.file_uploader("支持上传多个文件", accept_multiple_files=True)
    if files and st.button("⚡ 开始批量任务"):
        bar = st.progress(0)
        for i, f in enumerate(files):
            img = Image.open(f)
            # --- 修复核心：同样改用 predict 方法 ---
            pred_list = model.predict(img, conf=conf_threshold, iou=iou_threshold, agnostic_nms=True)
            res = pred_list[0] if isinstance(pred_list, list) else pred_list
            
            res_plotted = Image.fromarray(res.plot()[..., ::-1])
            found = [res.names[int(b.cls[0])] for b in res.boxes]
            add_record("批量", f.name, found, res_plotted)
            
            with st.expander(f"图片: {f.name} --- 【{'🟡 有病害' if found else '🟢 健康'}】"):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.image(res_plotted)
                with col_b:
                    # 批量模式也接入精细化报告
                    render_diagnosis_report(found)
            bar.progress((i+1)/len(files))
        st.balloons()

# 其他模块（历史记录、性能评估等）保持原有逻辑...沿用之前的完整版即可。
