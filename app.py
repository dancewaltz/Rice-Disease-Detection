import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd
import random
import string
import io

# --- 1. 核心路径与自适应配置 ---
MODEL_PATH = "best.pt" 
TRAIN_LOG_DIR = "." 

if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 2. 网页 UI 样式美化 ---
st.set_page_config(page_title="水稻智慧诊断平台", layout="wide", page_icon="🌾")

st.markdown("""
    <style>
    .main { background-color: #f9fbf9; }
    .main-title { color: #1b5e20; text-align: center; font-weight: 800; margin-bottom: 30px; }
    .diagnosis-card { 
        padding: 20px; border-radius: 15px; background-color: white; 
        border-left: 8px solid #2e7d32; box-shadow: 3px 3px 12px rgba(0,0,0,0.08); margin-bottom: 20px;
    }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.2em; background-color: #2e7d32; color: white; font-weight: bold; }
    #MainMenu, footer, header {visibility: hidden;}
    
    /* 核心布局：图片随窗口大小自适应缩放 */
    .stImage > img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 核心映射表 (严格对应最新 data.yaml 的 11 类缩写) ---
CLASS_NAMES_CN = {
    "BLBD": "白叶枯病",
    "BLSD": "叶尖枯病",
    "BSD": "褐斑病",
    "DPD": "东格鲁病(矮化病)",
    "FSD": "稻曲病",
    "Healthy": "健康植株",
    "NBD": "穗颈瘟",
    "NBSD": "窄褐斑病",
    "RBD": "稻瘟病",
    "RRSD": "草状矮化病",
    "SBD": "茎腐病"
}

# --- 4. 农艺知识百科库 (适配 11 类新标签) ---
DISEASE_WIKI = {
    "白叶枯病": {"desc": "叶片边缘出现黄白色条斑，边缘呈波浪状。湿度大时有菌脓。", "advice": "选用抗病品种；控制氮肥；发病初期喷施叶枯唑或农用链霉素。", "url": "https://baike.baidu.com/item/水稻白叶枯病"},
    "叶尖枯病": {"desc": "病斑从叶尖开始沿叶脉向下蔓延，呈现灰褐色或黄褐色枯死斑。", "advice": "及时排水晒田；喷施三环唑或多菌灵进行化学防治。", "url": "https://baike.baidu.com/item/水稻叶尖枯病"},
    "褐斑病": {"desc": "叶片出现芝麻粒大小褐色病斑，中心灰白色。常由高温高湿或缺钾引起。", "advice": "增施钾肥；种子消毒；喷施苯醚甲环唑或三环唑进行防治。", "url": "https://baike.baidu.com/item/水稻胡麻叶斑病"},
    "东格鲁病(矮化病)": {"desc": "病毒病。表现为植株严重矮缩，叶片呈橙黄色或黄色。", "advice": "核心是防治传毒叶蝉；及时拔除病株销毁，防止扩散。", "url": "https://baike.baidu.com/item/水稻东格鲁病"},
    "稻曲病": {"desc": "只发生于穗部，谷粒受害产生黑色粉末状菌球。", "advice": "破口前喷施井冈霉素或戊唑醇；调节施肥比例。", "url": "https://baike.baidu.com/item/水稻稻曲病"},
    "健康植株": {"desc": "叶片挺拔翠绿，无受害痕迹。体现了良好的田间肥水管理。", "advice": "继续保持当前科学管理方案，加强日常巡视预防。", "url": "https://baike.baidu.com/item/水稻/133543"},
    "穗颈瘟": {"desc": "发生在穗颈部，导致穗颈变黑枯死，形成白穗或半白穗。", "advice": "破口期是防治关键；喷施三环唑或春雷霉素进行预防。", "url": "https://baike.baidu.com/item/穗颈瘟"},
    "窄褐斑病": {"desc": "病斑短小、窄条状，多为红褐色。常在生长后期发生。", "advice": "改善田间通风透光；合理施肥，防止植株早衰。", "url": "https://baike.baidu.com/item/水稻窄褐斑病"},
    "稻瘟病": {"desc": "水稻主要病害，出现梭形病斑，中心灰白边缘红褐。", "advice": "加强肥水管理；破口期至齐穗期是化学防治的核心时段。", "url": "https://baike.baidu.com/item/稻瘟病"},
    "草状矮化病": {"desc": "植株极度矮化，分蘖极多，叶片狭窄且呈淡黄色。", "advice": "防治传毒害虫褐飞虱；病株深埋或烧毁。", "url": "https://baike.baidu.com/item/水稻草状矮化病"},
    "茎腐病": {"desc": "危害水稻茎基部，导致组织软腐并产生恶臭，引起倒伏。", "advice": "加强排灌管理；拔除病株；发病初期喷施井冈霉素。", "url": "https://baike.baidu.com/item/水稻茎腐病"}
}

# --- 5. 核心工具函数 ---

def generate_random_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def export_to_excel(history):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data = []
    for rec in history:
        main_cause = rec["结果"].split(", ")[0]
        advice = DISEASE_WIKI.get(main_cause, {}).get("advice", "咨询农技人员")
        data.append({
            "随机编号": rec["随机编号"], "检测时间": rec["时间"], "诊断结论": rec["结果"], "专家建议": advice
        })
    df = pd.DataFrame(data)
    df.to_excel(writer, index=False, sheet_name='检测记录')
    workbook = writer.book
    worksheet = writer.sheets['检测记录']
    worksheet.set_column('A:D', 20)
    worksheet.set_column('E:E', 35) 
    for i, rec in enumerate(history):
        img_data = io.BytesIO()
        rec["图"].save(img_data, format='PNG')
        worksheet.insert_image(i + 1, 4, f"img_{i}.png", {'image_data': img_data, 'x_scale': 0.12, 'y_scale': 0.12})
        worksheet.set_row(i + 1, 75) 
    writer.close()
    return output.getvalue()

@st.cache_resource
def load_yolo():
    if os.path.exists(MODEL_PATH): return YOLO(MODEL_PATH)
    return None

model = load_yolo()

def add_record(mode, filename, found_classes, result_img):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cn_names = [CLASS_NAMES_CN.get(c, c) for c in set(found_classes)]
    res_text = ", ".join(cn_names) if cn_names else "健康"
    st.session_state['history'].insert(0, {
        "随机编号": generate_random_id(), "时间": now, "结果": res_text, "图": result_img, "原始": found_classes
    })

def show_report(found_classes):
    if not found_classes:
        st.info("💡 诊断结果：目前检测结果显示植株生长状态良好。")
        return
    st.divider()
    for eng_label in set(found_classes):
        cn_key = CLASS_NAMES_CN.get(eng_label, eng_label)
        info = DISEASE_WIKI.get(cn_key, {"desc": "暂无该细分标签描述", "advice": "咨询农技员", "url": "#"})
        with st.container(border=True):
            st.markdown(f"### 💊 {cn_key}")
            st.write(f"**【病症特征】**：{info['desc']}")
            st.success(f"**【防治建议】**：{info['advice']}")
            if info['url'] != "#": st.link_button(f"🔗 查看 {cn_key} 百科", info['url'])

# --- 6. 主界面实现 ---
st.title("🌾 水稻病害检测识别系统")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 智能单图检测", "📂 专业批量分析", "🤳 手机拍照识别", "📜 历史数据中心", "📈 模型性能评估"])

with st.expander("🛠️ 高级算法参数设置"):
    col_v1, col_v2 = st.columns(2)
    with col_v1: conf_val = st.slider("置信度 (Conf)", 0.05, 1.0, 0.45)
    with col_v2: iou_val = st.slider("重叠过滤 (IoU)", 0.1, 0.9, 0.35)

# 模块 1：单图检测
with tab1:
    file = st.file_uploader("上传待检测照片...", type=['jpg','jpeg','png'], key="single_u")
    if file:
        img = Image.open(file)
        results = model(img, conf=conf_val, iou=iou_val)[0]
        res_plot = Image.fromarray(results.plot()[..., ::-1])
        _, mid1, mid2, _ = st.columns([0.5, 2, 2, 0.5])
        with mid1: st.image(img, caption="原始输入", use_container_width=True)
        with mid2: st.image(res_plot, caption="AI 诊断结果", use_container_width=True)
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        add_record("单图", file.name, found, res_plot)
        show_report(found)

# 模块 2：批量分析
with tab2:
    files = st.file_uploader("批量上传照片...", accept_multiple_files=True, key="multi_u")
    if files and st.button("🚀 执行批量推理"):
        for f in files:
            img = Image.open(f)
            res = model(img, conf=conf_val, iou=iou_val)[0]
            res_plot = Image.fromarray(res.plot()[..., ::-1])
            found = [res.names[int(b.cls[0])] for b in res.boxes]
            add_record("批量", f.name, found, res_plot)
            with st.expander(f"诊断详情：{f.name}"):
                b_c1, b_c2 = st.columns(2)
                with b_c1: st.image(img, caption="原图", use_container_width=True)
                with b_c2: st.image(res_plot, caption="结果图", use_container_width=True)
                show_report(found)

# 模块 3：拍照识别、历史、性能指标省略部分保持原逻辑...
# 模块 3：拍照识别
with tab3:
    st.info("💡 提示：本模块支持移动端直接调用摄像头进行田间巡检。")
    img_file = st.camera_input("请正对受害部位进行拍摄...")
    if img_file:
        img = Image.open(img_file)
        results = model(img, conf=conf_val, iou=iou_val)[0]
        res_plot = Image.fromarray(results.plot()[..., ::-1])
        st.image(res_plot, caption="实时抓拍结果", use_container_width=True)
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        add_record("拍照", "Capture.jpg", found, res_plot)
        show_report(found)

# 模块 4：历史中心
with tab4:
    if not st.session_state['history']:
        st.info("检测队列为空。")
    else:
        xlsx_data = export_to_excel(st.session_state['history'])
        st.download_button("📊 导出完整检测报告 (Excel 表格)", xlsx_data, "Rice_Report.xlsx", "application/vnd.ms-excel")
        st.markdown("---")
        for i, rec in enumerate(st.session_state['history']):
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 2, 1])
                with c1: st.image(rec["图"], use_container_width=True)
                with c2: 
                    st.write(f"🆔 **ID**: {rec['随机编号']}\n\n📅 **时间**: {rec['时间']}\n\n🔬 **结果**: {rec['结果']}")
                with c3:
                    if st.button("调取建议", key=f"hist_{i}"): show_report(rec["原始"])

# 模块 5：性能指标
with tab5:
    if os.path.exists(TRAIN_LOG_DIR):
        m1, m2 = st.columns(2)
        with m1:
            if os.path.exists("results.png"): st.image("results.png", caption="训练性能曲线", use_container_width=True)
        with m2:
            if os.path.exists("confusion_matrix.png"): st.image("confusion_matrix.png", caption="混淆矩阵", use_container_width=True)
