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
from fpdf import FPDF # 需要安装 fpdf2

# --- 1. 核心路径与自适应配置 ---
MODEL_PATH = "best.pt" 
TRAIN_LOG_DIR = "." 

if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 2. 网页 UI 样式美化 ---
st.set_page_config(page_title="水稻病害检测平台", layout="wide", page_icon="🌾")

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
    
    /* 核心修改：确保图片容器随窗口自适应 */
    .stImage > img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 核心映射表 ---
CLASS_NAMES_CN = {
    "BACTERIAL LEAF BLIGHT": "白叶枯病",
    "BROWN SPOT": "胡麻叶斑病",
    "DEFICIENCY- MAGNESIUM": "缺镁症",
    "DEFICIENCY- NITROGEN": "缺氮症",
    "DEFICIENCY- NITROGEN MANGANESE POTASSIUM MAGNESIUM and ZINC": "复合缺素(氮锰钾镁锌)",
    "DISEASE-  Narrow Brown Spot NUTRIENT DEFFICIENT- Nitrogen -N- Potassium -K- Calcium -Ca-": "窄褐斑病伴随复合缺素",
    "DISEASE- Bacterial Leaf Blight NUTRIENT DEFFICIENT- Silicon": "白叶枯病伴随缺硅",
    "DISEASE- Hispa NUTRIENT DEFFICENCY- N-A - Integrated pest management practices-": "铁甲虫害与综合治理",
    "DISEASE- Lead Scald NUTRIENT DEFFICIENT- Nitrogen -N- Potassium -K- Calcium -Ca- Sulfur -S-": "叶尖枯病伴随复合缺素",
    "HEALTHY": "健康植株",
    "HISPA": "水稻铁甲虫",
    "ISEASE- Leaf Blast NUTRIENT DEFFICIENT- Silicon- Nitrogen -N- Potassium -K- Potassium -K- Calcium -Ca-": "稻瘟病伴随复合缺素",
    "LEAFBLAST": "稻瘟病"
}

# --- 4. 农艺知识百科库 ---
DISEASE_WIKI = {
    "白叶枯病": {"desc": "叶片边缘出现黄白色条斑，边缘呈波浪状。", "advice": "选用抗病品种；控制氮肥；喷施叶枯唑。", "url": "https://baike.baidu.com/item/水稻白叶枯病"},
    "胡麻叶斑病": {"desc": "叶片出现芝麻粒大小褐色病斑，中心灰白色。", "advice": "增施钾肥；喷施苯醚甲环唑。", "url": "https://baike.baidu.com/item/水稻胡麻叶斑病"},
    "缺镁症": {"desc": "老叶叶脉间失绿变黄，叶脉保持绿色。", "advice": "增施硫酸镁。", "url": "https://baike.baidu.com/item/水稻缺镁症"},
    "缺氮症": {"desc": "植株矮小，叶片由下而上均匀发黄。", "advice": "补施尿素。", "url": "https://baike.baidu.com/item/水稻缺氮症"},
    "复合缺素(氮锰钾镁锌)": {"desc": "多种微量元素同时匮乏，叶片多重黄化。", "advice": "追施高品质三元复合肥。", "url": "https://baike.baidu.com/item/科学施肥"},
    "健康植株": {"desc": "叶片挺拔翠绿，光合效率高。", "advice": "继续保持当前管理。", "url": "https://baike.baidu.com/item/水稻/133543"},
    # ... 其余类别保持原样即可
}

# --- 5. 导出功能函数 ---

def generate_random_id():
    """生成随机编号"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def export_to_pdf(history):
    """生成 PDF 报告"""
    pdf = FPDF()
    pdf.add_page()
    # 解决中文乱码需要加载字体，如果没有字体文件可暂用英文或提示用户
    pdf.set_font("Arial", size=12) 
    pdf.cell(200, 10, txt="Rice Disease Detection Report", ln=True, align='C')
    
    for rec in history:
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"ID: {rec['随机编号']} | Time: {rec['时间']}", ln=True)
        pdf.cell(200, 10, txt=f"Result: {rec['结果']}", ln=True)
        # 将 PIL 存为临时文件供 PDF 使用
        temp_img = f"temp_{rec['随机编号']}.png"
        rec["图"].save(temp_img)
        pdf.image(temp_img, x=10, w=100)
        os.remove(temp_img)
    return pdf.output()

def export_to_excel(history):
    """生成带图片的 Excel 报告"""
    output = io.BytesIO()
    # 使用 xlsxwriter 引擎以支持图片插入
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # 准备基础数据（不含图片对象）
    data = []
    for rec in history:
        # 获取第一种病因的建议
        first_cause = rec["结果"].split(", ")[0]
        advice = DISEASE_WIKI.get(first_cause, {}).get("advice", "咨询农技人员")
        data.append({
            "随机编号": rec["随机编号"],
            "检测时间": rec["时间"],
            "病因结果": rec["结果"],
            "防治建议": advice
        })
    
    df = pd.DataFrame(data)
    df.to_excel(writer, index=False, sheet_name='检测记录')
    
    # 插入图片
    workbook = writer.book
    worksheet = writer.sheets['检测记录']
    for i, rec in enumerate(history):
        img_data = io.BytesIO()
        rec["图"].save(img_data, format='PNG')
        # 在 E 列插入图片，设置缩放
        worksheet.insert_image(i + 1, 4, f"img_{i}.png", {'image_data': img_data, 'x_scale': 0.2, 'y_scale': 0.2})
        worksheet.set_row(i + 1, 80) # 设置行高以适应图片

    writer.close()
    return output.getvalue()

# --- 6. 工具逻辑 ---
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
        "随机编号": generate_random_id(),
        "时间": now, 
        "结果": res_text, 
        "图": result_img, 
        "原始": found_classes
    })

def show_report(found_classes):
    if not found_classes:
        st.info("💡 诊断结果：目前检测结果显示植株健康。")
        return
    st.divider()
    for eng_label in set(found_classes):
        cn_key = CLASS_NAMES_CN.get(eng_label, eng_label)
        info = DISEASE_WIKI.get(cn_key, {"desc": "暂无该细分标签描述", "advice": "咨询农技员", "url": "#"})
        with st.container(border=True):
            st.markdown(f"### 💊 {cn_key}")
            st.write(f"**【病症特征】**：{info['desc']}")
            st.success(f"**【防治建议】**：{info['advice']}")
            if info['url'] != "#":
                st.link_button("🔗 查看百科详情", info['url'])

# --- 7. UI 实现 ---
st.title("🌾 水稻病害检测识别系统")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 智能单图检测", "📂 专业批量分析", "🤳 手机拍照识别", "📜 历史数据中心", "📈 模型性能评估"])

with st.expander("🛠️ 高级算法参数设置"):
    col_a, col_b = st.columns(2)
    with col_a: conf_val = st.slider("识别置信度 (Conf)", 0.05, 1.0, 0.45)
    with col_b: iou_val = st.slider("重叠过滤 (IoU)", 0.1, 0.9, 0.35)

# 核心业务模块逻辑（单图、拍照等）中使用 use_container_width=True
with tab1:
    file = st.file_uploader("上传叶片照片", type=['jpg','jpeg','png'], key="single_u")
    if file:
        img = Image.open(file)
        results = model(img, conf=conf_val, iou=iou_val)[0]
        res_plot = Image.fromarray(results.plot()[..., ::-1])
        # 核心修改：使用 use_container_width=True 实现放大缩小自适应
        st.image(res_plot, caption="AI 诊断结果", use_container_width=True)
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        add_record("单图", file.name, found, res_plot)
        show_report(found)

# 模块 4：历史中心（增加下载功能）
with tab4:
    if not st.session_state['history']:
        st.info("尚无检测记录。")
    else:
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            # 导出 PDF 按钮
            pdf_data = export_to_pdf(st.session_state['history'])
            st.download_button(label="📥 下载 PDF 诊断报告", data=pdf_data, file_name="Rice_Report.pdf", mime="application/pdf")
        with col_dl2:
            # 导出 Excel 按钮
            xlsx_data = export_to_excel(st.session_state['history'])
            st.download_button(label="📊 下载 Excel 数据表", data=xlsx_data, file_name="Rice_Records.xlsx", mime="application/vnd.ms-excel")
        
        st.markdown("---")
        for i, rec in enumerate(st.session_state['history']):
            with st.container(border=True):
                ca, cb, cc = st.columns([1, 2, 1])
                with ca: st.image(rec["图"], use_container_width=True)
                with cb: 
                    st.write(f"🆔 **编号**: {rec['随机编号']}")
                    st.write(f"📅 **时间**: {rec['时间']}")
                    st.write(f"🔬 **结果**: {rec['结果']}")
                with cc:
                    if st.button("查看建议", key=f"hist_{i}"):
                        show_report(rec["原始"])

# 模块 3 和 5 同理，将 st.image 的参数改为 use_container_width=True 即可
