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
from fpdf import FPDF 

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
    
    /* 核心修改：确保图片随窗口自适应缩放 */
    .stImage > img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 标签映射表 (严格对应 data.yaml) ---
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

# --- 4. 农艺知识百科库 (完全匹配 data.yaml 的 13 类标签) ---
DISEASE_WIKI = {
    "白叶枯病": {
        "desc": "叶片边缘出现黄白色条斑，边缘呈波浪状。湿度大时有菌脓，严重时全叶枯焦。",
        "advice": "选用抗病品种；控制氮肥；发病初期喷施叶枯唑或农用链霉素。",
        "url": "https://baike.baidu.com/item/水稻白叶枯病"
    },
    "胡麻叶斑病": {
        "desc": "叶片出现芝麻粒大小褐色病斑，中心灰白色。常由高温高湿或土壤缺钾引起。",
        "advice": "增施钾肥，提高抗性；种子消毒；喷施苯醚甲环唑或三环唑进行防治。",
        "url": "https://baike.baidu.com/item/水稻胡麻叶斑病"
    },
    "缺镁症": {
        "desc": "老叶叶脉间失绿变黄，但叶脉保持绿色，叶片呈现明显的条纹状失绿。",
        "advice": "施用钙镁磷肥或硫酸镁；叶面喷施1%-2%的硫酸镁溶液。",
        "url": "https://baike.baidu.com/item/水稻缺镁症"
    },
    "缺氮症": {
        "desc": "植株矮小，分蘖减少。叶片由下而上均匀发黄，叶尖枯萎。",
        "advice": "及时补施尿素或碳铵；配合叶面喷施1%的尿素水溶液。",
        "url": "https://baike.baidu.com/item/水稻缺氮症"
    },
    "复合缺素(氮锰钾镁锌)": {
        "desc": "多种大中微量元素同时匮乏，生长严重受阻，表现为多重黄化及发育迟缓。",
        "advice": "追施高品质三元复合肥；喷施含锌、锰、镁的多元叶面肥进行综合补偿。",
        "url": "https://baike.baidu.com/item/科学施肥"
    },
    "窄褐斑病伴随复合缺素": {
        "desc": "病斑呈窄条状且颜色深，同时伴有氮、钾、钙等养分失衡，抗病能力极差。",
        "advice": "改善土壤通透性；喷施针对性杀菌剂并补充中微量元素叶面肥。",
        "url": "https://baike.baidu.com/item/水稻窄褐斑病"
    },
    "白叶枯病伴随缺硅": {
        "desc": "典型细菌性病害且因缺硅导致表皮坚硬度不足，病原菌极易侵入叶片。",
        "advice": "喷施农用链霉素防治病害；基施或喷施硅酸钠、硅化肥增强表皮硬度。",
        "url": "https://baike.baidu.com/item/水稻白叶枯病"
    },
    "铁甲虫害与综合治理": {
        "desc": "叶片可见由害虫啃食形成的白色条斑。需结合农业管理与药剂进行综合防治。",
        "advice": "清理田边杂草；受害严重时喷施杀螟丹或高效氰戊菊酯。",
        "url": "https://baike.baidu.com/item/水稻铁甲虫"
    },
    "叶尖枯病伴随复合缺素": {
        "desc": "叶尖出现枯死斑并向下蔓延，生理代谢紊乱，伴随多种中微量元素匮乏。",
        "advice": "喷施三环唑防治病害；全面追施均衡肥料补充氮钾钙硫。",
        "url": "https://baike.baidu.com/item/水稻叶枯病"
    },
    "健康植株": {
        "desc": "叶片挺拔翠绿，无受害痕迹。体现了良好的田间肥水管理状态。",
        "advice": "继续保持当前科学管理方案，加强日常巡视预防。",
        "url": "https://baike.baidu.com/item/水稻/133543"
    },
    "水稻铁甲虫": {
        "desc": "叶片可见白色长条状食痕，系成虫啃食叶肉所致。严重时叶片枯焦如火烧。",
        "advice": "受害达标时喷施高效氰戊菊酯；注意清理田间杂草以减少越冬虫源。",
        "url": "https://baike.baidu.com/item/水稻铁甲虫"
    },
    "稻瘟病伴随复合缺素": {
        "desc": "出现梭形病斑（稻瘟病），且因硅、氮、钾、钙不足导致植株防御力极低。",
        "advice": "喷施稻瘟灵或三环唑；重点补充硅肥与钾肥增强细胞壁厚度。",
        "url": "https://baike.baidu.com/item/稻瘟病"
    },
    "稻瘟病": {
        "desc": "典型的“梭形斑”，中心灰白边缘红褐。被称为水稻“癌症”，极易爆发。",
        "advice": "控制氮肥、增施磷钾肥；发病期及时喷施三环唑、稻瘟灵或吡唑醚菌酯。",
        "url": "https://baike.baidu.com/item/稻瘟病"
    }
}

# --- 5. 核心逻辑函数 ---

def generate_random_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def export_to_pdf(history):
    """生成 PDF (修复 Unicode 编码报错)"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12) 
    pdf.cell(200, 10, txt="Rice Disease Detection Report", ln=True, align='C')
    for rec in history:
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"ID: {rec['随机编号']} | Time: {rec['时间']}", ln=True)
        # 避免 PDF 内部中文报错，此处仅在 PDF 记录英文诊断名
        eng_label = ", ".join(rec["原始"]) if rec["原始"] else "Healthy"
        pdf.cell(200, 10, txt=f"Diagnosis: {eng_label}", ln=True)
        temp_img = f"temp_{rec['随机编号']}.png"
        rec["图"].save(temp_img)
        pdf.image(temp_img, x=10, w=80) 
        os.remove(temp_img)
    return pdf.output()

def export_to_excel(history):
    """生成带图片和建议的 Excel"""
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data = []
    for rec in history:
        main_cause = rec["结果"].split(", ")[0]
        advice = DISEASE_WIKI.get(main_cause, {}).get("advice", "咨询农技员")
        data.append({
            "随机编号": rec["随机编号"],
            "时间": rec["时间"],
            "诊断结果": rec["结果"],
            "防治建议": advice
        })
    df = pd.DataFrame(data)
    df.to_excel(writer, index=False, sheet_name='检测数据')
    workbook = writer.book
    worksheet = writer.sheets['检测数据']
    for i, rec in enumerate(history):
        img_data = io.BytesIO()
        rec["图"].save(img_data, format='PNG')
        worksheet.insert_image(i + 1, 4, f"img_{i}.png", {'image_data': img_data, 'x_scale': 0.15, 'y_scale': 0.15})
        worksheet.set_row(i + 1, 70) 
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
        "随机编号": generate_random_id(),
        "时间": now, "结果": res_text, "图": result_img, "原始": found_classes
    })

def show_report(found_classes):
    if not found_classes:
        st.info("💡 诊断结果：植株目前生长状态良好。")
        return
    st.divider()
    for eng_label in set(found_classes):
        cn_key = CLASS_NAMES_CN.get(eng_label, eng_label)
        info = DISEASE_WIKI.get(cn_key, {"desc": "暂无该细分标签描述", "advice": "咨询农技人员", "url": "#"})
        with st.container(border=True):
            st.markdown(f"### 💊 {cn_key}")
            st.write(f"**【病症特征】**：{info['desc']}")
            st.success(f"**【防治建议】**：{info['advice']}")
            if info['url'] != "#": st.link_button(f"🔗 查看 {cn_key} 详情", info['url'])

# --- 8. 主界面布局 ---
st.title("🌾 水稻病害检测识别系统")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 智能单图检测", "📂 专业批量分析", "🤳 手机拍照识别", "📜 历史数据中心", "📈 模型性能评估"])

with st.expander("🛠️ 高级算法参数设置"):
    col_v1, col_v2 = st.columns(2)
    with col_v1: conf_val = st.slider("识别置信度 (Conf)", 0.05, 1.0, 0.45)
    with col_v2: iou_val = st.slider("重叠过滤 (IoU)", 0.1, 0.9, 0.35)

# 模块 1：单图检测
with tab1:
    file = st.file_uploader("请选择一张照片...", type=['jpg','jpeg','png'], key="single_u")
    if file:
        img = Image.open(file)
        results = model(img, conf=conf_val, iou=iou_val)[0]
        res_plot = Image.fromarray(results.plot()[..., ::-1])
        
        # 使用 [0.5, 2, 2, 0.5] 比例实现 PC 端舒适显示
        _, mid1, mid2, _ = st.columns([0.5, 2, 2, 0.5])
        with mid1: st.image(img, caption="原始输入", use_container_width=True)
        with mid2: st.image(res_plot, caption="AI 诊断结果", use_container_width=True)
        
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        add_record("单图", file.name, found, res_plot)
        show_report(found)

# 模块 2：批量分析
with tab2:
    files = st.file_uploader("批量上传多张照片...", accept_multiple_files=True, key="multi_u")
    if files and st.button("🚀 开始批量分析"):
        for f in files:
            img = Image.open(f)
            res = model(img, conf=conf_val, iou=iou_val)[0]
            found = [res.names[int(b.cls[0])] for b in res.boxes]
            with st.expander(f"查看结果：{f.name}"):
                st.image(Image.fromarray(res.plot()[..., ::-1]), use_container_width=True)
                show_report(found)

# 模块 3：拍照识别
with tab3:
    img_file = st.camera_input("请正对病害叶片拍摄...")
    if img_file:
        img = Image.open(img_file)
        results = model(img, conf=conf_val, iou=iou_val)[0]
        res_plot = Image.fromarray(results.plot()[..., ::-1])
        st.image(res_plot, caption="拍摄结果", use_container_width=True)
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        add_record("拍照", "Capture.jpg", found, res_plot)
        show_report(found)

# 模块 4：历史记录与下载
with tab4:
    if not st.session_state['history']:
        st.info("尚无检测记录。")
    else:
        dl1, dl2 = st.columns(2)
        with dl1:
            pdf_data = export_to_pdf(st.session_state['history'])
            st.download_button("📥 下载 PDF 报告", pdf_data, "Report.pdf", "application/pdf")
        with dl2:
            xlsx_data = export_to_excel(st.session_state['history'])
            st.download_button("📊 下载 Excel 记录表", xlsx_data, "Records.xlsx", "application/vnd.ms-excel")
        
        st.markdown("---")
        for i, rec in enumerate(st.session_state['history']):
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 2, 1])
                with c1: st.image(rec["图"], use_container_width=True)
                with c2: st.write(f"🆔 **ID**: {rec['随机编号']}\n\n📅 **时间**: {rec['时间']}\n\n🔬 **结果**: {rec['结果']}")
                with c3:
                    if st.button("查看建议", key=f"hist_{i}"): show_report(rec["原始"])

# 模块 5：模型指标
with tab5:
    if os.path.exists(TRAIN_LOG_DIR):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if os.path.exists("results.png"): st.image("results.png", caption="训练性能曲线", use_container_width=True)
        with col_m2:
            if os.path.exists("confusion_matrix.png"): st.image("confusion_matrix.png", caption="混淆矩阵", use_container_width=True)
