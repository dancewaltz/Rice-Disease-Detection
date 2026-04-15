import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd

# --- 1. 核心路径与自适应配置 ---
# 优先使用当前目录下的模型和日志图片
MODEL_PATH = "best.pt" 
TRAIN_LOG_DIR = "." 

# 初始化历史记录
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 2. 网页 UI 样式美化 (保留你的绿色主题) ---
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
    /* 隐藏默认侧边栏按钮，确保中心化体验 */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. 核心映射表 (严格对应你的 data.yaml) ---
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
    # 匹配 YAML 中拼写错误的标签
    "ISEASE- Leaf Blast NUTRIENT DEFFICIENT- Silicon- Nitrogen -N- Potassium -K- Potassium -K- Calcium -Ca-": "稻瘟病伴随复合缺素",
    "LEAFBLAST": "稻瘟病"
}

# --- 4. 农艺知识百科库 (使用中文 Key 进行索引) ---
DISEASE_WIKI = {
    "白叶枯病": {
        "desc": "叶片边缘出现黄白色条斑，边缘呈波浪状。严重时全叶枯焦，田间可见菌脓。",
        "advice": "选用抗病品种；控制氮肥；发病初期喷施叶枯唑或农用链霉素。",
        "url": "https://baike.baidu.com/item/水稻白叶枯病"
    },
    "胡麻叶斑病": {
        "desc": "叶片出现芝麻粒大小褐色病斑，中心灰白色。常由高温高湿、土壤缺钾引起。",
        "advice": "增施钾肥，提高植株抗性；喷施苯醚甲环唑或三环唑进行防治。",
        "url": "https://baike.baidu.com/item/水稻胡麻叶斑病"
    },
    "缺镁症": {
        "desc": "老叶叶脉间失绿变黄，但叶脉保持绿色，叶片呈现明显的条纹状失绿。",
        "advice": "增施钙镁磷肥或硫酸镁；叶面喷施1%-2%的硫酸镁溶液。",
        "url": "https://baike.baidu.com/item/水稻缺镁症"
    },
    "缺氮症": {
        "desc": "植株矮小，分蘖少。叶片由下而上均匀发黄，叶尖枯萎。",
        "advice": "及时补施尿素或碳铵；配合叶面喷施1%的尿素水溶液。",
        "url": "https://baike.baidu.com/item/水稻缺氮症"
    },
    "复合缺素(氮锰钾镁锌)": {
        "desc": "多种大中微量元素同时匮乏，生长严重受阻，表现为多重黄化症状。",
        "advice": "追施三元复合肥；喷施含锌、镁的微量元素叶面肥进行综合补偿。",
        "url": "https://baike.baidu.com/item/科学施肥"
    },
    "窄褐斑病伴随复合缺素": {
        "desc": "病斑呈窄条状且颜色深，同时伴有氮、钾、钙等养分失衡。",
        "advice": "改善土壤通透性；喷施针对性杀菌剂并补充中微量元素肥。",
        "url": "https://baike.baidu.com/item/水稻窄褐斑病"
    },
    "白叶枯病伴随缺硅": {
        "desc": "典型细菌性病害且因缺硅导致表皮坚硬度不足，病原菌更易侵入。",
        "advice": "喷施农用链霉素防治病害；基施或喷施硅酸钠、硅化肥。",
        "url": "https://baike.baidu.com/item/水稻白叶枯病"
    },
    "铁甲虫害与综合治理": {
        "desc": "虫害导致叶片出现白色条斑。需结合农业管理与药剂进行综合防治。",
        "advice": "清理田边杂草；受害严重时喷施杀螟丹或高效氰戊菊酯。",
        "url": "https://baike.baidu.com/item/水稻铁甲虫"
    },
    "叶尖枯病伴随复合缺素": {
        "desc": "叶尖出现枯死斑并向下蔓延，生理代谢紊乱，多种养分匮乏。",
        "advice": "喷施三环唑防治病害；全面补充氮钾肥，并施用石膏或硫基肥。",
        "url": "https://baike.baidu.com/item/水稻叶枯病"
    },
    "健康植株": {
        "desc": "叶片挺拔翠绿，无受害痕迹，光合作用效率高。",
        "advice": "继续保持当前管理方案，注意季节性病虫害预防。",
        "url": "https://baike.baidu.com/item/水稻/133543"
    },
    "水稻铁甲虫": {
        "desc": "叶片可见白色长条状食痕，系害虫啃食叶肉所致。严重时叶片枯焦。",
        "advice": "受害达标时喷施高效氰戊菊酯；清理田边杂草减少虫源。",
        "url": "https://baike.baidu.com/item/水稻铁甲虫"
    },
    "稻瘟病伴随复合缺素": {
        "desc": "出现梭形病斑（稻瘟病），且因多种元素不足导致抗性极低。",
        "advice": "喷施稻瘟灵或三环唑；重点补充硅肥与钾肥增强细胞壁厚度。",
        "url": "https://baike.baidu.com/item/稻瘟病"
    },
    "稻瘟病": {
        "desc": "典型的“梭形斑”，中心灰白边缘红褐。被称为水稻“癌症”。",
        "advice": "控制氮肥、增施磷钾肥；关键时刻喷施三环唑或吡唑醚菌酯。",
        "url": "https://baike.baidu.com/item/稻瘟病"
    }
}

# --- 5. 工具函数 ---
@st.cache_resource
def load_yolo():
    if os.path.exists(MODEL_PATH): return YOLO(MODEL_PATH)
    return None

model = load_yolo()

def add_record(mode, filename, found_classes, result_img):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 存储时先翻译成中文
    cn_names = [CLASS_NAMES_CN.get(c, c) for c in set(found_classes)]
    res_text = ", ".join(cn_names) if cn_names else "健康"
    st.session_state['history'].insert(0, {"时间": now, "结果": res_text, "图": result_img, "原始": found_classes})

def show_report(found_classes):
    """诊断报告渲染组件 - 包含翻译逻辑"""
    if not found_classes:
        st.info("💡 诊断结果：目前检测结果显示植株健康。")
        return
    st.divider()
    for eng_label in set(found_classes):
        # 将模型输出的英文标签翻译为中文 Key
        cn_key = CLASS_NAMES_CN.get(eng_label, eng_label)
        info = DISEASE_WIKI.get(cn_key, {"desc": "暂无该细分标签描述", "advice": "咨询农技员", "url": "#"})
        with st.container(border=True):
            st.markdown(f"### 💊 {cn_key}")
            st.write(f"**【病症特征】**：{info['desc']}")
            st.success(f"**【防治建议】**：{info['advice']}")
            if info['url'] != "#":
                st.link_button("🔗 查看百科详情", info['url'])

# --- 6. 核心 UI 布局 ---
st.title("🌾 水稻病害检测识别系统")
st.markdown("---")

# 中心标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 智能单图检测", "📂 专业批量分析", "🤳 手机拍照识别", "📜 历史数据中心", "📈 模型性能评估"])

# 隐藏式参数设置
with st.expander("🛠️ 高级算法参数设置（仅在光线较差或识别不准时调节）"):
    col_a, col_b = st.columns(2)
    with col_a:
        conf_val = st.slider("识别置信度 (Conf)", 0.05, 1.0, 0.45)
    with col_b:
        iou_val = st.slider("重叠过滤 (IoU)", 0.1, 0.9, 0.35)

# --- 7. 业务模块实现 ---

# 模块 1：单图检测
with tab1:
    file = st.file_uploader("请选择一张照片...", type=['jpg','jpeg','png'], key="single_u")
    if file:
        img = Image.open(file)
        results = model(img, conf=conf_val, iou=iou_val)[0]
        res_plot = Image.fromarray(results.plot()[..., ::-1])
        c1, c2 = st.columns(2)
        with c1: st.image(img, caption="原始输入", use_container_width=True)
        with c2: st.image(res_plot, caption="AI 诊断结果", use_container_width=True)
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        add_record("单图", file.name, found, res_plot)
        show_report(found)

# 模块 2：批量分析
with tab2:
    files = st.file_uploader("批量上传多张照片...", accept_multiple_files=True, key="multi_u")
    if files and st.button("🚀 开始自动化批量分析"):
        for f in files:
            img = Image.open(f)
            res = model(img, conf=conf_val, iou=iou_val)[0]
            res_plot = Image.fromarray(res.plot()[..., ::-1])
            found = [res.names[int(b.cls[0])] for b in res.boxes]
            with st.expander(f"查看结果：{f.name}"):
                st.image(res_plot)
                show_report(found)

# 模块 3：手机拍照识别
with tab3:
    st.info("💡 提示：点击下方按钮开启权限后，您可以直接拍照识别。")
    img_file = st.camera_input("请正对病害叶片拍摄...")
    if img_file:
        img = Image.open(img_file)
        results = model(img, conf=conf_val, iou=iou_val)[0]
        res_plot = Image.fromarray(results.plot()[..., ::-1])
        st.image(res_plot, caption="拍摄结果", use_container_width=True)
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        add_record("拍照", "Capture.jpg", found, res_plot)
        show_report(found)

# 模块 4：历史中心
with tab4:
    if not st.session_state['history']:
        st.info("尚无检测记录。")
    else:
        for i, rec in enumerate(st.session_state['history']):
            with st.container(border=True):
                ca, cb, cc = st.columns([1, 2, 1])
                with ca: st.image(rec["图"], width=150)
                with cb: st.write(f"📅 {rec['时间']}\n\n结果: **{rec['结果']}**")
                with cc:
                    if st.button("查看建议", key=f"hist_{i}"):
                        show_report(rec["原始"])

# 模块 5：模型指标
with tab5:
    st.header("📊 模型评估指标")
    if os.path.exists(TRAIN_LOG_DIR):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if os.path.exists("results.png"):
                st.image("results.png", caption="训练性能指标曲线")
            else:
                st.warning("未找到 results.png")
        with col_m2:
            if os.path.exists("confusion_matrix.png"):
                st.image("confusion_matrix.png", caption="混淆矩阵分析图")
            else:
                st.warning("未找到 confusion_matrix.png")
    else:
        st.error("模型日志目录不存在。")
