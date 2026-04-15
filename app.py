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
TRAIN_LOG_DIR = "."

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
# --- 2. 农艺知识百科库 (完整13类，修复404链接) ---
# --- 2. 农艺知识百科库 (完全匹配模型输出的 13 类标签) ---
DISEASE_WIKI = {
    # 1. 白叶枯病
    "BACTERIAL LEAF BLIGHT": {
        "name": "白叶枯病",
        "desc": "叶片边缘出现黄白色条斑，边缘呈波浪状。严重时全叶枯焦，田间可见菌脓。",
        "advice": "选用抗病品种；控制氮肥；发病初期喷施叶枯唑或农用链霉素。",
        "url": "https://baike.baidu.com/item/水稻白叶枯病"
    },
    # 2. 褐斑病
    "BROWN SPOT": {
        "name": "褐斑病",
        "desc": "叶片出现芝麻粒大小褐色病斑，中心灰白色。常由高温高湿、土壤缺钾引起。",
        "advice": "增施钾肥，提高植株抗性；喷施苯醚甲环唑或三环唑进行防治。",
        "url": "https://baike.baidu.com/item/水稻胡麻叶斑病"
    },
    # 3. 缺镁症
    "DEFICIENCY- MAGNESIUM": {
        "name": "缺镁症",
        "desc": "老叶叶脉间失绿变黄，但叶脉保持绿色，叶片呈现明显的条纹状失绿。",
        "advice": "增施钙镁磷肥或硫酸镁；叶面喷施1%-2%的硫酸镁溶液。",
        "url": "https://baike.baidu.com/item/水稻缺镁症"
    },
    # 4. 缺氮症
    "DEFICIENCY- NITROGEN": {
        "name": "缺氮症",
        "desc": "植株矮小，分蘖少。叶片由下而上均匀发黄，叶尖枯萎。",
        "advice": "及时补施尿素或碳铵；配合叶面喷施1%的尿素水溶液。",
        "url": "https://baike.baidu.com/item/水稻缺氮症"
    },
    # 5. 复合缺素 (氮镁锌) - 对应模型中的多元素混合标签
    "DEFICIENCY- NITROGEN MAGNESIUM and ZINC": {
        "name": "复合营养缺乏(氮/镁/锌)",
        "desc": "植株极度衰弱，表现为叶片黄化、条纹失绿以及基部白化点并存。",
        "advice": "紧急追施三元复合肥；喷施含锌、镁的微量元素叶面肥进行综合补偿。",
        "url": "https://baike.baidu.com/item/科学施肥"
    },
    # 6. 叶尖枯病 + 复合缺素 (对应你截图中的报错标签)
    "DISEASE- Lead Scald NUTRIENT DEFFICIENT- Nitrogen -N- Potassium -K- Calcium -Ca- Sulfur -S-": {
        "name": "叶枯病伴随复合缺素(氮钾钙硫)",
        "desc": "叶尖出现枯死斑且向下蔓延，同时伴有氮钾钙硫等多种中微量元素匮乏。",
        "advice": "喷施三环唑防治叶枯病；全面补充氮钾肥，并施用石膏或硫基肥补充钙硫。",
        "url": "https://baike.baidu.com/item/水稻叶枯病"
    },
    # 7. 缺硅症
    "NUTRIENT DEFFICIENT- Silicon": {
        "name": "缺硅症",
        "desc": "叶片下垂、手感柔软。植株抗性大幅下降，极易感染稻瘟病及受虫害侵袭。",
        "advice": "施用硅化肥（如钢渣硅肥）；叶面喷施液体硅肥以增强叶片刚性。",
        "url": "https://baike.baidu.com/item/水稻施肥技术"
    },
    # 8. 优良管理 (健康但带管理标签)
    "Best management practices": {
        "name": "健康/管理优良",
        "desc": "植株生长健壮，叶片色泽正常。体现了当前田间肥水管理处于科学状态。",
        "advice": "继续维持当前的肥水管理计划，做好日常巡视预防即可。",
        "url": "https://baike.baidu.com/item/水稻标准化生产技术"
    },
    # 9. 复合缺素 (氮钾钙硫)
    "Nitrogen Potassium Calcium Sulfur": {
        "name": "复合营养缺乏(氮钾钙硫)",
        "desc": "新旧叶片表现异常。叶缘焦枯且生长点受抑制。属于严重的养分失衡性障碍。",
        "advice": "调整土壤酸碱度；追施均衡的复合肥，并针对性补充硫钙等微量元素。",
        "url": "https://baike.baidu.com/item/水稻科学施肥"
    },
    # 10. 健康
    "HEALTHY": {
        "name": "健康植株",
        "desc": "叶片挺拔翠绿，无受害痕迹，光合作用效率高。",
        "advice": "保持当前管理方案，注意季节性病虫害预防。",
        "url": "https://baike.baidu.com/item/水稻/133543"
    },
    # 11. 水稻铁甲虫
    "HISPA": {
        "name": "水稻铁甲虫",
        "desc": "叶片可见白色长条状食痕。系害虫啃食叶肉导致。严重时叶片枯焦如火烧。",
        "advice": "清理田边杂草；受害达标时喷施高效氰戊菊酯或杀螟丹。",
        "url": "https://baike.baidu.com/item/水稻铁甲虫"
    },
    # 12. 稻瘟病
    "LEAFBLAST": {
        "name": "稻瘟病",
        "desc": "出现典型的梭形斑，中心灰白色，边缘红褐色。被称为水稻的“癌症”。",
        "advice": "控制氮肥、增施磷钾肥；关键时刻喷施三环唑、稻瘟灵或吡唑醚菌酯。",
        "url": "https://baike.baidu.com/item/稻瘟病"
    },
    # 13. 缺锌症
    "DEFICIENCY- ZINC": {
        "name": "缺锌症",
        "desc": "俗称“红苗病”。叶片中脉基部失绿发白，出现大量褐色小斑点。",
        "advice": "基施硫酸锌；发病后叶面喷施0.1%-0.2%的硫酸锌溶液。",
        "url": "https://baike.baidu.com/item/水稻缺锌症"
    }
}
# --- 3. 核心工具函数 ---
@st.cache_resource
def load_yolo():
    if os.path.exists(MODEL_PATH): return YOLO(MODEL_PATH)
    return None

model = load_yolo()

if 'history' not in st.session_state: st.session_state['history'] = []

def add_record(mode, filename, found_classes, result_img):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cn_names = [DISEASE_WIKI.get(c, {"name": c})["name"] for c in set(found_classes)]
    res_text = ", ".join(cn_names) if cn_names else "健康"
    st.session_state['history'].insert(0, {"时间": now, "结果": res_text, "图": result_img, "原始": found_classes})

def show_report(found_classes):
    if not found_classes:
        st.info("💡 诊断结果：植株目前健康。")
        return
    st.divider()
    for c in set(found_classes):
        info = DISEASE_WIKI.get(c, {"name": c, "desc": "暂无描述", "advice": "咨询农技员", "url": "#"})
        with st.container(border=True):
            st.markdown(f"### 💊 {info['name']}")
            st.write(f"**【病症特征】**：{info['desc']}")
            st.success(f"**【防治建议】**：{info['advice']}")
            st.link_button("🔗 查看百科详情", info['url'])

# --- 4. 屏幕中央 UI 布局 ---
st.title("🌾 水稻病害检测识别系统")
st.markdown("---")

# A. 核心功能标签页 (取代原有的侧边栏单选框)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 智能单图检测", "📂 专业批量分析", "📹 实时监控模式", "📊 历史数据中心", "📈 模型性能评估"])

# B. 隐藏式高级参数面板 (点击才会出来)
with st.expander("🛠️ 高级算法参数设置（仅在光线较差或识别不准时调节）"):
    col_a, col_b = st.columns(2)
    with col_a:
        conf_val = st.slider("识别置信度 (Conf)", 0.05, 1.0, 0.45, help="数值越高，识别越严苛，漏检率增加；数值越低，越敏感，误检率增加。")
    with col_b:
        iou_val = st.slider("重叠过滤 (IoU)", 0.1, 0.9, 0.35, help="用于处理重叠叶片。")

# --- 5. 业务模块实现 ---

# 模块 1：单图检测
with tab1:
    file = st.file_uploader("请选择一张水稻叶片照片...", type=['jpg','jpeg','png'], key="single_u")
    if file:
        img = Image.open(file)
        results = model(img, conf=conf_val, iou=iou_val)[0]
        res_plot = Image.fromarray(results.plot()[..., ::-1])
        c1, c2 = st.columns(2)
        with c1: st.image(img, caption="原始输入", use_container_width=True)
        with c2: st.image(res_plot, caption="AI 诊断图", use_container_width=True)
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        add_record("单图", file.name, found, res_plot)
        show_report(found)

# 模块 2：批量分析
with tab2:
    files = st.file_uploader("批量上传多张水稻照片...", accept_multiple_files=True, key="multi_u")
    if files and st.button("🚀 开始自动化批量分析"):
        for f in files:
            img = Image.open(f)
            res = model(img, conf=conf_val, iou=iou_val)[0]
            found = [res.names[int(b.cls[0])] for b in res.boxes]
            with st.expander(f"查看文件：{f.name}"):
                st.image(Image.fromarray(res.plot()[..., ::-1]))
                show_report(found)

# 模块 3：实时拍照巡检
with tab3:
    st.markdown("### 📸 实时拍照诊断")
    st.info("💡 提示：点击下方按钮开启权限后，您可以直接拍摄水稻叶片进行即时识别。")
    
    # 调用浏览器原生摄像头组件（适配手机与电脑）
    img_file = st.camera_input("请正对病害部位拍摄...")
    
    if img_file:
        # 将拍摄的数据流转化为 PIL 图像
        img = Image.open(img_file)
        
        # 执行 YOLOv11 推理
        # 使用您在“高级设置”中定义的 conf_val 和 iou_val
        results = model(img, conf=conf_val, iou=iou_val)[0]
        
        # 渲染检测框并处理颜色空间 (OpenCV 的 BGR 转为 RGB)
        res_plot = Image.fromarray(results.plot()[..., ::-1])
        
        # 居中展示识别结果
        st.image(res_plot, caption="实时检测画面", use_container_width=True)
        
        # 自动提取标签并生成专家诊断报告
        found = [results.names[int(b.cls[0])] for b in results.boxes]
        
        # 将此次拍照记录保存至“历史数据中心”
        add_record("实时拍照", f"Capture_{datetime.now().strftime('%H%M%S')}.jpg", found, res_plot)
        
        # 联动展示 13 类百科处方
        show_report(found)
# 模块 4：历史数据
with tab4:
    if not st.session_state['history']:
        st.write("暂无检测记录。")
    else:
        for rec in st.session_state['history']:
            with st.container(border=True):
                ca, cb, cc = st.columns([1, 2, 2])
                with ca: st.image(rec["图"], width=150)
                with cb: st.write(f"📅 {rec['时间']}\n\n检测结果：**{rec['结果']}**")
                with cc:
                    if st.button("展开防治建议", key=rec["时间"]):
                        show_report(rec["原始"])

# 模块 5：模型指标
with tab5:
    if os.path.exists(TRAIN_LOG_DIR):
        st.image(os.path.join(TRAIN_LOG_DIR, "results.png"), caption="模型训练性能指标曲线")
        st.image(os.path.join(TRAIN_LOG_DIR, "confusion_matrix.png"), caption="混淆矩阵分析图")
    else:
        st.warning("未检测到模型日志目录。")
