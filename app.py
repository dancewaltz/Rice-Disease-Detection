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
# --- 2. 农艺知识百科库 (完整13类，修复404链接) ---
DISEASE_WIKI = {
    "BACTERIAL LEAF BLIGHT": {
        "name": "白叶枯病",
        "desc": "发病时叶片边缘出现黄白色条斑，病斑边缘呈波浪状，湿度大时有菌脓。",
        "advice": "1. 选用抗病品种；2. 科学排灌，防止淹水；3. 发病初期喷施叶枯唑或农用链霉素。",
        "url": "https://baike.baidu.com/item/水稻白叶枯病"
    },
    "BROWN SPOT": {
        "name": "胡麻叶斑病",
        "desc": "叶片出现芝麻粒大小褐色病斑，中心灰白色，严重时导致叶片干枯卷缩。",
        "advice": "1. 增施钾肥，提高抗病力；2. 种子消毒；3. 喷施苯醚甲环唑或三环唑。",
        "url": "https://baike.baidu.com/item/水稻胡麻叶斑病"
    },
    "LEAF BLAST": {
        "name": "稻瘟病",
        "desc": "典型症状为梭形斑，中心灰白，边缘红褐色，严重时引起整片稻田枯焦。",
        "advice": "1. 避免偏施氮肥；2. 浅水勤灌；3. 关键时期喷施三环唑或稻瘟灵。",
        "url": "https://baike.baidu.com/item/稻瘟病"
    },
    "HISPA": {
        "name": "水稻铁甲虫",
        "desc": "成虫和幼虫均啃食叶肉，留下白色条斑，导致叶片枯黄，受害严重时如火烧状。",
        "advice": "1. 清理田边杂草；2. 受害严重时选用杀螟丹或高效氰戊菊酯进行喷雾。",
        "url": "https://baike.baidu.com/item/水稻铁甲虫"
    },
    "DEFICIENCY- NITROGEN": {
        "name": "缺氮症",
        "desc": "植株矮小，分蘖减少，老叶首先由叶尖向基部均匀变黄，严重时全株淡黄。",
        "advice": "1. 及时补充尿素等氮肥；2. 配合叶面喷施1%-2%的尿素溶液。",
        "url": "https://baike.baidu.com/item/水稻缺氮症"
    },
    "DEFICIENCY- POTASSIUM": {
        "name": "缺钾症",
        "desc": "老叶叶尖及边缘出现红褐色焦枯，形似火烧，根系发育不良，易倒伏。",
        "advice": "1. 追施氯化钾或硫酸钾；2. 喷施0.2%的磷酸二氢钾溶液。",
        "url": "https://baike.baidu.com/item/水稻缺钾症"
    },
    "DEFICIENCY- MAGNESIUM": {
        "name": "缺镁症",
        "desc": "叶脉间失绿变黄，但叶脉仍保持绿色，呈明显的条纹状，多发生在老叶。",
        "advice": "1. 施用钙镁磷肥或硫酸镁；2. 叶面喷施1%-2%的硫酸镁溶液。",
        "url": "https://baike.baidu.com/item/水稻缺镁症"
    },
    "DEFICIENCY- ZINC": {
        "name": "缺锌症",
        "desc": "叶片中脉基部失绿发白，出现褐色斑点，俗称“红苗病”，植株严重矮缩。",
        "advice": "1. 施用硫酸锌作为基肥；2. 发病后叶面喷施0.1%-0.2%的硫酸锌溶液。",
        "url": "https://baike.baidu.com/item/水稻缺锌症"
    },
    "NUTRIENT DEFICIENT- Silicon": {
        "name": "缺硅症",
        "desc": "叶片柔软下垂，抗病性明显变差，易感染稻瘟病及受虫害侵袭。",
        "advice": "1. 施用硅化肥，如硅酸钙或钢渣硅肥；2. 喷施液体硅肥增强叶片硬度。",
        "url": "https://baike.baidu.com/item/水稻施肥技术"
    },
    "DEFICIENCY- NITROGEN MAGNESIUM and ZINC": {
        "name": "复合缺素(氮镁锌)",
        "desc": "多种元素同时匮乏，表现为植株矮小且叶片大面积黄化、失绿及褐色斑点并存。",
        "advice": "1. 紧急施用三元复合肥；2. 喷施含有微量元素的多元叶面肥进行综合补偿。",
        "url": "https://baike.baidu.com/item/科学施肥"
    },
    "Nitrogen Potassium Calcium Sulfur": {
        "name": "复合缺素(氮钾钙硫)",
        "desc": "表现为新老叶交替变色，叶缘焦枯且生长点受阻，植株抗逆性极差。",
        "advice": "1. 优化肥水管理；2. 补充石膏（钙硫）及平衡氮钾肥；3. 改善土壤通透性。",
        "url": "https://baike.baidu.com/item/水稻科学施肥"
    },
    "HEALTHY": {
        "name": "健康植株",
        "desc": "叶片颜色翠绿，株型挺拔，无病斑及虫害痕迹，生长势头良好。",
        "advice": "继续保持科学的肥水管理，加强日常田间巡视，做好预防工作。",
        "url": "https://baike.baidu.com/item/水稻/133543"
    },
    "Best management practices": {
        "name": "优良田间状态",
        "desc": "农事管理得当，植株表现出极佳的抗性与生长状态，无明显的营养匮乏。",
        "advice": "建议维持当前的灌溉与施肥计划，定期清理杂草，保持田间通透性。",
        "url": "https://baike.baidu.com/item/水稻标准化生产技术"
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
