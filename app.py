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
