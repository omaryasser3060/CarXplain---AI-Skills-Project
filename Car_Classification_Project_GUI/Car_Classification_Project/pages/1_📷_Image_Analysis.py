import streamlit as st
from PIL import Image, ImageFilter
import plotly.express as px
import os
import time
import base64
import numpy as np
import io
import pandas as pd
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

try:
    from utils.model_helper import load_custom_model, smart_preprocess, make_gradcam_heatmap, overlay_heatmap, \
        get_last_conv_layer
    from utils.class_names import CAR_CLASSES
    from navbar.navbar import render_navbar
    from footer.footer import render_footer
except ImportError:
    def render_navbar():
        pass


    def render_footer():
        pass


    def load_custom_model(p):
        return None


    def smart_preprocess(i, m):
        return np.zeros((1, 224, 224, 3))


    def make_gradcam_heatmap(i, m, l):
        return np.zeros((10, 10))


    def overlay_heatmap(h, i):
        return np.array(i)


    def get_last_conv_layer(m):
        return None


    CAR_CLASSES = ["Car"]

ICON_ANALYSIS_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#00CCFF" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
  <path d="M2 12l5-5 5 5 5-5 5 5M2 12v5a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-5"></path>
</svg>
"""

icon_path = "analysis_icon.svg"
with open(icon_path, "w") as f:
    f.write(ICON_ANALYSIS_SVG)

st.set_page_config(
    page_title="Intelligent Analysis - CarXplain",
    page_icon=icon_path,
    layout="wide"
)


@st.cache_resource(show_spinner=False)
def get_cached_model(model_path):
    return load_custom_model(model_path)


ICON_UPLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 16 12 12 8 16"></polyline><line x1="12" y1="12" x2="12" y2="21"></line><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path><polyline points="16 16 12 12 8 16"></polyline></svg>"""
ICON_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>"""
ICON_RESULTS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>"""
ICON_IMAGE = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>"""

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_dir = os.path.join(project_root, 'models')


def load_css(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


assets_path = os.path.join(project_root, 'assets', 'global.css')
load_css(assets_path)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #020c1a; border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 204, 255, 0.5); border-radius: 4px; border: 1px solid #020c1a; }
    ::-webkit-scrollbar-thumb:hover { background: #00CCFF; box-shadow: 0 0 10px rgba(0, 204, 255, 0.7); }
    .body-bg {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -2;
        background: linear-gradient(-45deg, #020c1a, #0b2f4f, #005f73, #0a9396);
        background-size: 400% 400%; animation: gradientBG 20s ease infinite;
    }
    @keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    .main-header-container { display: flex; align-items: center; gap: 20px; margin-bottom: 1rem; margin-top: -120px; }
    .main-header-container .icon-box { display: flex; justify-content: center; align-items: center; color: #00CCFF; text-shadow: 0 0 15px rgba(0, 204, 255, 0.5); }
    .main-header-container .text-box h1 { font-size: 2.75rem; font-weight: 700; margin: 0; line-height: 1.1; background: linear-gradient(90deg, #33DFFF, #00CCFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .main-header-container .text-box p { font-size: 1.1rem; font-weight: 300; color: #BBBBBB; margin: 0.5rem 0 0 0; }
    .styled-hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 204, 255, 0), rgba(0, 204, 255, 0.5), rgba(0, 204, 255, 0)); margin-top: 0.5rem; margin-bottom: 1.5rem; }
    .section-header { border-bottom: 2px solid #00CCFF; padding-bottom: 10px; margin-bottom: 1.5rem; font-weight: 600; font-size: 1.5rem; display: flex; align-items: center; gap: 12px; }
    .section-header svg { color: #00CCFF; }
    .stButton > button { width: 100%; background-color: #00CCFF; color: #020c1a; font-weight: 700; border: none; padding: 0.75rem; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #33DFFF; transform: translateY(-2px); box-shadow: 0 0 15px rgba(0, 204, 255, 0.4); }
    .stButton > button:disabled { background-color: rgba(255, 255, 255, 0.2); color: #888888; }
    .browse-button-only { margin-top: 0.5rem; }
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        background-color: rgba(0, 204, 255, 0.03) !important; border: 2px dashed rgba(0, 204, 255, 0.5) !important; border-radius: 10px !important; padding: 1rem; transition: border 0.3s;
    }
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #00CCFF !important; background-color: rgba(0, 204, 255, 0.1) !important;
    }
    .img-container { width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; padding: 20px; position: relative; border-radius: 8px; overflow: hidden; background: rgba(0,0,0,0.2); }
    .img-container img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; }
    .img-container.scan-effect { border: 2px solid rgba(0, 204, 255, 0.5); animation: pulse-border 1.5s infinite alternate; }
    .img-container.scan-effect::before { 
        content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
        background-image: linear-gradient(#00CCFF 1px, transparent 1px), linear-gradient(90deg, #00CCFF 1px, transparent 1px);
        background-size: 40px 40px; background-position: 0 0; z-index: 10; opacity: 0.3;
        animation: grid-move 3s linear infinite;
    }
    @keyframes grid-move { 0% { background-position: 0 0; } 100% { background-position: 40px 40px; } }
    @keyframes pulse-border { from { box-shadow: 0 0 5px rgba(0, 204, 255, 0.3); } to { box-shadow: 0 0 20px rgba(0, 204, 255, 0.6); } }
    .custom-loader { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding-top: 150px; }
    .custom-loader .loader-spinner { width: 50px; height: 50px; border: 4px solid rgba(255, 255, 255, 0.2); border-top-color: #00CCFF; border-radius: 50%; animation: spin 1s linear infinite; }
    .custom-loader p { font-weight: 600; color: #E0E0E0; animation: pulse-text 1.5s infinite ease-in-out; margin: 10px 0 0 0; }
    @keyframes pulse-text { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .summary-metric-card { background-color: rgba(0, 0, 0, 0.2); border: 1px solid rgba(0, 204, 255, 0.3); border-radius: 10px; padding: 20px 10px; text-align: center; transition: all 0.3s ease-in-out; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    .summary-metric-card:hover { border-color: #00CCFF; box-shadow: 0 0 15px rgba(0, 204, 255, 0.3); transform: translateY(-5px); background-color: rgba(0, 204, 255, 0.05); }
    .summary-metric-card .label { color: #AAAAAA; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .summary-metric-card .value { color: #FFFFFF; font-size: 1.3rem; font-weight: 700; text-shadow: 0 0 10px rgba(0, 204, 255, 0.5); line-height: 1.2; }
    .summary-metric-card .sub-text { font-size: 0.75rem; margin-top: 5px; font-weight: 500; }
    .positive { color: #00CCFF; }
</style>
""", unsafe_allow_html=True)


def show_custom_toast(message, type="success"):
    icon = "check-circle-fill" if type == "success" else "exclamation-triangle-fill"
    color = "#00CCFF" if type == "success" else "#FF4136"
    title = "Success" if type == "success" else "Notice"
    st.markdown(f"""
    <div style="position: fixed; top: 80px; right: 20px; z-index: 99999; width: 350px; animation: toastIn 0.5s forwards;">
        <div style="background: rgba(2, 12, 26, 0.95); backdrop-filter: blur(10px); border-left: 5px solid {color}; border-radius: 4px; padding: 15px 20px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5); color: #fff; display: flex; align-items: center; gap: 15px;">
            <div style="font-size: 1.5rem; color: {color};"><i class="bi bi-{icon}"></i></div>
            <div>
                <h4 style="margin: 0 0 5px 0; font-size: 1rem; font-weight: 700; color: #fff;">{title}</h4>
                <p style="margin: 0; font-size: 0.85rem; color: #ddd;">{message}</p>
            </div>
        </div>
    </div>
    <style>@keyframes toastIn {{ from {{ opacity: 0; transform: translateX(100%); }} to {{ opacity: 1; transform: translateX(0); }} }}</style>
    """, unsafe_allow_html=True)


def generate_analysis_report(result_data, img_bytes, cam_bytes=None, chart_bytes=None):
    try:
        buffer = io.BytesIO()
        PAGE_W, PAGE_H = A4
        MARGIN_X = 0.6 * inch

        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * inch, bottomMargin=1.1 * inch, leftMargin=MARGIN_X,
                                rightMargin=MARGIN_X)

        COLOR_BG = colors.HexColor('#020c1a')
        COLOR_PANEL = colors.HexColor('#0b1d36')
        COLOR_NEON = colors.HexColor('#00CCFF')
        COLOR_TEXT = colors.white
        COLOR_DIM = colors.HexColor('#8899A6')

        def header_footer_gen(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(COLOR_BG)
            canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

            main_title = "Intelligent Analysis Report"
            sub_title = "Car Classification"

            canvas.setFont("Helvetica-Bold", 24)
            canvas.setFillColor(COLOR_TEXT)
            canvas.drawString(MARGIN_X, PAGE_H - 55, main_title)

            canvas.setFont("Helvetica-Bold", 18)
            canvas.setFillColor(COLOR_NEON)
            canvas.drawString(MARGIN_X + canvas.stringWidth(main_title, "Helvetica-Bold", 24), PAGE_H - 55, sub_title)

            canvas.setStrokeColor(COLOR_NEON)
            canvas.setLineWidth(0.8)
            canvas.line(MARGIN_X, PAGE_H - 70, PAGE_W - MARGIN_X, PAGE_H - 70)

            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(COLOR_DIM)
            canvas.drawString(MARGIN_X, 40, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            canvas.drawRightString(PAGE_W - MARGIN_X, 40, f"Page {doc.page}")

            canvas.restoreState()

        styles = getSampleStyleSheet()
        style_h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=16,
                                  textColor=COLOR_NEON, spaceBefore=20, spaceAfter=12)
        style_normal = ParagraphStyle('Normal_W', parent=styles['Normal'], textColor=colors.white)

        story = [Spacer(1, 0.25 * inch)]

        story.append(Paragraph("01 // SUMMARY RESULTS", style_h1))
        data = [
            ["Model Architecture", result_data['model_name']],
            ["Detected Class", result_data['top_class']],
            ["Confidence Score", f"{result_data['confidence'] * 100:.2f}%"]
        ]

        col_w = (PAGE_W - 2 * MARGIN_X) / 2
        t = Table(data, colWidths=[col_w * 0.6, col_w * 1.4])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), COLOR_PANEL),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#051426')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('GRID', (0, 0), (-1, -1), 0.5, COLOR_DIM),
            ('BOX', (0, 0), (-1, -1), 1, COLOR_NEON),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

        story.append(Paragraph("02 // VISUAL EVIDENCE", style_h1))
        img_table_data = []
        row = []
        if img_bytes:
            img1 = RLImage(io.BytesIO(img_bytes), width=3.3 * inch, height=2.5 * inch)
            row.append([img1, Paragraph("Original Input", style_normal)])
        if cam_bytes:
            img2 = RLImage(io.BytesIO(cam_bytes), width=3.3 * inch, height=2.5 * inch)
            row.append([img2, Paragraph("Grad-CAM Heatmap", style_normal)])

        final_img_data = [[item[0] for item in row], [item[1] for item in row]] if row else []
        if final_img_data:
            t_imgs = Table(final_img_data, colWidths=[3.5 * inch] * len(row))
            t_imgs.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ]))
            story.append(t_imgs)

        story.append(Spacer(1, 20))

        if chart_bytes:
            story.append(Paragraph("03 // PROBABILITY DISTRIBUTION", style_h1))
            chart_img = RLImage(io.BytesIO(chart_bytes), width=PAGE_W - 2 * MARGIN_X, height=3 * inch)
            story.append(chart_img)

        doc.build(story, onFirstPage=header_footer_gen, onLaterPages=header_footer_gen)
        return buffer.getvalue(), f"Analysis_Report_{datetime.now().strftime('%H%M%S')}.pdf"
    except Exception as e:
        print(f"Report Error: {e}")
        return None, None


def on_file_upload():
    if st.session_state.uploader_key:
        st.session_state.img_bytes_current = st.session_state.uploader_key.getvalue()
        st.session_state.analysis_result = None
        st.session_state.camera_enabled = False


def on_camera_capture():
    if st.session_state.camera_key:
        st.session_state.img_bytes_current = st.session_state.camera_key.getvalue()
        st.session_state.analysis_result = None
        st.session_state.camera_enabled = False


def main():
    st.markdown('<div class="body-bg"></div>', unsafe_allow_html=True)
    render_navbar()

    if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None
    if 'gradcam_bytes' not in st.session_state: st.session_state.gradcam_bytes = None
    if 'loading_analysis' not in st.session_state: st.session_state.loading_analysis = False
    if 'img_bytes_current' not in st.session_state: st.session_state.img_bytes_current = None
    if 'camera_enabled' not in st.session_state: st.session_state.camera_enabled = False

    st.markdown(f"""
    <div class="main-header-container">
        <div class="icon-box" style="animation: pulse-tech 3s infinite;">{ICON_ANALYSIS_SVG}</div>
        <div class="text-box">
            <h1>Intelligent Analysis</h1>
            <p>Deep Learning inference with Explainable AI visualization.</p>
        </div>
    </div>
    <hr class="styled-hr">
    <style>@keyframes pulse-tech {{ 0% {{ opacity: 0.5; }} 50% {{ opacity: 1; text-shadow: 0 0 10px #00CCFF; }} 100% {{ opacity: 0.5; }} }}</style>
    """, unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([1, 0.15, 1.2])

    with col1:
        st.markdown(f'<div class="section-header">{ICON_UPLOAD} <span>Input Data</span></div>', unsafe_allow_html=True)
        visual_frame = st.container(border=True, height=350)
        with visual_frame:
            scan_class = "scan-effect" if st.session_state.loading_analysis else ""
            if st.session_state.img_bytes_current:
                img_b64 = base64.b64encode(st.session_state.img_bytes_current).decode()
                st.markdown(
                    f"""<div class="img-container {scan_class}"><img src="data:image/png;base64,{img_b64}" /></div>""",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""<div style="height: 100%; min-height: 300px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; color: #777;"><div style="margin-bottom: 8px; opacity: 0.7; color: #00CCFF;">{ICON_IMAGE}</div><p style="margin: 0; font-size: 1rem; color: #AAAAAA;">Waiting for image...</p></div>""",
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab_upload, tab_camera = st.tabs(["Upload Image", "Use Camera"])
        with tab_upload:
            st.markdown('<div class="browse-button-only">', unsafe_allow_html=True)
            st.file_uploader("Choose Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed",
                             key="uploader_key", on_change=on_file_upload)
            st.markdown('</div>', unsafe_allow_html=True)
        with tab_camera:
            if not st.session_state.camera_enabled:
                if st.button("Activate Camera", use_container_width=True):
                    st.session_state.camera_enabled = True
                    st.rerun()
            else:
                st.camera_input("Take a picture", label_visibility="collapsed", key="camera_key",
                                on_change=on_camera_capture)
                if st.button("Close Camera", use_container_width=True):
                    st.session_state.camera_enabled = False
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">{ICON_SETTINGS} <span>Configuration</span></div>',
                    unsafe_allow_html=True)
        model_choice = st.selectbox("Select Model Architecture", ["InceptionV3", "ResNet50", "EfficientNetB4"],
                                    label_visibility="collapsed")

        model_paths = {
            "InceptionV3": os.path.join(models_dir, "1-inceptionv3-training-code.keras"),
            "ResNet50": os.path.join(models_dir, "resnet50_best.keras"),
            "EfficientNetB4": os.path.join(models_dir, "efficientnetb4_best_model.keras")
        }

        st.markdown("<br>", unsafe_allow_html=True)
        btn_disabled = st.session_state.loading_analysis or (st.session_state.img_bytes_current is None)
        if st.button("START ANALYSIS", use_container_width=True, disabled=btn_disabled):
            st.session_state.loading_analysis = True
            st.session_state.analysis_result = None
            st.rerun()
        st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="section-header">{ICON_RESULTS} <span>Analysis Results</span></div>',
                    unsafe_allow_html=True)
        result_container = st.container(border=False)

        with result_container:
            if st.session_state.loading_analysis:
                st.markdown(
                    """<div class="custom-loader"><div class="loader-spinner"></div><p style="color:#00CCFF; font-weight:600;">Processing Image...</p><small style="color:#888;">Feature Extraction • Grad-CAM Generation</small></div>""",
                    unsafe_allow_html=True)
                try:
                    image = Image.open(io.BytesIO(st.session_state.img_bytes_current)).convert("RGB")
                    model_path = model_paths[model_choice]
                    if os.path.exists(model_path):
                        model = get_cached_model(model_path)
                        processed_img = smart_preprocess(image, model_choice)
                        start_time = time.time()
                        preds = model.predict(processed_img)
                        inf_time = time.time() - start_time
                        top_3_indices = preds[0].argsort()[-3:][::-1]
                        top_class = CAR_CLASSES[top_3_indices[0]]
                        confidence = preds[0][top_3_indices[0]]
                        result_data = {"top_class": top_class, "confidence": confidence, "top_3_indices": top_3_indices,
                                       "preds": preds, "model_name": model_choice, "inference_time": inf_time}
                        st.session_state.analysis_result = result_data
                        last_conv = get_last_conv_layer(model)
                        st.session_state.gradcam_bytes = None
                        if last_conv:
                            try:
                                heatmap = make_gradcam_heatmap(processed_img, model, last_conv)
                                cam_img = overlay_heatmap(heatmap, image)
                                cam_io = io.BytesIO()
                                Image.fromarray(cam_img).save(cam_io, format='PNG')
                                st.session_state.gradcam_bytes = cam_io.getvalue()
                            except Exception:
                                pass
                    else:
                        st.error(f"Model file not found: {model_path}")
                    st.session_state.loading_analysis = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.loading_analysis = False

            elif st.session_state.analysis_result:
                res = st.session_state.analysis_result
                st.markdown(
                    f"""<div style="background: linear-gradient(90deg, rgba(0, 204, 255, 0.1), rgba(0, 100, 255, 0.1)); border: 1px solid #00CCFF; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px; box-shadow: 0 0 15px rgba(0, 204, 255, 0.1);"><p style="color: #AAAAAA; margin: 0 0 5px 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px;">Detected Vehicle</p><h2 style="color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 700; text-shadow: 0 0 10px rgba(0, 204, 255, 0.6);">{res['top_class']}</h2></div>""",
                    unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(
                        f'<div class="summary-metric-card"><div class="label">Model Used</div><div class="value" style="font-size:1rem;">{res["model_name"]}</div></div>',
                        unsafe_allow_html=True)
                with m2:
                    st.markdown(
                        f'<div class="summary-metric-card"><div class="label">Confidence</div><div class="value positive">{res["confidence"]:.1%}</div></div>',
                        unsafe_allow_html=True)
                with m3:
                    st.markdown(
                        f'<div class="summary-metric-card"><div class="label">Proc. Time</div><div class="value">{res["inference_time"]:.2f}s</div></div>',
                        unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                tab_res1, tab_res2, tab_res3 = st.tabs(["Visual Evidence", "Probabilities", "Explanation"])

                with tab_res1:
                    c_img1, c_img2 = st.columns(2)
                    with c_img1:
                        st.markdown(
                            '<div style="text-align:center; color:#888; margin-bottom:5px;">Original Input</div>',
                            unsafe_allow_html=True)
                        st.image(st.session_state.img_bytes_current, use_container_width=True)
                    with c_img2:
                        st.markdown(
                            '<div style="text-align:center; color:#00CCFF; margin-bottom:5px;">Grad-CAM Heatmap</div>',
                            unsafe_allow_html=True)
                        if st.session_state.gradcam_bytes:
                            st.image(st.session_state.gradcam_bytes, use_container_width=True)
                        else:
                            st.info("Heatmap unavailable")

                with tab_res2:
                    top_3 = res['top_3_indices']
                    chart_data = {"Class": [CAR_CLASSES[i] for i in top_3],
                                  "Confidence": [res['preds'][0][i] for i in top_3]}
                    fig = px.bar(chart_data, x="Confidence", y="Class", orientation='h', text_auto='.1%',
                                 color="Confidence", color_continuous_scale=['#004488', '#00CCFF'])
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                                      height=250, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                with tab_res3:
                    st.markdown(
                        f"""<div style="background:rgba(255,255,255,0.05); padding:15px; border-radius:8px; border-left:3px solid #00CCFF;"><h4 style="margin:0; color:white;">Result: {res['top_class']}</h4><p style="color:#aaa; font-size:0.9rem; margin-top:5px;">The {res['model_name']} model detected <b>{res['top_class']}</b> with high confidence. The heatmap highlights the regions contributing to this decision.</p></div>""",
                        unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("GENERATE PDF REPORT", use_container_width=True):
                    chart_img_bytes = fig.to_image(format="png", width=600, height=300, scale=2)

                    pdf_bytes, pdf_name = generate_analysis_report(
                        st.session_state.analysis_result,
                        st.session_state.img_bytes_current,
                        st.session_state.gradcam_bytes,
                        chart_img_bytes
                    )
                    if pdf_bytes:
                        st.download_button(label="Download PDF", data=pdf_bytes, file_name=pdf_name,
                                           mime="application/pdf", use_container_width=True)
                        show_custom_toast("PDF Report Generated!", "success")
            else:
                st.markdown(
                    f"""<div style="height: 400px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; color: #777;"><div style="margin-bottom: 5px; opacity: 0.6;">{ICON_RESULTS}</div><p style="margin: 0; font-size: 1.1rem; font-weight: 500;">Analysis results will appear here</p></div>""",
                    unsafe_allow_html=True)

    render_footer()


if __name__ == "__main__":
    main()