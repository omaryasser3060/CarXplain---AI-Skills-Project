import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import io
import time
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

try:
    from utils.model_helper import load_custom_model, smart_preprocess
    from utils.class_names import CAR_CLASSES
    from navbar.navbar import render_navbar
    from footer.footer import render_footer
except ImportError as e:
    def render_navbar():
        pass


    def render_footer():
        pass


    def load_custom_model(p):
        return None


    def smart_preprocess(i, m):
        return np.zeros((1, 224, 224, 3))


    CAR_CLASSES = ["Car"]

ICON_COMPARE_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#00CCFF" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
  <line x1="18" y1="20" x2="18" y2="10"></line>
  <line x1="12" y1="20" x2="12" y2="4"></line>
  <line x1="6" y1="20" x2="6" y2="14"></line>
</svg>
"""
icon_path = "compare_icon.svg"
with open(icon_path, "w") as f:
    f.write(ICON_COMPARE_SVG)

st.set_page_config(
    page_title="Model Benchmark - CarXplain",
    page_icon=icon_path,
    layout="wide",
    initial_sidebar_state="collapsed"
)

ICON_UPLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 16 12 12 8 16"></polyline><line x1="12" y1="12" x2="12" y2="21"></line><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path><polyline points="16 16 12 12 8 16"></polyline></svg>"""
ICON_RESULTS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>"""
ICON_WINNER = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00CCFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"></circle><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"></polyline></svg>"""
ICON_IMAGE = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>"""

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_dir = os.path.join(project_root, 'models')

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");

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

    .main-header-container { display: flex; align-items: center; gap: 20px; margin-bottom: 1rem; margin-top: -50px; }
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

    .glass-card {
        background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(0, 204, 255, 0.2);
        border-radius: 10px; padding: 20px; margin-bottom: 20px; transition: all 0.3s ease;
    }
    .glass-card:hover { border-color: #00CCFF; transform: translateY(-5px); background: rgba(0, 204, 255, 0.05); }
    .glass-card.winner { border: 2px solid #00CCFF; box-shadow: 0 0 20px rgba(0, 204, 255, 0.2); background: rgba(0, 204, 255, 0.1); }

    .model-name { color: #AAAAAA; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
    .model-result { color: #FFFFFF; font-size: 1.4rem; font-weight: 700; margin: 5px 0; }
    .confidence-bar-bg { width: 100%; background: rgba(255,255,255,0.1); height: 6px; border-radius: 3px; overflow: hidden; margin-top: 10px; }
    .confidence-bar-fill { height: 100%; background: linear-gradient(90deg, #005f73, #00CCFF); border-radius: 3px; }

</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_all_models():
    models = {}
    try:
        models['InceptionV3'] = load_custom_model(os.path.join(models_dir, "1-inceptionv3-training-code.keras"))
        models['ResNet50'] = load_custom_model(os.path.join(models_dir, "resnet50_best.keras"))
        models['EfficientNetB4'] = load_custom_model(os.path.join(models_dir, "efficientnetb4_best_model.keras"))
    except Exception as e:
        print(f"Error loading models: {e}")
    return models


def generate_comparison_report(image_bytes, results):
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

            main_title = "CarAI Benchmark"
            sub_title = "Model Comparison"

            canvas.setFont("Helvetica-Bold", 24)
            canvas.setFillColor(COLOR_TEXT)
            canvas.drawString(MARGIN_X, PAGE_H - 55, main_title)

            canvas.setFont("Helvetica-Bold", 18)
            canvas.setFillColor(COLOR_NEON)
            canvas.drawString(MARGIN_X + canvas.stringWidth(main_title, "Helvetica-Bold", 24), PAGE_H - 55, sub_title)

            canvas.setStrokeColor(COLOR_NEON)
            canvas.setLineWidth(0.8)
            canvas.line(MARGIN_X, PAGE_H - 70, PAGE_W - MARGIN_X, PAGE_H - 70)

            canvas.restoreState()

        styles = getSampleStyleSheet()
        style_h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=15,
                                  textColor=COLOR_NEON, spaceBefore=20, spaceAfter=12)

        story = [Spacer(1, 0.25 * inch)]

        story.append(Paragraph("01 // INPUT IMAGE", style_h1))
        if image_bytes:
            img_io = io.BytesIO(image_bytes)
            pil_img = Image.open(img_io).convert("RGB")
            orig_w, orig_h = pil_img.size
            aspect = orig_h / float(orig_w)
            target_w = 5 * inch
            target_h = target_w * aspect
            if target_h > 3 * inch:
                target_h = 3 * inch
                target_w = target_h / aspect

            rl_img = RLImage(img_io, width=target_w, height=target_h)
            story.append(rl_img)
            story.append(Spacer(1, 20))

        story.append(Paragraph("02 // BENCHMARK RESULTS", style_h1))

        data = [["MODEL ARCHITECTURE", "PREDICTED CLASS", "CONFIDENCE"]]
        for res in results:
            data.append([res['Model'], res['Class'], f"{res['Conf']:.2%}"])

        col_w = (PAGE_W - 2 * MARGIN_X) / 3
        t = Table(data, colWidths=[col_w] * 3)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_PANEL),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_NEON),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.Color(1, 1, 1, 0.05)),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(t)

        doc.build(story, onFirstPage=header_footer_gen, onLaterPages=header_footer_gen)
        return buffer.getvalue(), f"Benchmark_Report_{datetime.now().strftime('%H%M')}.pdf"
    except Exception as e:
        print(e)
        return None, None


def on_file_upload():
    if st.session_state.uploader_comp_key:
        st.session_state.comp_img_bytes = st.session_state.uploader_comp_key.getvalue()
        st.session_state.comp_results = None
        st.session_state.comp_camera_enabled = False


def on_camera_capture():
    if st.session_state.camera_comp_key:
        st.session_state.comp_img_bytes = st.session_state.camera_comp_key.getvalue()
        st.session_state.comp_results = None
        st.session_state.comp_camera_enabled = False


def main():
    st.markdown('<div class="body-bg"></div>', unsafe_allow_html=True)

    render_navbar()

    if 'comp_loading' not in st.session_state: st.session_state.comp_loading = False
    if 'comp_results' not in st.session_state: st.session_state.comp_results = None
    if 'comp_img_bytes' not in st.session_state: st.session_state.comp_img_bytes = None
    if 'comp_camera_enabled' not in st.session_state: st.session_state.comp_camera_enabled = False

    st.markdown(f"""
        <div class="main-header-container">
            <div class="icon-box" style="animation: pulse-tech 3s infinite;">{ICON_COMPARE_SVG}</div>
            <div class="text-box">
                <h1>Model Benchmark</h1>
                <p>Compare performance of InceptionV3, ResNet50, and EfficientNetB4 side-by-side.</p>
            </div>
        </div>
        <hr class="styled-hr">
        <style>@keyframes pulse-tech {{ 0% {{ opacity: 0.5; }} 50% {{ opacity: 1; text-shadow: 0 0 10px #00CCFF; }} 100% {{ opacity: 0.5; }} }}</style>
    """, unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([1, 0.1, 1.5])

    with col1:
        st.markdown(f'<div class="section-header">{ICON_UPLOAD} <span>Input Data</span></div>', unsafe_allow_html=True)

        visual_frame = st.container(border=True, height=350)

        with visual_frame:
            scan_class = "scan-effect" if st.session_state.comp_loading else ""

            if st.session_state.comp_img_bytes:
                img_b64 = base64.b64encode(st.session_state.comp_img_bytes).decode()
                st.markdown(
                    f"""<div class="img-container {scan_class}"><img src="data:image/png;base64,{img_b64}" /></div>""",
                    unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    height: 100%;
                    min-height: 300px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    text-align: center;
                    color: #777;
                ">
                    <div style="margin-bottom: 8px; opacity: 0.7; color: #00CCFF;">
                        {ICON_IMAGE}
                    </div>
                    <p style="margin: 0; font-size: 1rem; color: #AAAAAA;">
                        Waiting for image...
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab_upload, tab_camera = st.tabs(["Upload Image", "Use Camera"])

        with tab_upload:
            st.markdown('<div class="browse-button-only">', unsafe_allow_html=True)
            st.file_uploader(
                "Choose Image",
                type=["jpg", "png", "jpeg"],
                label_visibility="collapsed",
                key="uploader_comp_key",
                on_change=on_file_upload
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_camera:
            if not st.session_state.comp_camera_enabled:
                if st.button("Activate Camera", use_container_width=True):
                    st.session_state.comp_camera_enabled = True
                    st.rerun()
            else:
                st.camera_input(
                    "Take a picture",
                    label_visibility="collapsed",
                    key="camera_comp_key",
                    on_change=on_camera_capture
                )
                if st.button("Close Camera", use_container_width=True):
                    st.session_state.comp_camera_enabled = False
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        btn_disabled = st.session_state.comp_loading or (st.session_state.comp_img_bytes is None)
        if st.button("RUN BENCHMARK", use_container_width=True, disabled=btn_disabled):
            st.session_state.comp_loading = True
            st.session_state.comp_results = None
            st.rerun()

    with col2:
        st.markdown(f'<div class="section-header">{ICON_RESULTS} <span>Analysis Results</span></div>',
                    unsafe_allow_html=True)

        if st.session_state.comp_loading:
            st.markdown("""
                <div class="custom-loader">
                    <div class="loader-spinner"></div>
                    <p>Running Inference...</p>
                    <small>Loading models & processing inputs</small>
                </div>
            """, unsafe_allow_html=True)

            try:
                models = load_all_models()
                if not models:
                    st.error("No models found!")
                    st.session_state.comp_loading = False
                    st.stop()

                image = Image.open(io.BytesIO(st.session_state.comp_img_bytes)).convert("RGB")

                results = []

                if 'InceptionV3' in models:
                    img_inc = smart_preprocess(image, "InceptionV3")
                    p1 = models['InceptionV3'].predict(img_inc, verbose=0)
                    results.append(
                        {"Model": "InceptionV3", "Class": CAR_CLASSES[p1[0].argmax()], "Conf": float(p1[0].max())})

                if 'ResNet50' in models:
                    img_res = smart_preprocess(image, "ResNet50")
                    p2 = models['ResNet50'].predict(img_res, verbose=0)
                    results.append(
                        {"Model": "ResNet50", "Class": CAR_CLASSES[p2[0].argmax()], "Conf": float(p2[0].max())})

                if 'EfficientNetB4' in models:
                    img_eff = smart_preprocess(image, "EfficientNetB4")
                    p3 = models['EfficientNetB4'].predict(img_eff, verbose=0)
                    results.append(
                        {"Model": "EfficientNetB4", "Class": CAR_CLASSES[p3[0].argmax()], "Conf": float(p3[0].max())})

                st.session_state.comp_results = results
                st.session_state.comp_loading = False
                st.rerun()

            except Exception as e:
                st.error(f"Processing Error: {e}")
                st.session_state.comp_loading = False

        elif st.session_state.comp_results:
            results = st.session_state.comp_results

            winner = max(results, key=lambda x: x['Conf'])

            st.markdown(f"""
                <div class="glass-card winner" style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="model-name" style="color:#00CCFF;">HIGHEST CONFIDENCE</div>
                        <div class="model-result" style="font-size:1.8rem;">{winner['Model']}</div>
                        <div style="color:#EEE;">{winner['Class']} ({winner['Conf']:.1%})</div>
                    </div>
                    <div style="color:#00CCFF; font-size:3rem; opacity:0.8;">{ICON_WINNER}</div>
                </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]

            for i, res in enumerate(results):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="model-name">{res['Model']}</div>
                        <div class="model-result">{res['Class']}</div>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" style="width: {res['Conf'] * 100}%;"></div>
                        </div>
                        <div style="text-align:right; font-size:0.8rem; margin-top:5px; color:#00CCFF;">{res['Conf']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

            df_res = pd.DataFrame(results)
            fig = px.bar(df_res, x="Model", y="Conf", color="Conf",
                         color_continuous_scale=["#0b2f4f", "#00CCFF"],
                         text_auto='.2%', title="Confidence Comparison")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white', xaxis_title="", yaxis_title="Confidence"
            )
            fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Generate Comparison Report", use_container_width=True):
                pdf_bytes, fname = generate_comparison_report(st.session_state.comp_img_bytes, results)
                if pdf_bytes:
                    st.download_button("Download PDF", pdf_bytes, fname, "application/pdf", use_container_width=True)

        else:
            st.markdown(f"""
                <div style="height: 400px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; color: #777; border: 1px dashed rgba(0, 204, 255, 0.2); border-radius: 10px;">
                    <div style="margin-bottom: 5px; opacity: 0.6;">{ICON_RESULTS}</div>
                    <p style="margin: 0; margin-bottom: 10px; font-size: 1.1rem; font-weight: 500;">Results will appear here</p>
                    <small>Upload an image and run benchmark to compare models.</small>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 60px;'></div>", unsafe_allow_html=True)

    render_footer()


if __name__ == "__main__":
    main()