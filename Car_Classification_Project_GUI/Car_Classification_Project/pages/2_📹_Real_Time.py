import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import io
import base64
from datetime import datetime
from PIL import Image
import plotly.express as px
import gc
import tensorflow as tf

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
    st.error(f"Error importing project modules: {e}")
    st.stop()

ICON_VIDEO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#00CCFF" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
  <path d="M23 7l-7 5 7 5V7z" />
  <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
</svg>
"""
icon_path = "live_car_icon.svg"
with open(icon_path, "w") as f:
    f.write(ICON_VIDEO_SVG)

st.set_page_config(
    page_title="Real-Time Inspector - CarXplain",
    page_icon=icon_path,
    layout="wide",
    initial_sidebar_state="collapsed"
)

ICON_DASHBOARD = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>"""
ICON_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>"""
ICON_LIVE = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 7l-7 5 7 5V7z"></path><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>"""
ICON_HISTORY = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>"""
ICON_BEST_SHOT = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFD700" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>"""

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
snap_dir = os.path.join(project_root, "snapshots")
os.makedirs(snap_dir, exist_ok=True)

AVAILABLE_MODELS = {
    "EfficientNet-B4": "efficientnetb4_best_model.keras",
    "ResNet-50": "resnet50_best.keras",
    "Inception-V3": "1-inceptionv3-training-code.keras"
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");

    html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #020c1a; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 204, 255, 0.5); border-radius: 4px; }

    .body-bg {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -2;
        background: radial-gradient(circle at center, #0b2f4f 0%, #020c1a 100%);
    }

    .main-header-container { display: flex; align-items: center; gap: 20px; margin-bottom: 1rem; margin-top: -50px; }
    .main-header-container .icon-box { display: flex; justify-content: center; align-items: center; color: #00CCFF; text-shadow: 0 0 15px rgba(0, 204, 255, 0.5); }
    .main-header-container .text-box h1 { font-size: 2.75rem; font-weight: 700; margin: 0; line-height: 1.1; background: linear-gradient(90deg, #33DFFF, #00CCFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .main-header-container .text-box p { font-size: 1.1rem; font-weight: 300; color: #BBBBBB; margin: 0.5rem 0 0 0; }
    .styled-hr { border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 204, 255, 0), rgba(0, 204, 255, 0.5), rgba(0, 204, 255, 0)); margin-top: 0.5rem; margin-bottom: 1.5rem; }

    .section-header { border-bottom: 2px solid #00CCFF; padding-bottom: 10px; margin-bottom: 1.5rem; font-weight: 600; font-size: 1.5rem; display: flex; align-items: center; gap: 12px; }
    .section-header svg { color: #00CCFF; }

    .stButton > button { width: 100%; background-color: #00CCFF; color: #020c1a; font-weight: 700; border: none; padding: 0.75rem; transition: all 0.3s ease; }
    .stButton > button:hover { background-color: #33DFFF; transform: translateY(-2px); box-shadow: 0 0 15px rgba(0, 204, 255, 0.4); }

    .video-frame { border: 2px solid rgba(0, 204, 255, 0.3); border-radius: 8px; overflow: hidden; position: relative; box-shadow: 0 0 20px rgba(0, 204, 255, 0.1); background: rgba(0, 0, 0, 0.3); }

    .summary-metric-card { background-color: rgba(0, 0, 0, 0.2); border: 1px solid rgba(0, 204, 255, 0.3); border-radius: 10px; padding: 15px 5px; text-align: center; }
    .summary-metric-card .label { color: #AAAAAA; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 5px; }
    .summary-metric-card .value { color: #FFFFFF; font-size: 1.4rem; font-weight: 700; text-shadow: 0 0 10px rgba(0, 204, 255, 0.5); }
    .positive { color: #00CCFF; } .neutral { color: #777; }

    .best-shot-container { border: 2px solid #FFD700; border-radius: 8px; overflow: hidden; position: relative; box-shadow: 0 0 15px rgba(255, 215, 0, 0.2); margin-bottom: 10px; }
    .best-shot-badge { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.8); color: #FFD700; padding: 5px 10px; border-radius: 4px; font-weight: bold; border: 1px solid #FFD700; }

    .history-card { background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255, 255, 255, 0.1); border-left: 3px solid #00CCFF; border-radius: 8px; padding: 15px; margin-bottom: 10px; transition: all 0.2s ease; }
    .history-card:hover { border-color: #00CCFF; transform: translateX(5px); background: rgba(0, 204, 255, 0.05); }
    .history-header { display: flex; justify-content: space-between; align-items: center; font-weight: 700; color: #00CCFF; font-size: 1rem; }
    .hist-meta { font-family: 'Courier New', monospace; font-size: 0.85rem; color: #AAAAAA; margin-top: 4px; }

    .stDataFrame { box-shadow: none !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_car_model(model_filename):
    path = os.path.join(project_root, 'models', model_filename)
    gc.collect()
    return load_custom_model(path)


def save_snapshot(frame, prefix="snap"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    path = os.path.join(snap_dir, filename)
    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return path


def display_detection_charts(df_logs):
    if df_logs is not None and not df_logs.empty:
        chart_theme = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        col1, col2 = st.columns(2)
        with col1:
            counts = df_logs['Car_Model'].value_counts().reset_index()
            counts.columns = ['Model', 'Count']
            fig1 = px.bar(counts, x='Model', y='Count', title="Detections by Model", color='Count',
                          color_continuous_scale='Blues')
            fig1.update_layout(**chart_theme)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.pie(df_logs, names='Car_Model', title='Detection Distribution', hole=0.4,
                          color_discrete_sequence=px.colors.sequential.Blues_r)
            fig2.update_layout(**chart_theme)
            st.plotly_chart(fig2, use_container_width=True)


@st.dialog("Detailed Session Log", width="large")
def view_history_popup(session):
    m1, m2, m3 = st.columns(3)
    m1.metric("Duration", f"{session['duration']:.1f}s")
    det_count = len(session['df']) if session['df'] is not None else 0
    m2.metric("Total Detections", det_count)

    best_conf_disp = "N/A"
    if session.get('best_detection'):
        best_conf_disp = f"{session['best_detection']['conf']:.1%}"
    m3.metric("Top Confidence", best_conf_disp)

    st.markdown("---")

    if session.get('best_detection'):
        st.markdown("#### 🏆 Best Detection")
        bd = session['best_detection']
        if os.path.exists(bd['path']):
            st.image(bd['path'], caption=f"{bd['class']} ({bd['conf']:.1%})", width=400)

    st.markdown("---")
    if session['df'] is not None and not session['df'].empty:
        st.markdown("#### Detection Log")
        st.dataframe(session['df'], use_container_width=True)
        display_detection_charts(session['df'])


def generate_session_report(session_data):
    try:
        buffer = io.BytesIO()
        PAGE_W, PAGE_H = A4
        MARGIN_X = 0.6 * inch
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * inch, bottomMargin=1.1 * inch, leftMargin=MARGIN_X,
                                rightMargin=MARGIN_X)
        COLOR_BG = colors.HexColor('#020c1a')
        COLOR_PANEL = colors.HexColor('#0b1d36')
        COLOR_NEON = colors.HexColor('#00CCFF')
        COLOR_TEAL = colors.HexColor('#0A9396')
        COLOR_TEXT = colors.white
        COLOR_DIM = colors.HexColor('#8899A6')

        def header_footer_gen(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(COLOR_BG)
            canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
            main_title = "CarAI Report "
            sub_title = "Live Inspector"
            canvas.setFont("Helvetica-Bold", 24)
            canvas.setFillColor(COLOR_TEXT)
            canvas.drawString(MARGIN_X, PAGE_H - 55, main_title)
            canvas.setFont("Helvetica-Bold", 18)
            canvas.setFillColor(COLOR_NEON)
            canvas.drawString(MARGIN_X + canvas.stringWidth(main_title, "Helvetica-Bold", 24), PAGE_H - 55, sub_title)
            canvas.setStrokeColor(COLOR_NEON)
            canvas.setLineWidth(0.8)
            canvas.line(MARGIN_X, PAGE_H - 70, PAGE_W - MARGIN_X, PAGE_H - 70)
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(COLOR_DIM)
            canvas.drawString(MARGIN_X, 40, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            canvas.drawRightString(PAGE_W - MARGIN_X, 40, f"Page {doc.page}")
            canvas.restoreState()

        styles = getSampleStyleSheet()
        style_h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=15,
                                  textColor=COLOR_NEON, spaceBefore=20, spaceAfter=12)
        story = [Spacer(1, 0.25 * inch), Paragraph("01 // SESSION METRICS", style_h1)]
        df = session_data['df']

        frames_count = session_data.get('frames_count', 0)
        duration = session_data.get('duration', 1)
        fps_avg = frames_count / duration if duration > 0 else 0
        detections_count = len(df) if df is not None and not df.empty else 0

        col_w = (PAGE_W - 2 * MARGIN_X) / 3
        kpi_data = [["DURATION", "AVG FPS", "MODELS DETECTED"],
                    [f"{duration:.1f}s", f"{fps_avg:.1f}", str(detections_count)]]
        t_metrics = Table(kpi_data, colWidths=[col_w] * 3)
        t_metrics.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_PANEL),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLOR_DIM), ('TEXTCOLOR', (0, 1), (-1, 1), COLOR_NEON),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 16), ('BOTTOMPADDING', (0, 1), (-1, 1), 12),
            ('BOX', (0, 0), (-1, -1), 0.4, COLOR_TEAL)
        ]))
        story.append(t_metrics)
        story.append(Spacer(1, 22))

        if session_data.get('best_detection'):
            story.append(Paragraph("02 // HIGHEST CONFIDENCE DETECTION", style_h1))
            bd = session_data['best_detection']
            if os.path.exists(bd['path']):
                img = RLImage(bd['path'], width=5 * inch, height=3.75 * inch)
                story.append(img)
                story.append(Spacer(1, 10))

                info_data = [[f"MODEL: {bd['class']}", f"CONFIDENCE: {bd['conf']:.2%}", f"TIME: {bd['time']}"]]
                t_info = Table(info_data, colWidths=[2 * inch, 2 * inch, 2 * inch])
                t_info.setStyle(TableStyle([
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                    ('BACKGROUND', (0, 0), (-1, -1), COLOR_PANEL),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('BOX', (0, 0), (-1, -1), 1, colors.gold)
                ]))
                story.append(t_info)
            story.append(Spacer(1, 25))

        if df is not None and not df.empty:
            story.append(PageBreak())
            story.append(Paragraph("03 // ANALYTICS", style_h1))
            chart_theme = dict(plot_bgcolor='#020c1a', paper_bgcolor='#020c1a', font=dict(color='white'))
            counts = df['Car_Model'].value_counts().reset_index()
            counts.columns = ['Model', 'Count']
            fig1 = px.bar(counts, x="Model", y="Count", color_discrete_sequence=['#00CCFF'])
            fig1.update_layout(**chart_theme)
            story.append(
                RLImage(io.BytesIO(fig1.to_image(format="png", width=820, height=350)), width=PAGE_W - 2 * MARGIN_X,
                        height=3.3 * inch))

        doc.build(story, onFirstPage=header_footer_gen, onLaterPages=header_footer_gen)
        return buffer.getvalue(), f"CarAI_Report_{datetime.now().strftime('%H%M')}.pdf"
    except Exception as e:
        print(f"PDF Error: {e}")
        return None, None


def main():
    st.markdown('<div class="body-bg"></div>', unsafe_allow_html=True)
    render_navbar()

    if 'history' not in st.session_state: st.session_state.history = []
    if 'run_rt' not in st.session_state: st.session_state.run_rt = False
    if 'rt_logs' not in st.session_state: st.session_state.rt_logs = []

    if 'best_detection' not in st.session_state: st.session_state.best_detection = None
    if 'best_conf_so_far' not in st.session_state: st.session_state.best_conf_so_far = -1.0

    if 'show_stop_dialog' not in st.session_state: st.session_state.show_stop_dialog = False
    if 'start_time_ref' not in st.session_state: st.session_state.start_time_ref = 0
    if 'accumulated_time' not in st.session_state: st.session_state.accumulated_time = 0
    if 'temp_session_data' not in st.session_state: st.session_state.temp_session_data = None

    st.markdown(f"""
        <div class="main-header-container">
            <div class="icon-box" style="animation: pulse-tech 3s infinite;">{ICON_VIDEO_SVG}</div>
            <div class="text-box">
                <h1>Real-Time Inspector</h1>
                <p>Live stream car model recognition utilizing Advanced CNNs.</p>
            </div>
        </div>
        <hr class="styled-hr">
        <style>@keyframes pulse-tech {{ 0% {{ opacity: 0.5; }} 50% {{ opacity: 1; text-shadow: 0 0 10px #00CCFF; }} 100% {{ opacity: 0.5; }} }}</style>
    """, unsafe_allow_html=True)

    @st.dialog("Session Interrupted")
    def stop_confirmation_dialog():
        st.write("Session paused. Choose action:")
        col_res, col_save, col_disc = st.columns(3)
        with col_res:
            if st.button("Resume", use_container_width=True):
                st.session_state.run_rt = True
                st.session_state.start_time_ref = time.time()
                st.session_state.show_stop_dialog = False
                st.rerun()
        with col_save:
            if st.button("End & Save", use_container_width=True):
                if st.session_state.temp_session_data:
                    st.session_state.history.append(st.session_state.temp_session_data)
                    st.success("Session Saved to History!")

                st.session_state.temp_session_data = None
                st.session_state.accumulated_time = 0
                st.session_state.run_rt = False
                st.session_state.show_stop_dialog = False
                st.rerun()
        with col_disc:
            if st.button("Discard", use_container_width=True):
                st.session_state.temp_session_data = None
                st.session_state.accumulated_time = 0
                st.session_state.run_rt = False
                st.session_state.show_stop_dialog = False
                st.rerun()

    if st.session_state.show_stop_dialog: stop_confirmation_dialog()

    col_input, col_metrics = st.columns([1.5, 1])

    with col_input:
        st.markdown(f'<div class="section-header">{ICON_LIVE} <span>Live Feed</span></div>', unsafe_allow_html=True)
        video_container = st.empty()
        timer_placeholder = st.empty()

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1: start_btn = st.button("START SESSION", disabled=st.session_state.run_rt, use_container_width=True)
        with c2:
            if st.button("STOP SESSION", disabled=not st.session_state.run_rt, use_container_width=True):
                st.session_state.run_rt = False
                st.session_state.accumulated_time += (time.time() - st.session_state.start_time_ref)
                st.session_state.show_stop_dialog = True
                st.rerun()

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">{ICON_SETTINGS} <span>Configuration</span></div>',
                    unsafe_allow_html=True)

        selected_model_name = st.selectbox("Select Classification Model", list(AVAILABLE_MODELS.keys()), index=0)
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

    with col_metrics:
        st.markdown(f'<div class="section-header">{ICON_DASHBOARD} <span>Live Insights</span></div>',
                    unsafe_allow_html=True)
        kpi_container = st.container(height=100, border=False)
        with kpi_container: kpi_placeholder = st.empty()

        st.markdown(
            f'<div style="color:#FFD700; font-weight:bold; margin-bottom:5px;">{ICON_BEST_SHOT} Best Capture So Far</div>',
            unsafe_allow_html=True)
        best_shot_placeholder = st.empty()

        st.caption("Detection Log")
        log_container = st.container(height=300, border=True)
        with log_container: log_placeholder = st.empty()

    if start_btn:
        st.session_state.run_rt = True
        st.session_state.show_stop_dialog = False
        st.session_state.rt_logs = []
        st.session_state.best_detection = None
        st.session_state.best_conf_so_far = -1.0
        st.session_state.temp_session_data = None
        st.session_state.accumulated_time = 0
        st.session_state.start_time_ref = time.time()
        st.rerun()

    if st.session_state.run_rt:
        try:
            model_file = AVAILABLE_MODELS[selected_model_name]
            model = load_car_model(model_file)
            clean_model_name = selected_model_name.replace("-", "")

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                st.error("Optical Sensor Unavailable!")
                st.session_state.run_rt = False
            else:
                frame_count = 0
                last_ui_update = time.time()
                SKIP_FRAMES = 5
                current_label_text = "Scanning..."
                current_color = (100, 100, 100)

                while st.session_state.run_rt:
                    ret, frame = cap.read()
                    if not ret: break

                    current_time = time.time()
                    elapsed_in_this_run = current_time - st.session_state.start_time_ref
                    total_elapsed = st.session_state.accumulated_time + elapsed_in_this_run

                    if total_elapsed >= 60:
                        st.toast("Session Limit Reached (60s)", icon="🏁")

                        final_session_data = {
                            "id": len(st.session_state.history) + 1,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "duration": total_elapsed, "frames_count": frame_count,
                            "best_detection": st.session_state.best_detection,
                            "df": pd.DataFrame(list(st.session_state.rt_logs)) if st.session_state.rt_logs else None
                        }
                        st.session_state.history.append(final_session_data)

                        st.session_state.run_rt = False
                        st.session_state.accumulated_time = 0
                        st.rerun()
                        break

                    if frame_count % SKIP_FRAMES == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_frame)
                        processed_input = smart_preprocess(pil_img, clean_model_name)

                        preds = model(processed_input, training=False).numpy()
                        top_idx = np.argmax(preds[0])
                        top_prob = preds[0][top_idx]
                        top_class = CAR_CLASSES[top_idx]

                        if top_prob >= conf_threshold:
                            current_color = (0, 204, 255)
                            current_label_text = f"{top_class}: {top_prob:.1%}"

                            st.session_state.rt_logs.append({
                                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                                "Car_Model": top_class,
                                "Confidence": top_prob
                            })

                            if top_prob > st.session_state.best_conf_so_far:
                                st.session_state.best_conf_so_far = top_prob
                                frame_to_save = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                path = save_snapshot(frame_to_save, prefix="best")

                                st.session_state.best_detection = {
                                    "class": top_class,
                                    "conf": top_prob,
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "path": path
                                }
                        else:
                            current_color = (100, 100, 100)
                            current_label_text = "Scanning..."

                    cv2.rectangle(frame, (10, 10), (320, 60), (0, 0, 0), -1)
                    cv2.putText(frame, current_label_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, current_color, 2)

                    display_frame = cv2.resize(frame, (640, 480))
                    video_container.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB",
                                          use_container_width=True)
                    frame_count += 1

                    if current_time - last_ui_update > 0.5:
                        remaining = 60 - total_elapsed
                        progress = min(total_elapsed / 60.0, 1.0)
                        timer_placeholder.markdown(f"""
                            <div style="display:flex; justify-content:space-between; color:#00CCFF; font-size:0.8rem; margin-bottom:2px; font-weight:bold;">
                                <span><i class="bi bi-record-circle-fill" style="color:#FF4136;"></i> LIVE</span>
                                <span>{remaining:.0f}s LEFT</span>
                            </div>
                            <div class="timer-container"><div class="timer-bar" style="width: {progress * 100}%;"></div></div>
                        """, unsafe_allow_html=True)

                        with kpi_placeholder.container():
                            fps = frame_count / total_elapsed if total_elapsed > 0 else 0
                            k1, k2 = st.columns(2)
                            with k1: st.markdown(
                                f"""<div class="summary-metric-card"><div class="label">FPS</div><div class="value">{fps:.1f}</div></div>""",
                                unsafe_allow_html=True)
                            with k2:
                                count = len(st.session_state.rt_logs)
                                st.markdown(
                                    f"""<div class="summary-metric-card"><div class="label">Count</div><div class="value" style="font-size:1.5rem;">{count}</div></div>""",
                                    unsafe_allow_html=True)

                        with best_shot_placeholder.container():
                            if st.session_state.best_detection:
                                bd = st.session_state.best_detection
                                st.markdown(f"""
                                    <div class="best-shot-container">
                                        <div class="best-shot-badge">{bd['conf']:.1%}</div>
                                        <img src="data:image/jpeg;base64,{base64.b64encode(open(bd['path'], "rb").read()).decode()}" style="width:100%;">
                                        <div style="background:rgba(0,0,0,0.7); color:white; padding:5px; text-align:center; font-size:0.9rem;">{bd['class']}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("Waiting for high confidence detection...")

                        with log_placeholder.container():
                            if st.session_state.rt_logs:
                                df_disp = pd.DataFrame(st.session_state.rt_logs[-8:]).iloc[::-1]
                                df_disp['Confidence'] = df_disp['Confidence'].apply(lambda x: f"{x:.1%}")
                                st.dataframe(df_disp, use_container_width=True, hide_index=True)
                        last_ui_update = current_time

                    st.session_state.temp_session_data = {
                        "id": len(st.session_state.history) + 1,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "duration": total_elapsed, "frames_count": frame_count,
                        "best_detection": st.session_state.best_detection,
                        "df": pd.DataFrame(list(st.session_state.rt_logs)) if st.session_state.rt_logs else None
                    }

                cap.release()

        except Exception as e:
            st.error(f"Runtime Error: {e}")
            st.session_state.run_rt = False

    else:
        video_container.markdown(f'''
            <div class="video-frame" style="height:350px; display:flex; align-items:center; justify-content:center; flex-direction:column; border-style:dashed; opacity:0.7;">
                <div style="color:#555; font-size:3rem; margin-bottom:10px;">{ICON_VIDEO_SVG}</div>
                <h4 style="color:#AAA; margin:0;">Sensor Offline</h4>
                <small style="color:#555; margin-top:5px;">Click START SESSION to begin</small>
            </div>
        ''', unsafe_allow_html=True)
        with kpi_placeholder:
            st.write("")
        with best_shot_placeholder:
            st.markdown(
                f'<div style="text-align:center; color:#555; padding:20px; border:1px dashed #444; border-radius:10px;">Best shot will appear here</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="section-header">{ICON_HISTORY} <span>Session History</span></div>',
                unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No recorded sessions yet.")
    else:
        for session in reversed(st.session_state.history):
            with st.container():
                c_img, c_info, c_action = st.columns([1, 3, 1])
                with c_img:
                    if session.get('best_detection') and os.path.exists(session['best_detection']['path']):
                        st.image(session['best_detection']['path'], use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="height:80px; background:rgba(0,204,255,0.05); border:1px dashed #00CCFF; border-radius:4px; display:flex; align-items:center; justify-content:center; color:#555;">No Image</div>',
                            unsafe_allow_html=True)
                with c_info:
                    det_count = len(session['df']) if session['df'] is not None else 0
                    best_conf_txt = f"{session['best_detection']['conf']:.1%}" if session.get(
                        'best_detection') else "N/A"
                    st.markdown(
                        f"""<div class="history-card"><div class="history-header"><span>SESSION LOG #{session['id']:02d}</span><span style="color:#FFF;">{session['duration']:.1f}s</span></div><div class="hist-meta"><i class="bi bi-calendar"></i> {session['timestamp']} &nbsp;|&nbsp; <i class="bi bi-trophy"></i> Best: {best_conf_txt} &nbsp;|&nbsp;<i class="bi bi-box"></i> {det_count} Detections</div></div>""",
                        unsafe_allow_html=True)
                with c_action:
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_pdf, col_view = st.columns(2)
                    with col_pdf:
                        if st.button("PDF", key=f"btn_pdf_{session['id']}", use_container_width=True):
                            with st.spinner("Generating Report..."):
                                pdf_bytes, fname = generate_session_report(session)
                                if pdf_bytes: st.download_button("Download", pdf_bytes, fname, "application/pdf",
                                                                 key=f"dl_{session['id']}", use_container_width=True)
                    with col_view:
                        if st.button("View", key=f"btn_view_{session['id']}", use_container_width=True):
                            view_history_popup(session)

    st.markdown("<div style='margin-bottom: 60px;'></div>", unsafe_allow_html=True)
    render_footer()


if __name__ == "__main__":
    main()