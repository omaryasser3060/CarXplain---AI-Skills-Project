import streamlit as st
import os
from navbar.navbar import render_navbar
from footer.footer import render_footer


ICON_SPEEDOMETER_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="#00CCFF" viewBox="0 0 16 16">
  <path d="M8 4a.5.5 0 0 1 .5.5V6a.5.5 0 0 1-1 0V4.5A.5.5 0 0 1 8 4zM3.732 5.732a.5.5 0 0 1 .707 0l.915.914a.5.5 0 1 1-.707.707l-.914-.915a.5.5 0 0 1 0-.707zM2 10a.5.5 0 0 1 .5-.5h1.586a.5.5 0 0 1 0 1H2.5A.5.5 0 0 1 2 10zm9.5 0a.5.5 0 0 1 .5-.5h1.5a.5.5 0 0 1 0 1H12a.5.5 0 0 1-.5-.5zm.754-4.246a.389.389 0 0 0-.527-.02L7.547 9.31a.91.91 0 1 0 1.302 1.258l3.434-4.297a.389.389 0 0 0-.029-.518z"/>
  <path fill-rule="evenodd" d="M0 10a8 8 0 1 1 15.547 2.661c-.442 1.253-1.845 1.602-2.932 1.25C11.309 13.488 9.475 13 8 13c-1.474 0-3.31.488-4.615.911-1.087.352-2.49.003-2.932-1.25A7.988 7.988 0 0 1 0 10zm8-7a7 7 0 0 0-6.603 9.329c.203.575.923.876 1.68.63C4.397 12.533 6.358 12 8 12s3.604.532 4.923.96c.757.245 1.477-.056 1.68-.631A7 7 0 0 0 8 3z"/>
</svg>
"""


icon_path = "app_icon.svg"
with open(icon_path, "w") as f:
    f.write(ICON_SPEEDOMETER_SVG)


st.set_page_config(
    page_title="CarXplain AI",
    page_icon=icon_path,
    layout="wide",
    initial_sidebar_state="collapsed"
)


def local_css(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_name}")

local_css("assets/style.css")


ICON_CAR_HERO = """
<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
  <path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2" />
  <circle cx="7" cy="17" r="2" />
  <circle cx="17" cy="17" r="2" />
  <path d="M5 17h8" />
</svg>
"""
ICON_ANALYSIS = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>"""
ICON_LIVE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>"""
ICON_COMPARE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>"""


def create_team_card(name, role):
    return f"""
    <div class="team-card">
        <div class="team-img" style="display:flex; align-items:center; justify-content:center; font-size:2rem; color:#00CCFF; margin:0 auto 15px auto;">
            {name[0]}
        </div>
        <div class="team-name">{name}</div>
        <div class="team-role">{role}</div>
    </div>
    """


render_navbar()

st.markdown(f"""
<div class="hero-container" style="margin-top: -120px;">
    <div class="hero-icon">{ICON_CAR_HERO}</div>
    <div class="hero-title">CarXplain AI</div>
    <div class="hero-subtitle">Next-Generation Automotive Intelligence</div>
    <p style="color:#888; max-width: 600px; margin: 0 auto 2rem auto; font-size: 1.1rem;">
        Empowering users with state-of-the-art CNN models to detect, classify, and analyze vehicles with Explainable AI precision.
    </p>
    <a href="Image_Analysis" target="_self" style="text-decoration:none;">
        <button style="
            background: linear-gradient(90deg, #00CCFF, #0078FF);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 30px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 0 20px rgba(0, 204, 255, 0.4);
            font-size: 1rem;
        ">
            Start Analysis
        </button>
    </a>
</div>
""", unsafe_allow_html=True)


c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <a href="Image_Analysis" target="_self" style="text-decoration: none; color: inherit;">
        <div class="feature-card">
            <div class="feature-icon">{ICON_ANALYSIS}</div>
            <div class="feature-title">Smart Analysis</div>
            <div class="feature-desc">
                Identify car make & model with high accuracy using InceptionV3 & ResNet50. Includes Grad-CAM visualization.
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <a href="Real_Time" target="_self" style="text-decoration: none; color: inherit;">
        <div class="feature-card">
            <div class="feature-icon">{ICON_LIVE}</div>
            <div class="feature-title">Live Detection</div>
            <div class="feature-desc">
                Real-time vehicle recognition via webcam powered by EfficientNetB4 for low-latency inference.
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <a href="Model_Comparison" target="_self" style="text-decoration: none; color: inherit;">
        <div class="feature-card">
            <div class="feature-icon">{ICON_COMPARE}</div>
            <div class="feature-title">Model Benchmark</div>
            <div class="feature-desc">
                Compare performance metrics across different architectures to select the best model for your needs.
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)


st.markdown('<div class="team-header"><h2>Meet The <span>Minds</span></h2><p style="color:#888;">The HNU Project Team</p></div>', unsafe_allow_html=True)

tc1, tc2, tc3, tc4, tc5 = st.columns(5)
with tc1: st.markdown(create_team_card("Ismail Ibrahim", ""), unsafe_allow_html=True)
with tc2: st.markdown(create_team_card("Youssef Atef", ""), unsafe_allow_html=True)
with tc3: st.markdown(create_team_card("Omar Yasser", ""), unsafe_allow_html=True)
with tc4: st.markdown(create_team_card("Ahmed Ali", ""), unsafe_allow_html=True)
with tc5: st.markdown(create_team_card("Ahmed Ibrahim", ""), unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 60px;'></div>", unsafe_allow_html=True)


render_footer()