import streamlit as st
import os

def render_navbar():
    """
    Renders the fixed navbar with correct links to app.py
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_file_path = os.path.join(current_dir, 'navbar.css')

    # 2. تحميل الستايل
    try:
        with open(css_file_path, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f" Error: Could not find CSS file at {css_file_path}")


    navbar_html = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">

<div class="glass-navbar">
<a href="/" target="_self" class="navbar-logo">
<i class="bi bi-speedometer2" style="font-size: 2rem; color: #00CCFF;"></i>
<div class="logo-text">Car<span>Xplain</span></div>
</a>

<div class="nav-links">
<a href="/" target="_self" class="nav-item">
<i class="bi bi-house-door-fill"></i> Home
</a>

<a href="Image_Analysis" target="_self" class="nav-item">
<i class="bi bi-camera-fill"></i> Analysis
</a>

<a href="Real_Time" target="_self" class="nav-item">
<i class="bi bi-broadcast"></i> Live
</a>

<a href="Model_Comparison" target="_self" class="nav-item">
<i class="bi bi-bar-chart-fill"></i> Compare
</a>
</div>
</div>
"""

    st.markdown(navbar_html, unsafe_allow_html=True)