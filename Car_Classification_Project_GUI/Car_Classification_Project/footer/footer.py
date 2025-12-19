import streamlit as st

def render_footer():
    st.markdown("""
    <style>
        .glass-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            padding: 10px 0;
            background: rgba(2, 12, 26, 0.9);
            backdrop-filter: blur(5px);
            color: #666;
            font-size: 0.8rem;
            z-index: 9999;
            border-top: 1px solid rgba(0, 204, 255, 0.1);
        }
    </style>
    <div class="glass-footer">
        CarXplain AI Project © 2025 | Designed by <span style="color:#00CCFF;">HNU Team</span>
    </div>
    """, unsafe_allow_html=True)