import streamlit as st
import app_blood_cells
import app_brain_tumor
import app_report

st.set_page_config(page_title="MultiModal Diagnosis AI", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to:", 
    ["📊 Project Report", "🩸 Blood Cell Analysis", "🧠 Brain Tumor Detection"],
    help="Select a page to navigate"
)

if page == "📊 Project Report":
    app_report.app()
elif page == "🩸 Blood Cell Analysis":
    app_blood_cells.app()
elif page == "🧠 Brain Tumor Detection":
    app_brain_tumor.app()

# Add footer
st.sidebar.divider()
st.sidebar.markdown("""
---
**MultiModal Diagnosis AI** v1.0
- YOLOv8 for tumor detection
- GoogLeNet for classification
- GPU-accelerated training
""")
