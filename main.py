import streamlit as st
import app_blood_cells
import app_brain_tumor

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["App blood cells", "App brain cells"])

if page == "App blood cells":
    app_blood_cells.app()
elif page == "App brain cells":
    app_brain_tumor.app()
