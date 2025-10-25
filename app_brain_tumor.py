import streamlit as st
from ultralytics import YOLO
from PIL import Image

def app(): 
    
        
    st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
    st.title(" Brain Tumor Detection with YOLOv8")
    
    st.markdown("""
    Upload a **brain MRI image** to detect and localize tumors using a pre-trained YOLOv8 model.
    This version uses the **no-augmentation** model.  
    """)
    
    @st.cache_resource
    def load_yolo():
        model_path = "models/yolo_brain_noaug/weights/best.pt"
        model = YOLO(model_path)
        return model
    
    yolo_model = load_yolo()
    
    uploaded_file = st.file_uploader(" Upload a brain MRI image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI", use_column_width=True)
    
        st.subheader("🔹 Detection Result")
        results = yolo_model.predict(source=image, conf=0.25, save=False)
    
        result_img = results[0].plot()        # BGR numpy
        result_img = Image.fromarray(result_img[..., ::-1])  # to RGB
        st.image(result_img, caption="Detected Tumor(s)", use_column_width=True)
    
    else:
        st.info("👆 Please upload a brain MRI image to start detection.")
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and YOLOv8.")
    