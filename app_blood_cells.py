import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def app() :
        
    st.set_page_config(page_title="Blood Cell Classification", layout="centered")
    st.title("🩸 Blood Cell Classification with GoogLeNet")
    
    st.markdown("""
    Upload an image of **blood cells** to classify it into one of the following categories:
    **Benign**, **Pre-B**, **Pro-B**, **Early Pre-B**.
    """)
    
    @st.cache_resource
    def load_model():
        model_path = "models/best_googlenet_model.pth"
    
        model = models.googlenet(pretrained=False, aux_logits=False)
        num_classes = 4
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    model = load_model()
    
    CLASS_NAMES = ['Benign', 'Pre-B', 'Pro-B', 'Early Pre-B']
    
    uploaded_file = st.file_uploader(" Upload a blood cell image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Blood Cell Image", use_column_width=True)
    
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0)
    
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            pred_idx = torch.argmax(probs).item()
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx].item() * 100
    
        st.subheader("🔹 Classification Result")
        st.write(f"**Predicted Class:** {pred_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    
    else:
        st.info(" Please upload a blood cell image to start classification.")
    
    st.markdown("---")
    st.markdown("Made with  using Streamlit and PyTorch.")
    