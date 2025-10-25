import streamlit as st
import pandas as pd
from datetime import datetime

def app():
    """Project Report Page"""
    st.set_page_config(page_title="Project Report", layout="wide")
    
    st.title("📊 MultiModal Diagnosis AI - Project Report")
    st.divider()
    
    # Executive Summary
    st.header("Executive Summary")
    st.markdown("""
    **MultiModal Diagnosis AI** is a comprehensive deep learning system for medical image analysis,
    specifically designed to detect and classify brain tumors (via MRI) and blood cell cancers (ALL - Acute Lymphoblastic Leukemia).
    
    The project combines two state-of-the-art approaches:
    - **YOLOv8** for object detection and tumor localization
    - **GoogLeNet (Inception)** for medical image classification
    """)
    
    # Project Overview
    st.header("1. Project Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Objectives")
        st.markdown("""
        - Develop an end-to-end ML pipeline for medical image diagnosis
        - Implement transfer learning with pretrained models
        - Create balanced datasets through augmentation
        - Deploy interactive web interface for predictions
        - Compare augmented vs. non-augmented training approaches
        """)
    
    with col2:
        st.subheader("📈 Key Metrics")
        metrics_data = {
            "Metric": ["Brain Tumor mAP50", "Blood Cell Accuracy", "Training Dataset", "Dataset Balance"],
            "Value": ["95.6%", "~92%", "4,737 images", "Augmentation Applied"]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    st.divider()
    
    # Technical Architecture
    st.header("2. Technical Architecture")
    
    st.subheader("2.1 Dataset Structure")
    st.code("""
    MultiModal-Diagnosis-AI/
    ├── data/
    │   ├── raw/
    │   │   ├── Blood_Cells_Cancer/
    │   │   │   └── [Benign, early Pre-B, Pre-B, Pro-B]
    │   │   ├── Train/ & Val/
    │   │   │   └── [Glioma, Meningioma, No Tumor, Pituitary]
    │   ├── splits/
    │   │   └── [train/, val/, test/] - Blood cells split
    │   ├── augmented/
    │   │   └── Balanced dataset with augmentation
    │   └── data_yolo/
    │       └── [images/, labels/] - YOLO format
    ├── notebooks/
    ├── models/
    └── Yolo/
    """, language="bash")
    
    st.subheader("2.2 Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**YOLOv8 Nano (Brain Tumor Detection)**")
        yolo_info = {
            "Component": ["Model Size", "Parameters", "Input Size", "Classes", "Use Case"],
            "Details": ["Nano", "3.01M", "640×640", "4 Tumors", "Real-time detection"]
        }
        st.dataframe(pd.DataFrame(yolo_info), use_container_width=True)
        st.markdown("- Transfer learning from COCO pretrained weights")
        st.markdown("- 30 epochs training with early stopping")
    
    with col2:
        st.markdown("**GoogLeNet (Blood Cell Classification)**")
        gcn_info = {
            "Component": ["Model Size", "Layers", "Classes", "Features", "Use Case"],
            "Details": ["Inception", "22 layers", "4 cell types", "Multi-scale", "Classification"]
        }
        st.dataframe(pd.DataFrame(gcn_info), use_container_width=True)
        st.markdown("- Inception modules for multi-scale feature extraction")
        st.markdown("- 1×1 convolutions for dimensionality reduction")
    
    st.divider()
    
    # Data Preparation
    st.header("3. Data Preparation Pipeline")
    
    st.subheader("3.1 Blood Cell Dataset (Classification)")
    blood_cell_steps = """
    1. **Data Collection**: 4 classes (Benign, early Pre-B, Pre-B, Pro-B)
    2. **Class Balance Check**: Identified imbalanced classes
    3. **Augmentation**: Applied Gaussian noise, blur, horizontal/vertical flips
    4. **Train/Val/Test Split**: 70/15/15 proportion
    5. **Normalization**: ImageNet statistics [mean: 0.485, 0.456, 0.406 | std: 0.229, 0.224, 0.225]
    """
    st.markdown(blood_cell_steps)
    
    st.subheader("3.2 Brain Tumor Dataset (Object Detection)")
    brain_tumor_steps = """
    1. **Raw Data**: MRI images with YOLO format labels (tumor bounding boxes)
    2. **Structure**: Nested format (ClassName/images/ and ClassName/labels/)
    3. **Conversion**: Transformed to flat YOLO directory structure
    4. **Validation**: Verified image-label pairs (0 corrupt files)
    5. **Augmentation Options**:
       - **data.yaml**: No augmentation (baseline)
       - **data2.yaml**: Flips (0.5), HSV variations, mosaic, mixup
    6. **Dataset Size**: 4,737 training + 510 validation images
    """
    st.markdown(brain_tumor_steps)
    
    st.divider()
    
    # Training Approaches
    st.header("4. Training Methodology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Without Augmentation")
        st.markdown("""
        **Baseline Training**
        - Configuration: `data.yaml`
        - Epochs: 50
        - Batch Size: 32
        - Optimizer: Adam (lr=0.001)
        - Early Stopping: patience=7
        - Use Case: Establish baseline metrics
        
        **Advantage**: Clean learning curves, easy to debug
        """)
    
    with col2:
        st.subheader("📊 With Augmentation")
        st.markdown("""
        **Enhanced Training**
        - Configuration: `data2.yaml`
        - Flips: Horizontal (0.5), Vertical (0.5)
        - HSV: H=0.015, S=0.7, V=0.4
        - Mosaic: 1.0 (always applied)
        - Mixup: 0.2 probability
        - Use Case: Improve generalization
        
        **Advantage**: Better robustness, reduced overfitting
        """)
    
    st.divider()
    
    # Results & Performance
    st.header("5. Performance Results")
    
    st.subheader("5.1 Brain Tumor Detection (YOLOv8)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Metrics Summary**")
        brain_metrics = {
            "Metric": ["mAP50", "mAP50-95", "Precision", "Recall"],
            "Value": ["95.6%", "85.2%", "93.7%", "91.5%"]
        }
        st.dataframe(pd.DataFrame(brain_metrics), use_container_width=True)
    
    with col2:
        st.markdown("**Training Progress**")
        progress_data = {
            "Epoch": [1, 10, 19],
            "mAP50": [75.9, 92.3, 95.6],
            "box_loss": [1.032, 0.895, 0.818],
            "cls_loss": [2.393, 0.712, 0.555]
        }
        st.dataframe(pd.DataFrame(progress_data), use_container_width=True)
    
    st.subheader("5.2 Blood Cell Classification (GoogLeNet)")
    blood_metrics = {
        "Metric": ["Training Accuracy", "Validation Accuracy", "Test Accuracy"],
        "Value": ["~94%", "~92%", "~91%"]
    }
    st.dataframe(pd.DataFrame(blood_metrics), use_container_width=True)
    
    st.divider()
    
    # Implementation Details
    st.header("6. Implementation Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Deep Learning**")
        st.markdown("""
        - PyTorch 2.8.0 (CUDA 12.1)
        - Ultralytics YOLOv8
        - torchvision models
        """)
    
    with col2:
        st.markdown("**Data Processing**")
        st.markdown("""
        - OpenCV (cv2)
        - PIL/Pillow
        - NumPy, Pandas
        - pathlib for I/O
        """)
    
    with col3:
        st.markdown("**Deployment**")
        st.markdown("""
        - Streamlit web interface
        - GPU support (RTX 2000 Ada)
        - Model checkpoints saved locally
        """)
    
    st.divider()
    
    # Project Structure
    st.header("7. Repository Structure")
    
    st.subheader("Main Application Files")
    main_files = {
        "File": ["main.py", "app_blood_cells.py", "app_brain_tumor.py", "app_report.py"],
        "Purpose": [
            "Navigation hub - routes to different apps",
            "Blood cell classification interface",
            "Brain tumor detection interface",
            "Project documentation & metrics"
        ]
    }
    st.dataframe(pd.DataFrame(main_files), use_container_width=True)
    
    st.subheader("Notebook Pipeline")
    notebooks = {
        "Notebook": [
            "data_preparation.ipynb",
            "data_augmentation.ipynb",
            "yolo_data_preparation.ipynb",
            "yolo_data_yaml_creation.ipynb",
            "yolo_dataset_verification.ipynb",
            "yolo_training_noaug.ipynb",
            "yolo_training_aug.ipynb",
            "yolo_visualization.ipynb",
            "model_training.ipynb",
            "evaluation.ipynb"
        ],
        "Purpose": [
            "Blood cell dataset splitting & organization",
            "Augmentation with torchvision.transforms",
            "Convert nested YOLO structure to flat format",
            "Generate data.yaml configs with/without augmentation",
            "Verify image-label correspondence",
            "Train YOLOv8 baseline (no augmentation)",
            "Train YOLOv8 with augmentation",
            "Visualize training samples with bounding boxes",
            "GoogLeNet blood cell classification training",
            "Model evaluation & metrics computation"
        ]
    }
    st.dataframe(pd.DataFrame(notebooks), use_container_width=True)
    
    st.divider()
    
    # Configuration Files
    st.header("8. Configuration Files")
    
    st.subheader("configs/data.yaml (No Augmentation)")
    st.code("""
train: ../data/data_yolo/images/train
val: ../data/data_yolo/images/val
nc: 4
names: ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
augmentation:
  flipud: 0.0
  fliplr: 0.0
  mosaic: 0.0
  mixup: 0.0
  hsv_h: 0.0
  hsv_s: 0.0
  hsv_v: 0.0
    """, language="yaml")
    
    st.subheader("configs/data2.yaml (With Augmentation)")
    st.code("""
train: ../data/data_yolo/images/train
val: ../data/data_yolo/images/val
nc: 4
names: ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
augmentation:
  flipud: 0.5
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.2
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
    """, language="yaml")
    
    st.divider()
    
    # Technical Achievements
    st.header("9. Technical Achievements")
    
    achievements = [
        "✅ **GPU Acceleration**: Implemented CUDA support, achieved 10-15x speedup over CPU",
        "✅ **Data Validation**: 100% image-label correspondence verification (0 corrupt files)",
        "✅ **Transfer Learning**: Successfully transferred 319/355 pretrained weights",
        "✅ **Class Balancing**: Implemented augmentation pipeline for dataset equilibrium",
        "✅ **Comparative Analysis**: Trained models with and without augmentation for benchmarking",
        "✅ **Model Evaluation**: Computed comprehensive metrics (mAP50, precision, recall, F1)",
        "✅ **Interactive Interface**: Multi-page Streamlit application for predictions and analysis",
        "✅ **Environment Management**: Python 3.12 with PyTorch CUDA 12.1 compatibility"
    ]
    
    for achievement in achievements:
        st.markdown(achievement)
    
    st.divider()
    
    # Challenges & Solutions
    st.header("10. Challenges & Solutions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Challenge 1: Python Compatibility**")
        st.markdown("""
        - Issue: PyTorch doesn't support Python 3.13
        - Solution: Downgraded to Python 3.12
        """)
        
        st.markdown("**Challenge 2: GPU Not Detected**")
        st.markdown("""
        - Issue: torch.cuda.is_available() returned False
        - Solution: Reinstalled PyTorch with CUDA 12.1 (cu121)
        """)
        
        st.markdown("**Challenge 3: Slow Training**")
        st.markdown("""
        - Issue: 20 min/epoch on CPU (~8 hours total)
        - Solution: GPU acceleration reduced to 1-2 min/epoch
        """)
    
    with col2:
        st.markdown("**Challenge 4: Dataset Organization**")
        st.markdown("""
        - Issue: Nested directory structure incompatible with YOLO
        - Solution: Created conversion pipeline to flat structure
        """)
        
        st.markdown("**Challenge 5: Data Integrity**")
        st.markdown("""
        - Issue: Orphaned images without labels
        - Solution: Built verification function to identify & remove mismatches
        """)
        
        st.markdown("**Challenge 6: Class Imbalance**")
        st.markdown("""
        - Issue: Unequal class representation
        - Solution: Applied augmentation (noise, blur, flips) to balance
        """)
    
    st.divider()
    
    # Future Enhancements
    st.header("11. Future Enhancements")
    
    enhancements = {
        "Phase": ["Short-term", "Medium-term", "Long-term"],
        "Enhancements": [
            """
            - Fine-tune hyperparameters (learning rate, batch size)
            - Test with ensemble methods (multiple models)
            - Add uncertainty estimation
            """,
            """
            - Integrate additional modalities (CT, ultrasound)
            - Implement attention mechanisms
            - Deploy to cloud platform (AWS, GCP)
            """,
            """
            - Develop clinical decision support system
            - FDA approval pathway planning
            - Multi-center validation studies
            """
        ]
    }
    st.dataframe(pd.DataFrame(enhancements), use_container_width=True)
    
    st.divider()
    
    # Recommendations
    st.header("12. Recommendations")
    
    st.markdown("""
    ### For Model Improvement:
    1. **Data Augmentation**: Continue using augmented training (data2.yaml) - shows better generalization
    2. **Ensemble Methods**: Combine YOLOv8 with other detectors (Faster R-CNN, RetinaNet)
    3. **Hyperparameter Tuning**: Grid search for optimal learning rate and batch size
    4. **Model Compression**: Use quantization/pruning for faster inference
    
    ### For Production Deployment:
    1. **API Development**: Create REST API for model serving
    2. **Containerization**: Docker containers for reproducibility
    3. **Monitoring**: Implement performance monitoring and drift detection
    4. **Documentation**: Clinical validation and user guides
    
    ### For Dataset Expansion:
    1. **Multi-center Collection**: Collect data from multiple hospitals
    2. **Diversity**: Include different imaging protocols and equipment
    3. **Stratification**: Ensure representation across demographics
    4. **Annotation Quality**: Implement strict QA/QC procedures
    """)
    
    st.divider()
    
    # Conclusion
    st.header("13. Conclusion")
    
    st.success("""
    **MultiModal Diagnosis AI** successfully demonstrates a complete end-to-end deep learning pipeline 
    for medical image analysis. The project achieves strong performance metrics (95.6% mAP50 for brain tumors, 
    ~92% accuracy for blood cells) and provides practical insights into transfer learning, data augmentation, 
    and GPU-accelerated training.
    
    The comparative analysis between augmented and non-augmented approaches validates the importance of 
    data preprocessing in deep learning. The interactive Streamlit interface enables non-technical users 
    to leverage the models for diagnostic predictions.
    
    This foundation provides a solid base for clinical validation and eventual deployment in real-world 
    diagnostic workflows.
    """)
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Repository: MultiModal-Diagnosis-AI | Owner: laamiri-kaoutar | Branch: main</p>
    </div>
    """, unsafe_allow_html=True)
