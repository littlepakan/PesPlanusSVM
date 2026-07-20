import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pickle
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Pes Planus Diagnosis System",
    page_icon="👣",
    layout="centered"
)

# Custom Soft/Warm Pastel Styling
st.markdown("""
    <style>
    .main { background-color: #FAF6F0; color: #5D544C; }
    h1, h2, h3 { color: #8E7A6E; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button {
        background-color: #E3D3C4; color: #5D544C; border-radius: 20px;
        border: none; padding: 10px 24px; font-weight: bold;
    }
    .stButton>button:hover { background-color: #D4BFA7; color: #5D544C; }
    .css-10trblm { color: #5D544C; }
    </style>
""", unsafe_allow_html=True)

# --- Model Loader ---
@st.cache_resource
def load_svm_model():
    model_path = "ffsvm.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        st.error(f"Error: Model file '{model_path}' not found in directory.")
        return None

model = load_svm_model()

# --- Image Preprocessing & Feature Extraction ---
def process_xray(image, target_features=300):
    """
    Converts PIL image to grayscale and resizes to match the exact 300 features 
    expected by your trained SVM model.
    """
    # Convert PIL Image to Grayscale numpy array
    img_array = np.array(image.convert('L'))
    
    # 10 x 30 pixels = 300 total flattened features to match ffsvm.pkl
    resized = cv2.resize(img_array, (30, 10)) 
    flattened = resized.flatten().astype(np.float64)
        
    # Standard normal scaling
    flattened_scaled = (flattened - np.mean(flattened)) / (np.std(flattened) + 1e-7)
    
    return np.array([flattened_scaled])

# --- UI Interface ---
st.title("👣 Diagnosis of Pes Planus from X-ray Images")
st.write("Upload a lateral or weight-bearing foot radiographic image to run an automated SVM evaluation.")
st.write("---")

uploaded_file = st.file_uploader("Choose an X-ray image file...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image cleanly
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption='Uploaded Radiograph', use_column_width=True)
        
    with col2:
        st.subheader("Analysis & Diagnosis")
        
        if model is None:
            st.warning("Prediction unavailable: Model configuration missing.")
        else:
            # Check features expected by your unpickled SVC model
            expected_features = getattr(model, "n_features_in_", 300)
            
            with st.spinner("Processing structural features..."):
                try:
                    # Extract features matching the model layout
                    features = process_xray(image, target_features=expected_features)
                    
                    # Make Prediction
                    prediction = model.predict(features)[0]
                    
                    st.markdown("### **Results:**")
                    
                    # Mapping predictions based on structural class codes
                    if prediction == 1:
                        st.error("⚠️ **Pes Planus (Flat Foot) Detected**")
                    else:
                        st.success("✅ **Normal Foot Alignment**")
                        
                    # Safely handle decision boundary score
                    try:
                        score = model.decision_function(features)[0]
                        st.text(f"SVM Boundary Distance Score: {score:.4f}")
                    except Exception:
                        pass
                        
                except Exception as e:
                    st.error(f"Feature Dimension Mismatch error: {e}")
                    st.info(f"Your trained model expects an input matrix size of exactly: **{expected_features} features**.")

st.markdown("---")
st.caption("Disclaimer: This tool is intended as a deep-learning processing simulator and decision support assistant. Actual diagnostic results should be validated by a licensed radiologist.")