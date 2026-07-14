# -*- coding: utf-8 -*-
"""
Wine Classifier Web App - Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ===== ตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="🍷 Wine Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Custom CSS สำหรับความสวยงาม =====
st.markdown("""
<style>
    /* พื้นหลังหลัก */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Card */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Prediction Result */
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-box h2 {
        margin: 0;
        font-size: 2rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ===== โหลดโมเดล =====
@st.cache_resource
def load_model():
    """โหลดโมเดลและ scaler จากไฟล์"""
    try:
        model = joblib.load('model_files/dt_model.pkl')
        scaler = joblib.load('model_files/scaler.pkl')
        features = joblib.load('model_files/feature_names.pkl')
        return model, scaler, features
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์โมเดล กรุณาวางโฟลเดอร์ 'model_files' ในไดเรกทอรีเดียวกัน")
        st.stop()


model, scaler, features = load_model()

# ชื่อคลาสไวน์
WINE_NAMES = {
    0: "🍷 Class 0 - ไวน์ชนิดที่ 1",
    1: "🍷 Class 1 - ไวน์ชนิดที่ 2",
    2: "🍷 Class 2 - ไวน์ชนิดที่ 3"
}

# ===== Sidebar =====
with st.sidebar:
    st.markdown("## 🍷 Wine Classifier")
    st.markdown("---")
    st.markdown("""
    **แอปพลิเคชันจำแนกประเภทไวน์**  
    ใช้โมเดล Decision Tree ในการวิเคราะห์คุณสมบัติทางเคมีของไวน์
    """)
    st.markdown("---")
    st.markdown("### 📊 ข้อมูล")
    st.markdown("- **Dataset:** Wine Dataset")
    st.markdown("- **Model:** Decision Tree")
    st.markdown("- **Features:** 13 คุณสมบัติ")
    st.markdown("- **Classes:** 3 ประเภท")
    st.markdown("---")
    st.markdown("### 🛠️ เทคโนโลยี")
    st.markdown("- Python 3.x")
    st.markdown("- scikit-learn")
    st.markdown("- Streamlit")


# ===== Header =====
st.markdown("""
<div class="main-header">
    <h1>🍷 Wine Quality Classifier</h1>
    <p>ระบบจำแนกประเภทไวน์ด้วย Decision Tree Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ===== Input Form =====
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🔬 ป้อนข้อมูลคุณสมบัติทางเคมี")
    
    # สร้าง input fields สำหรับแต่ละ feature
    input_data = {}
    
    # แบ่ง features เป็น 2 คอลัมน์
    half = len(features) // 2 + 1
    left_features = features[:half]
    right_features = features[half:]
    
    col_left, col_right = st.columns(2)
    
    # ค่าขอบเขตสำหรับแต่ละ feature (จากข้อมูล Wine)
    feature_ranges = {
        'alcohol': (10.0, 15.0, 12.5),
        'malic_acid': (0.5, 5.0, 2.5),
        'ash': (1.3, 3.5, 2.3),
        'alcalinity_of_ash': (10.0, 30.0, 19.0),
        'magnesium': (70.0, 160.0, 100.0),
        'total_phenols': (0.8, 4.0, 2.3),
        'flavanoids': (0.2, 5.0, 2.0),
        'nonflavanoid_phenols': (0.1, 1.0, 0.4),
        'proanthocyanins': (0.4, 4.0, 1.6),
        'color_intensity': (1.5, 17.0, 5.0),
        'hue': (0.3, 1.7, 0.95),
        'od280/od315_of_diluted_wines': (1.2, 4.0, 2.6),
        'proline': (250.0, 1700.0, 750.0)
    }
    
    with col_left:
        for feat in left_features:
            min_v, max_v, default = feature_ranges.get(feat, (0, 10, 5))
            input_data[feat] = st.number_input(
                feat.replace('_', ' ').title(),
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(default),
                step=0.1
            )
    
    with col_right:
        for feat in right_features:
            min_v, max_v, default = feature_ranges.get(feat, (0, 10, 5))
            input_data[feat] = st.number_input(
                feat.replace('_', ' ').title(),
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(default),
                step=0.1
            )
    
    # ปุ่มทำนาย
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 ทำนายผล", use_container_width=True)


with col2:
    st.markdown("### 📋 ข้อมูลที่ป้อน")
    
    # แสดงข้อมูลที่ป้อนในรูปแบบ DataFrame
    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df.T.rename(columns={0: 'ค่า'}), use_container_width=True)
    
    # ปุ่ม Reset
    if st.button("🔄 รีเซ็ตค่า", use_container_width=True):
        st.rerun()


# ===== Prediction =====
if predict_button:
    with st.spinner("🤖 กำลังวิเคราะห์ข้อมูล..."):
        # Transform ข้อมูลด้วย scaler เดียวกันกับตอนฝึก
        input_array = np.array([list(input_data.values())])
        input_scaled = scaler.transform(input_array)
        
        # ทำนาย
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        st.markdown("---")
        
        # แสดงผลการทำนาย
        st.markdown(f"""
        <div class="prediction-box">
            <h2>{WINE_NAMES[prediction]}</h2>
            <p style="margin-top: 0.5rem; font-size: 1.2rem;">
                ความมั่นใจ: {probabilities[prediction]*100:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # แสดงความน่าจะเป็นของแต่ละคลาส
        st.markdown("### 📊 ความน่าจะเป็นของแต่ละคลาส")
        
        prob_df = pd.DataFrame({
            'คลาส': [WINE_NAMES[i] for i in range(3)],
            'ความน่าจะเป็น (%)': [p * 100 for p in probabilities]
        })
        
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        with col_b:
            # Bar chart แสดงความน่าจะเป็น
            chart_df = pd.DataFrame(
                probabilities * 100,
                index=['Class 0', 'Class 1', 'Class 2'],
                columns=['ความน่าจะเป็น (%)']
            )
            st.bar_chart(chart_df, color='#667eea')
        
        # Feature Importance
        st.markdown("---")
        st.markdown("### 🎯 Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        # กรองเฉพาะ features ที่มี importance > 0
        importance_df = importance_df[importance_df['Importance'] > 0]
        
        if len(importance_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'], 
                    color='#667eea', edgecolor='white')
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance จากโมเดล Decision Tree')
            st.pyplot(fig)
        else:
            st.info("โมเดลไม่ได้ใช้ features ใดๆ ในการตัดสินใจ")


# ===== Footer =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎓 พัฒนาเพื่อการศึกษา | Decision Tree Classifier with Streamlit</p>
</div>
""", unsafe_allow_html=True)