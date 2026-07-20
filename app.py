import os
import pickle
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. ตั้งค่าหน้า Streamlit
# ==========================================
st.set_page_config(page_title="X-Ray Prediction App", layout="wide")
st.title("🦴 ระบบวิเคราะห์ภาพ X-Ray (VGG-16 + Final SVM)")
st.markdown("---")

# ==========================================
# 2. กำหนด Transformation (ต้องเหมือนตอนเทรนเป๊ะ)
# ==========================================
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = {0: "✅ Normal (ปกติ)", 1: "⚠️ Pes Planus (เท้าแบน)"}

# ==========================================
# 3. โหลดโมเดลแบบ Cache (VGG-16 และ ffsvm.pkl)
# ==========================================
@st.cache_resource(show_spinner="⏳ กำลังโหลดโมเดล VGG-16 และ SVM...")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. โหลด VGG-16
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1]) # ตัดชั้นสุดท้ายออก
    vgg16 = vgg16.to(device)
    vgg16.eval()
    
    # 2. โหลด SVM ที่คุณเตรียมไว้ (ffsvm.pkl)
    svm_path = "ffsvm.pkl"
    
    if not os.path.exists(svm_path):
        st.error(f"❌ ไม่พบไฟล์โมเดล: `{svm_path}`\nโปรดตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกันกับ app.py")
        st.stop()
        
    with open(svm_path, 'rb') as f:
        svm_clf = pickle.load(f)
        
    return vgg16, svm_clf, device

# โหลดโมเดลทันทีที่เปิดเว็บ
vgg16, svm_clf, device = load_models()
st.sidebar.success(f"✅ โหลดโมเดล `ffsvm.pkl` สำเร็จ!\nDevice: {device}")

# ==========================================
# 4. Main UI: แบ่งเป็น 2 Tabs
# ==========================================
tab1, tab2 = st.tabs(["📸 ทำนายผลรูปภาพเดี่ยว", "📊 ทดสอบกับชุดข้อมูล (Test Set)"])

# ------------------------------------------
# TAB 1: ทำนายผลรูปภาพเดี่ยว (Upload)
# ------------------------------------------
with tab1:
    st.subheader("อัปโหลดภาพ X-Ray เพื่อวิเคราะห์")
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ (.jpg, .png)", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        # แสดงรูปภาพ
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="รูปภาพที่อัปโหลด", use_column_width=True)
            
        with col2:
            with st.spinner("🧠 กำลังวิเคราะห์รูปภาพ..."):
                # 1. Preprocess
                img_tensor = data_transforms(image).unsqueeze(0).to(device)
                
                # 2. Extract Features (VGG-16)
                with torch.no_grad():
                    features = vgg16(img_tensor).cpu().numpy()
                
                # ตรวจสอบ Input ป้องกัน Error จากโมเดล
                expected_features = getattr(svm_clf, "n_features_in_", 4096)
                if expected_features < 4096:
                    st.error(f"⚠️ โมเดล `ffsvm.pkl` ต้องการฟีเจอร์จำนวน {expected_features} ตัว (อาจจำเป็นต้องใช้ RFE Selector ร่วมด้วย)")
                    st.stop()
                
                # 3. Predict (SVM)
                prediction = svm_clf.predict(features)[0]
                
                # แสดงผล
                st.markdown("### ผลการวิเคราะห์:")
                result_text = CLASS_NAMES.get(prediction, f"ไม่ทราบค่า ({prediction})")
                
                if prediction == 0:
                    st.success(result_text)
                else:
                    st.error(result_text)
                    
                # (Optional) แสดง Confidence Score จาก Decision Function
                if hasattr(svm_clf, "decision_function"):
                    decision_score = svm_clf.decision_function(features)[0]
                    st.caption(f"Decision Score: {decision_score:.4f} (ยิ่งห่างจาก 0 ยิ่งมั่นใจ)")

# ------------------------------------------
# TAB 2: ทดสอบกับชุดข้อมูล
# ------------------------------------------
with tab2:
    st.subheader("ประเมินผลโมเดลกับชุดข้อมูล Test")
    
    # ฟังก์ชันช่วยสร้าง Label Map
    def get_label_map(root_dir):
        label_dict = {}
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(root, file))
                        if 'img_name' in df.columns and 'label' in df.columns:
                            for _, row in df.iterrows():
                                base_name = str(row['img_name']).replace('.png', '').replace('.jpg', '')
                                label_dict[base_name] = int(row['label'])
                    except Exception:
                        pass
        return label_dict

    base_dir = st.text_input("Path ของโฟลเดอร์โปรเจกต์:", value=".")
    
    if st.button("🚀 เริ่มประเมินผล"):
        if not os.path.exists(base_dir):
            st.error("ไม่พบโฟลเดอร์ที่ระบุ")
        else:
            with st.spinner("กำลังสแกน CSV และเตรียมข้อมูล..."):
                label_map = get_label_map(base_dir)
                
            class SimpleXRayDataset(Dataset):
                def __init__(self, root_dir, label_dict, transform=None):
                    self.transform = transform
                    self.image_paths = []
                    self.labels = []
                    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
                    if os.path.exists(root_dir):
                        for root, _, files in os.walk(root_dir):
                            for file in files:
                                if file.lower().endswith(valid_exts):
                                    base_name = os.path.splitext(file)[0]
                                    if base_name in label_dict:
                                        self.image_paths.append(os.path.join(root, file))
                                        self.labels.append(label_dict[base_name])
                def __len__(self): return len(self.image_paths)
                def __getitem__(self, idx):
                    image = Image.open(self.image_paths[idx]).convert('RGB')
                    if self.transform: image = self.transform(image)
                    return image, self.labels[idx]

            test_dir = os.path.join(base_dir, 'global_test')
            test_ds = SimpleXRayDataset(test_dir, label_map, transform=data_transforms)
            
            st.info(f"พบรูปภาพใน Test Set ที่ตรงกับ CSV: **{len(test_ds)}** รูป")
            
            if len(test_ds) > 0:
                with st.spinner("กำลังสกัด Feature และทำนายผล..."):
                    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
                    all_preds = []
                    all_labels = []
                    
                    expected_features = getattr(svm_clf, "n_features_in_", 4096)
                    if expected_features < 4096:
                        st.error(f"⚠️ โมเดล SVM ต้องการ Input {expected_features} ตัว ไม่สามารถประเมินผลด้วยข้อมูล 4,096 ฟีเจอร์ได้โดยตรง")
                        st.stop()

                    with torch.no_grad():
                        for images, labels in loader:
                            feats = vgg16(images.to(device)).cpu().numpy()
                            preds = svm_clf.predict(feats)
                            all_preds.extend(preds)
                            all_labels.extend(labels.numpy())
                
                # คำนวณผล
                acc = accuracy_score(all_labels, all_preds)
                report = classification_report(all_labels, all_preds, target_names=['Normal', 'PesPlanus'], output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                
                st.markdown("### 📊 ผลการประเมิน")
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{acc * 100:.2f}%")
                col2.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
            else:
                st.warning("ไม่พบรูปภาพในโฟลเดอร์ `global_test` ที่ตรงกับไฟล์ CSV")