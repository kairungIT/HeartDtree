import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# ============================================================
# สร้างโมเดลอัตโนมัติถ้ายังไม่มี
# ============================================================
@st.cache_resource
def load_or_create_model():
    model_file = 'heart_disease_pipeline.joblib'
    
    # ถ้าไม่มีไฟล์โมเดล ให้สร้างใหม่
    if not os.path.exists(model_file):
        st.warning("⚠️ ไม่พบไฟล์โมเดล กำลังสร้างโมเดลใหม่...")
        
        # โหลดข้อมูล
        df = pd.read_csv('Heart4.csv')
        
        # Preprocessing
        df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
        df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
        
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        num_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        cat_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 
                        'ExerciseAngina', 'ST_Slope']
        
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
        
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(
                random_state=42,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5
            ))
        ])
        
        model_pipeline.fit(X_train, y_train)
        
        # บันทึกโมเดล
        joblib.dump(model_pipeline, model_file)
        st.success("✅ สร้างโมเดลสำเร็จ!")
        
        return model_pipeline
    else:
        # โหลดโมเดลที่มีอยู่แล้ว
        return joblib.load(model_file)

# โหลดโมเดล
model = load_or_create_model()

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8edf3 100%);
    }
    h1 {
        color: #1a365d;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
    }
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================
st.markdown("# 🫀 ระบบทำนายความเสี่ยงโรคหัวใจ")
st.markdown("##### *Heart Disease Prediction System using Decision Tree*")
st.markdown("---")

# ============================================================
# Sidebar - Input Form
# ============================================================
with st.sidebar:
    st.markdown("### 📝 กรอกข้อมูลสุขภาพ")
    st.markdown("---")
    
    st.markdown("#### 👤 ข้อมูลทั่วไป")
    age = st.number_input("อายุ (Age)", min_value=20, max_value=100, value=55, step=1)
    sex = st.selectbox("เพศ (Sex)", options=[1, 0], format_func=lambda x: "👨 ชาย" if x==1 else "👩 หญิง")
    
    st.markdown("#### 💔 อาการ")
    chest_pain = st.selectbox(
        "ประเภทอาการเจ็บหน้าอก",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 
                               3: "Non-anginal Pain", 4: "Asymptomatic"}[x]
    )
    exercise_angina = st.selectbox(
        "เจ็บหน้าอกเมื่อออกกำลังกาย",
        options=[0, 1],
        format_func=lambda x: "❌ ไม่" if x==0 else "✅ ใช่"
    )
    
    st.markdown("#### 📊 ค่าวัดร่างกาย")
    resting_bp = st.number_input("ความดันโลหิตขณะพัก (mm Hg)", min_value=80, max_value=200, value=130)
    cholesterol = st.number_input("คอเลสเตอรอล (mg/dl)", min_value=100, max_value=600, value=240)
    max_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (bpm)", min_value=60, max_value=220, value=150)
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    
    st.markdown("#### 🔬 ผลตรวจอื่นๆ")
    fasting_bs = st.selectbox("น้ำตาลในเลือด > 120 mg/dl", options=[0, 1], format_func=lambda x: "❌ ไม่" if x==0 else "✅ ใช่")
    resting_ecg = st.selectbox(
        "ผล Electrocardiogram",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "ST-T abnormality", 2: "Hypertrophy"}[x]
    )
    st_slope = st.selectbox(
        "ความชันของ ST segment",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x]
    )
    
    st.markdown("---")
    predict_button = st.button("🔮 ทำนายผล", type="primary", use_container_width=True)

# ============================================================
# Main Content
# ============================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📋 ข้อมูลที่กรอก")
    input_data = pd.DataFrame([{
        'Age': age, 'Sex': sex, 'ChestPainType': chest_pain,
        'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
        'MaxHR': max_hr, 'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak, 'ST_Slope': st_slope
    }])
    st.dataframe(input_data.T.rename(columns={0: 'ค่า'}), use_container_width=True)

with col2:
    st.markdown("### 🎯 ผลการทำนาย")
    
    if predict_button:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        if prediction == 1:
            risk_prob = probability[1] * 100
            st.markdown(f"""
            <div class="result-card" style="border-left: 5px solid #e53e3e;">
                <h2 style="color: #e53e3e; margin: 0;">⚠️ พบความเสี่ยง</h2>
                <p style="font-size: 18px; color: #555;">มีความเสี่ยงเป็นโรคหัวใจ</p>
                <hr style="margin: 1rem 0;">
                <p style="font-size: 14px; color: #777;">ระดับความมั่นใจ</p>
                <h1 style="color: #e53e3e; margin: 0;">{risk_prob:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            st.warning("**คำแนะนำ:** ควรปรึกษาแพทย์ผู้เชี่ยวชาญ")
        else:
            safe_prob = probability[0] * 100
            st.markdown(f"""
            <div class="result-card" style="border-left: 5px solid #38a169;">
                <h2 style="color: #38a169; margin: 0;">✅ ไม่พบความเสี่ยง</h2>
                <p style="font-size: 18px; color: #555;">ไม่พบสัญญาณของโรคหัวใจ</p>
                <hr style="margin: 1rem 0;">
                <p style="font-size: 14px; color: #777;">ระดับความมั่นใจ</p>
                <h1 style="color: #38a169; margin: 0;">{safe_prob:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            st.success("**ดีใจด้วย!** รักษาสุขภาพต่อไป")
        
        st.markdown("#### 📊 การกระจายความน่าจะเป็น")
        prob_df = pd.DataFrame({
            'สถานะ': ['ไม่เป็นโรค', 'เป็นโรค'],
            'ความน่าจะเป็น (%)': [probability[0]*100, probability[1]*100]
        })
        st.bar_chart(prob_df.set_index('สถานะ'))
    else:
        st.info("👈 กรุณากรอกข้อมูลและกดปุ่ม **ทำนายผล**")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; padding: 1rem;">
    <p>⚕️ <strong>คำเตือน:</strong> ผลลัพธ์เป็นเพียงการคาดการณ์ทางสถิติ 
    ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้</p>
    <p>🤖 Powered by <strong>Decision Tree</strong> | Built with <strong>Streamlit</strong></p>
</div>
""", unsafe_allow_html=True)