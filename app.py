import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS - ออกแบบสวยงามเรียบง่าย
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Kanit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8edf3 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        margin: 0.5rem 0 0 0;
    }
    
    .sidebar-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .sidebar-card h4 {
        color: #1a365d;
        margin-top: 0;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-card.danger {
        border-left: 6px solid #e53e3e;
    }
    
    .result-card.success {
        border-left: 6px solid #38a169;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .metric-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8edf3 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load Model & Scaler
# ============================================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        num_imputer = joblib.load('num_imputer.pkl')
        cat_imputer = joblib.load('cat_imputer.pkl')
        metadata = joblib.load('metadata.pkl')
        return model, scaler, encoder, num_imputer, cat_imputer, metadata
    except FileNotFoundError as e:
        st.error(f"❌ ไม่พบไฟล์: {e}")
        st.stop()

model, scaler, encoder, num_imputer, cat_imputer, metadata = load_models()

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🫀 ระบบทำนายความเสี่ยงโรคหัวใจ</h1>
    <p>Heart Disease Prediction System using Decision Tree</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar - Input Form
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-card"><h4>👤 ข้อมูลทั่วไป</h4>', unsafe_allow_html=True)
    age = st.number_input("อายุ (Age)", min_value=20, max_value=100, value=55, step=1)
    sex = st.selectbox("เพศ (Sex)", options=[1, 0], 
                       format_func=lambda x: "👨 ชาย" if x==1 else "👩 หญิง")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card"><h4>💔 อาการ</h4>', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card"><h4>📊 ค่าวัดร่างกาย</h4>', unsafe_allow_html=True)
    resting_bp = st.number_input("ความดันโลหิตขณะพัก (mm Hg)", min_value=80, max_value=200, value=130)
    cholesterol = st.number_input("คอเลสเตอรอล (mg/dl)", min_value=0, max_value=600, value=240)
    max_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (bpm)", min_value=60, max_value=220, value=150)
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card"><h4>🔬 ผลตรวจอื่นๆ</h4>', unsafe_allow_html=True)
    fasting_bs = st.selectbox("น้ำตาลในเลือด > 120 mg/dl", options=[0, 1], 
                              format_func=lambda x: "❌ ไม่" if x==0 else "✅ ใช่")
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
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    predict_button = st.button("🔮 ทำนายผล", type="primary", use_container_width=True)

# ============================================================
# Main Content
# ============================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 📋 ข้อมูลที่กรอก")
    
    input_data = pd.DataFrame([{
        'Age': age, 'Sex': sex, 'ChestPainType': chest_pain,
        'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
        'MaxHR': max_hr, 'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak, 'ST_Slope': st_slope
    }])
    
    # แปลงชื่อคอลัมน์เป็นภาษาไทยสำหรับแสดงผล
    thai_names = {
        'Age': 'อายุ', 'Sex': 'เพศ', 'ChestPainType': 'อาการเจ็บหน้าอก',
        'RestingBP': 'ความดันโลหิต', 'Cholesterol': 'คอเลสเตอรอล',
        'FastingBS': 'น้ำตาลในเลือด', 'RestingECG': 'ผล ECG',
        'MaxHR': 'อัตราการเต้นหัวใจ', 'ExerciseAngina': 'เจ็บหน้าอกเมื่อออกกำลังกาย',
        'Oldpeak': 'ST Depression', 'ST_Slope': 'ความชัน ST'
    }
    
    display_df = input_data.T.rename(index=thai_names, columns={0: 'ค่า'})
    st.dataframe(display_df, use_container_width=True, height=450)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 ผลการทำนาย")
    
    if predict_button:
        # Transform ข้อมูลตามขั้นตอนที่ Train
        # 1. Numerical: Impute + Scale
        num_data = input_data[metadata['num_features']].values
        num_data_imputed = num_imputer.transform(num_data)
        num_data_scaled = scaler.transform(num_data_imputed)
        
        # 2. Categorical: Impute + Encode
        cat_data = input_data[metadata['cat_features']].values
        cat_data_imputed = cat_imputer.transform(cat_data)
        cat_data_encoded = encoder.transform(cat_data_imputed)
        
        # 3. รวม Features
        input_transformed = np.hstack([num_data_scaled, cat_data_encoded])
        
        # 4. ทำนาย
        prediction = model.predict(input_transformed)[0]
        probability = model.predict_proba(input_transformed)[0]
        
        if prediction == 1:
            risk_prob = probability[1] * 100
            st.markdown(f"""
            <div class="result-card danger">
                <h2 style="color: #e53e3e; margin: 0;">⚠️ พบความเสี่ยง</h2>
                <p style="font-size: 18px; color: #555;">มีความเสี่ยงเป็นโรคหัวใจ</p>
                <hr style="margin: 1rem 0;">
                <p style="font-size: 14px; color: #777;">ระดับความมั่นใจ</p>
                <h1 style="color: #e53e3e; margin: 0;">{risk_prob:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            st.warning("**คำแนะนำ:** ควรปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อตรวจวินิจฉัยเพิ่มเติม")
        else:
            safe_prob = probability[0] * 100
            st.markdown(f"""
            <div class="result-card success">
                <h2 style="color: #38a169; margin: 0;">✅ ไม่พบความเสี่ยง</h2>
                <p style="font-size: 18px; color: #555;">ไม่พบสัญญาณของโรคหัวใจ</p>
                <hr style="margin: 1rem 0;">
                <p style="font-size: 14px; color: #777;">ระดับความมั่นใจ</p>
                <h1 style="color: #38a169; margin: 0;">{safe_prob:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            st.success("**ดีใจด้วย!** รักษาสุขภาพต่อไปด้วยการออกกำลังกายและทานอาหารที่มีประโยชน์")
        
        # Progress Bar
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
    <p>⚕️ <strong>คำเตือน:</strong> ผลลัพธ์เป็นเพียงการคาดการณ์ทางสถิติจากโมเดล Machine Learning 
    ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้</p>
    <p>🤖 Powered by <strong>Decision Tree Classifier</strong> | Built with <strong>Streamlit</strong></p>
</div>
""", unsafe_allow_html=True)