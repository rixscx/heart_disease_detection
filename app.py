import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #d32f2f;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 2rem;
        color: #424242;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #d32f2f;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Main application header
st.markdown("<h1 class='main-header'>Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>This application helps predict the risk of heart disease based on patient data using machine learning algorithms.</div>", unsafe_allow_html=True)

# Create sidebar navigation
with st.sidebar:
    st.image("https://www.heart.org/-/media/Images/Health-Topics/Heart-Attack/Heart-Attack-Lifestyle-Image.jpg", width=300)
    st.title("Navigation")
    
    app_mode = st.radio(
        "Select a section",
        ["📊 Dashboard", "💡 Prediction", "📝 Data Upload", "ℹ️ About"]
    )

# Main application logic based on mode
if app_mode == "📊 Dashboard":
    st.markdown("<h2 class='sub-header'>Heart Health Dashboard</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='success-box'>Explore key heart health indicators and trends</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        st.info("Age distribution visualization would appear here")
    
    with col2:
        st.subheader("Risk Factors")
        st.info("Risk factors visualization would appear here")

elif app_mode == "💡 Prediction":
    st.markdown("<h2 class='sub-header'>Heart Disease Risk Prediction</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-box'>Enter patient data to predict heart disease risk</div>", unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 100, 50)
            gender = st.selectbox("Gender", ["Male", "Female"])
            chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        
        with col2:
            resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        
        with col3:
            rest_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
            max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
            exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        
        submit_button = st.form_submit_button(label="Predict Risk")
    
    if submit_button:
        st.markdown("<div class='success-box'>Prediction results would appear here</div>", unsafe_allow_html=True)
        st.balloons()

elif app_mode == "📝 Data Upload":
    st.markdown("<h2 class='sub-header'>Upload Patient Data</h2>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Upload patient data for batch processing or analysis</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded file containing {data.shape[0]} records with {data.shape[1]} features.")
            
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            st.subheader("Data Analysis Options")
            if st.button("Generate Statistics"):
                st.write(data.describe())
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif app_mode == "ℹ️ About":
    st.markdown("<h2 class='sub-header'>About This Application</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    <p>The Heart Disease Prediction System is a clinical decision support tool designed to help healthcare professionals assess patient risk for heart disease.</p>
    
    <p>This application uses machine learning algorithms trained on historical patient data to identify patterns and predict risk factors for heart disease.</p>
    </div>
    """, unsafe_allow_html=True)
