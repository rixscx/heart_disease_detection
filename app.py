import streamlit as st
import pickle
import numpy as np
import os

# Load the trained model safely
model_path = os.path.join(os.path.dirname(__file__), "heart_disease_model.pkl")

try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("🚨 Model file not found! Please ensure 'heart_disease_model.pkl' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# Custom Streamlit CSS for enhanced UI
st.markdown("""
    <style>
        body {
            background-color: #0F2027;
            background-image: linear-gradient(to right, #0F2027, #203A43, #2C5364);
            color: white;
        }
        .main {
            background-color: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.2);
        }
        h1 {
            text-align: center;
            color: #00D4FF;
        }
        .stButton>button {
            background: linear-gradient(to right, #FF416C, #FF4B2B);
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            transition: 0.3s;
            font-size: 18px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #FF4B2B, #FF416C);
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Main UI layout
st.markdown("<h1>💓 Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("### 📋 Enter Patient Details")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🔢 Age", min_value=18, max_value=100, value=50)
    sex = st.selectbox("⚧ Sex", ["Male", "Female"])
    cp = st.selectbox("🔥 Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("💉 Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("🩸 Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    restecg = st.selectbox("📊 Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("💓 Max Heart Rate", min_value=60, max_value=220, value=150)
    exang = st.selectbox("🚴 Exercise-Induced Angina", [0, 1])
    oldpeak = st.number_input("📉 ST Depression", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
    slope = st.selectbox("📈 Slope of Peak ST Segment", [0, 1, 2])
    ca = st.number_input("🔬 Major Vessels Colored", min_value=0, max_value=4, value=0)
    thal = st.selectbox("🧬 Thalassemia Type", [0, 1, 2, 3])

# Convert categorical inputs
sex = 1 if sex == "Male" else 0

# Create a NumPy array for prediction
input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Prediction Button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_features)
        result = "🛑 High Risk of Heart Disease" if prediction[0] == 1 else "✅ Low Risk of Heart Disease"
        
        # Display prediction result with styling
        st.markdown(f"""
            <div style="
                background: {'#FF4B2B' if prediction[0] == 1 else '#28a745'};
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                font-size: 20px;
                color: white;
                font-weight: bold;">
                {result}
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"🚨 Prediction Error: {e}")

st.markdown("</div>", unsafe_allow_html=True)
