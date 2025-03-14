import streamlit as st
import os
import base64

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

# Dependency management with friendly error messages
try:
    # Essential libraries
    import pandas as pd
    import numpy as np
    
    # Optional libraries with graceful fallbacks
    dependencies = {
        "visualization": {
            "name": "Data Visualization",
            "modules": ["matplotlib.pyplot", "seaborn"],
            "available": False,
            "module_objects": {}
        },
        "ml": {
            "name": "Machine Learning",
            "modules": ["sklearn.decomposition", "sklearn.preprocessing", "sklearn.ensemble", "joblib"],
            "available": False,
            "module_objects": {}
        },
        "interactive": {
            "name": "Interactive Visualization",
            "modules": ["plotly.express", "plotly.graph_objects"],
            "available": False,
            "module_objects": {}
        },
        "image": {
            "name": "Image Processing",
            "modules": ["PIL.Image"],
            "available": False,
            "module_objects": {}
        }
    }
    
    # Check and import dependencies
    missing_deps = []
    for category, info in dependencies.items():
        all_available = True
        for module_name in info["modules"]:
            try:
                module_parts = module_name.split('.')
                if len(module_parts) > 1:
                    # Handle nested imports like sklearn.preprocessing
                    base_module = __import__(module_parts[0], fromlist=[module_parts[1]])
                    module = getattr(base_module, module_parts[1])
                else:
                    # Handle direct imports
                    module = __import__(module_name)
                
                # Store the module object for later use
                info["module_objects"][module_name] = module
            except ImportError:
                all_available = False
                missing_deps.append(f"{module_name} (needed for {info['name']})")
        
        info["available"] = all_available
    
    # Create sidebar navigation
    with st.sidebar:
        st.image("https://www.heart.org/-/media/Images/Health-Topics/Heart-Attack/Heart-Attack-Lifestyle-Image.jpg", width=300)
        st.title("Navigation")
        
        app_mode = st.radio(
            "Select a section",
            ["📊 Dashboard", "💡 Prediction", "📝 Data Upload", "ℹ️ About"]
        )
        
        if missing_deps:
            st.sidebar.markdown("### Attention")
            st.sidebar.warning("Some features may be limited due to missing dependencies.")
    
    # Main application logic based on mode
    if app_mode == "📊 Dashboard":
        st.markdown("<h2 class='sub-header'>Heart Health Dashboard</h2>", unsafe_allow_html=True)
        
        if dependencies["visualization"]["available"] and dependencies["interactive"]["available"]:
            st.markdown("<div class='success-box'>Explore key heart health indicators and trends</div>", unsafe_allow_html=True)
            
            # Sample dashboard content
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Age Distribution")
                # Visualization would go here
                st.info("Age distribution visualization would appear here")
            
            with col2:
                st.subheader("Risk Factors")
                # Visualization would go here
                st.info("Risk factors visualization would appear here")
            
        else:
            st.markdown("<div class='warning-box'>Enhanced visualizations are not available. Please contact your administrator to enable full dashboard features.</div>", unsafe_allow_html=True)
            
            # Fallback to basic visualizations
            st.write("Basic statistics would be shown here")
    
    elif app_mode == "💡 Prediction":
        st.markdown("<h2 class='sub-header'>Heart Disease Risk Prediction</h2>", unsafe_allow_html=True)
        
        if dependencies["ml"]["available"]:
            st.markdown("<div class='info-box'>Enter patient data to predict heart disease risk</div>", unsafe_allow_html=True)
            
            # Create input form
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
                # Prediction logic would go here
                st.balloons()
                
        else:
            st.markdown("<div class='warning-box'>Prediction functionality is currently unavailable. Please contact your administrator to enable this feature.</div>", unsafe_allow_html=True)
    
    elif app_mode == "📝 Data Upload":
        st.markdown("<h2 class='sub-header'>Upload Patient Data</h2>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Upload patient data for batch processing or analysis</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"Successfully uploaded file containing {data.shape[0]} records with {data.shape[1]} features.")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Data analysis options
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
        
        <h3>How It Works</h3>
        <ol>
            <li>Enter patient data including demographics, symptoms, and test results</li>
            <li>The system analyzes the data using advanced machine learning models</li>
            <li>A prediction is generated based on similar historical cases</li>
            <li>Results are displayed with interpretable risk factors</li>
        </ol>
        
        <h3>Disclaimer</h3>
        <p>This tool is intended to support clinical decision-making but should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment decisions.</p>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Application Error: {str(e)}")
    st.markdown("""
    <div class='warning-box'>
    <p>The Heart Disease Prediction System encountered an error during initialization. This may be due to missing dependencies or configuration issues.</p>
    
    <p>Please contact your system administrator for assistance.</p>
    </div>
    """, unsafe_allow_html=True)
