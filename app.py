import streamlit as st
import os
import base64
import importlib

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
    .info-box, .success-box, .warning-box {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .info-box { background-color: #e3f2fd; }
    .success-box { background-color: #e8f5e9; }
    .warning-box { background-color: #fff8e1; }
    .stButton>button {
        background-color: #d32f2f;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>This application helps predict the risk of heart disease using machine learning.</div>", unsafe_allow_html=True)

# Dependency management
dependencies = {
    "visualization": ["matplotlib.pyplot", "seaborn"],
    "ml": ["sklearn.decomposition", "sklearn.preprocessing", "sklearn.ensemble", "joblib"],
    "interactive": ["plotly.express", "plotly.graph_objects"],
    "image": ["PIL.Image"]
}

missing_deps = []
loaded_modules = {}

for category, modules in dependencies.items():
    for module in modules:
        try:
            loaded_modules[module] = importlib.import_module(module)
        except ImportError:
            missing_deps.append(module)

# Sidebar Navigation
with st.sidebar:
    st.image("https://www.heart.org/-/media/Images/Health-Topics/Heart-Attack/Heart-Attack-Lifestyle-Image.jpg", width=300)
    st.title("Navigation")
    app_mode = st.radio("Select a section", ["📊 Dashboard", "💡 Prediction", "📝 Data Upload", "ℹ️ About"])
    if missing_deps:
        st.sidebar.warning("Some features may be limited due to missing dependencies.")

if app_mode == "📊 Dashboard":
    st.markdown("<h2 class='sub-header'>Heart Health Dashboard</h2>", unsafe_allow_html=True)
    if "matplotlib.pyplot" in loaded_modules and "plotly.express" in loaded_modules:
        st.markdown("<div class='success-box'>Explore key heart health indicators and trends</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='warning-box'>Enhanced visualizations unavailable.</div>", unsafe_allow_html=True)

elif app_mode == "💡 Prediction":
    st.markdown("<h2 class='sub-header'>Heart Disease Risk Prediction</h2>", unsafe_allow_html=True)
    if "sklearn.ensemble" in loaded_modules:
        with st.form("prediction_form"):
            age = st.slider("Age", 18, 100, 50)
            gender = st.selectbox("Gender", ["Male", "Female"])
            submit_button = st.form_submit_button(label="Predict Risk")
        if submit_button:
            st.markdown("<div class='success-box'>Prediction results appear here.</div>", unsafe_allow_html=True)
            st.balloons()
    else:
        st.markdown("<div class='warning-box'>Prediction unavailable.</div>", unsafe_allow_html=True)

elif app_mode == "📝 Data Upload":
    st.markdown("<h2 class='sub-header'>Upload Patient Data</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        try:
            data = loaded_modules.get("pandas").read_csv(uploaded_file)
            st.success(f"Successfully uploaded {data.shape[0]} records.")
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif app_mode == "ℹ️ About":
    st.markdown("""
    <h2 class='sub-header'>About This Application</h2>
    <div class='info-box'>
    <p>The Heart Disease Prediction System is a clinical decision support tool.</p>
    <h3>How It Works</h3>
    <ol>
        <li>Enter patient data</li>
        <li>Machine learning analyzes data</li>
        <li>Prediction results displayed</li>
    </ol>
    <h3>Disclaimer</h3>
    <p>This tool is for educational purposes and should not replace medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

if missing_deps:
    st.error(f"Missing dependencies: {', '.join(missing_deps)}")
