import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with heartbeat animation and divider
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #1a1a1a 0%, #2c3e50 100%);
        font-family: 'Arial', sans-serif;
        color: #ecf0f1;
    }
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .sub-header {
        font-size: 2rem;
        color: #ecf0f1;
        border-bottom: 3px solid #e74c3c;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(231, 76, 60, 0.5);
    }
    .info-box, .success-box, .warning-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #1a1a1a;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        animation: slideIn 0.5s ease;
    }
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .info-box { background-color: #e3f2fd; border-left: 6px solid #2196f3; }
    .success-box { background-color: #e8f5e9; border-left: 6px solid #4caf50; }
    .warning-box { background-color: #fff8e1; border-left: 6px solid #ff9800; }
    .stButton>button {
        background-color: #e74c3c;
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #c0392b;
        transform: scale(1.05);
    }
    .stSelectbox, .stSlider, .stMultiselect {
        width: 100% !important;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: rgba(255, 255, 255, 0.05);
        border: none;
        color: #ecf0f1;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #e74c3c;
        color: white;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #c0392b;
    }
    .summary-card {
        background: rgba(76, 175, 80, 0.1);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 5px solid #4caf50;
        transition: transform 0.3s ease;
    }
    .summary-card:hover {
        transform: translateY(-5px);
    }
    .sidebar .sidebar-content {
        background: rgba(44, 62, 80, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    .sidebar .sidebar-content h2 {
        color: #e74c3c;
        text-align: center;
    }
    .stRadio > label {
        color: #ecf0f1;
        font-weight: bold;
        transition: color 0.3s ease;
    }
    .stRadio > label:hover {
        color: #e74c3c;
    }
    .stExpander {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stExpander > div > div {
        color: #ecf0f1;
        font-weight: bold;
    }
    .stExpander > div > div:hover {
        color: #e74c3c;
    }
    .stExpander > div > div > div > p {
        color: #ecf0f1;
    }
    .heartbeat {
        display: inline-block;
        font-size: 2rem;
        color: #e74c3c;
        animation: beat 1s infinite;
    }
    @keyframes beat {
        0% { transform: scale(1); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    .divider {
        border-top: 1px solid #ecf0f1;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.write("This application helps predict the risk of heart disease using advanced machine learning techniques.")

# Define expected columns
expected_columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Sidebar Navigation
with st.sidebar:
    st.title("Navigation")
    app_mode = st.radio("Select a section",
                        ["📊 Dashboard", "💡 Prediction", "📝 Data Upload",
                         "📋 Prediction Inputs Guide", "ℹ️ About"])

# Load or create pipeline
@st.cache_resource
def load_or_create_pipeline():
    try:
        return joblib.load("heart_disease_model_pipeline.pkl")
    except FileNotFoundError:
        st.warning("Model pipeline not found. Please run the training process first.")
        return None

pipeline = load_or_create_pipeline()

# Function to load datasets
@st.cache_data
def load_heart_disease_datasets():
    dataset_urls = {
        "Cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "Switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
        "Hungary": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
        "VA Long Beach": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
    }
    
    datasets = []
    for name, url in dataset_urls.items():
        try:
            df = pd.read_csv(url, header=None, na_values='?')
            df.columns = expected_columns + ["target"]
            df["source"] = name
            datasets.append(df)
        except Exception as e:
            st.warning(f"Could not load {name} dataset: {str(e)}")
    
    if datasets:
        combined_data = pd.concat(datasets, ignore_index=True)
        combined_data["target"] = combined_data["target"].apply(lambda x: 0 if x == 0 else 1)
        return combined_data
    else:
        return None

# Function to process data and train model
def process_and_train_model(data):
    for col in expected_columns:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")
    
    data_cleaned = data.copy()
    data_cleaned = data_cleaned.fillna(data_cleaned.mean(numeric_only=True))
    
    X = data_cleaned[expected_columns]
    y = data_cleaned["target"]
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    pca = PCA(n_components=0.95)
    ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingClassifier(random_state=42))
        ],
        voting='soft'
    )
    pipeline = Pipeline([
        ('scaler', scaler),
        ('pca', pca),
        ('classifier', ensemble)
    ])
    
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, "heart_disease_model_pipeline.pkl")
    
    data_processed = data.copy()
    data_processed["Prediction"] = pipeline.predict(X)
    data_processed["Probability of Heart Disease"] = pipeline.predict_proba(X)[:, 1]
    data_processed["Prediction"] = data_processed["Prediction"].map({0: "Low Risk", 1: "High Risk"})
    
    X_scaled = scaler.fit_transform(X)
    X_pca = pca.fit_transform(X_scaled)
    pca_data = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_data["target"] = y.values
    
    return pipeline, data_processed, X_test, y_test, pca_data

# Dashboard
if app_mode == "📊 Dashboard":
    st.markdown("<h2 class='sub-header'>Heart Health Dashboard</h2>", unsafe_allow_html=True)
    st.image("https://static.vecteezy.com/system/resources/previews/006/763/421/large_2x/cardiology-concept-banner-wireframe-low-poly-style-red-heart-vector.jpg", width=1030, caption="Heart Health Visualization")

    if "uploaded_data" not in st.session_state:
        st.markdown("### Loading Default Datasets")
        data = load_heart_disease_datasets()
        if data is not None:
            st.session_state["raw_data"] = data
            st.markdown("<div class='success-box'>Successfully loaded {} records from 4 datasets.</div>".format(data.shape[0]), unsafe_allow_html=True)
            
            pipeline, data_processed, X_test, y_test, pca_data = process_and_train_model(data)
            st.session_state["pipeline"] = pipeline
            st.session_state["uploaded_data"] = data_processed
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["pca_data"] = pca_data
            st.markdown("<div class='success-box'>Model trained and pipeline saved successfully!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='warning-box'>Failed to load default datasets. Please upload your own data in the Data Upload section.</div>", unsafe_allow_html=True)

    if "uploaded_data" in st.session_state and "pipeline" in st.session_state:
        data = st.session_state["uploaded_data"]
        pipeline = st.session_state["pipeline"]
        st.markdown("<div class='success-box'>Data loaded successfully! Visualizations ready!</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3>Filters</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            age_range = st.slider("Filter by Age Range", float(data["age"].min()), float(data["age"].max()),
                                  (float(data["age"].min()), float(data["age"].max())))
        with col2:
            risk_filter = st.multiselect("Filter by Prediction", ["Low Risk", "High Risk"],
                                         default=["Low Risk", "High Risk"])
        with col3:
            source_filter = st.multiselect("Filter by Dataset Source", data["source"].unique(),
                                           default=data["source"].unique())
        filtered_data = data[(data["age"] >= age_range[0]) & (data["age"] <= age_range[1])]
        if risk_filter:
            filtered_data = filtered_data[filtered_data["Prediction"].isin(risk_filter)]
        if source_filter:
            filtered_data = filtered_data[filtered_data["source"].isin(source_filter)]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Prediction Distribution", "📈 Age vs Heart Rate", "📦 Cholesterol Box",
            "🔍 Correlation", "🌐 PCA Visualization", "📉 Source Comparison",
            "📋 Feature Distributions", "📈 Model Performance"
        ])

        with tab1:
            st.markdown("### Prediction Distribution")
            prediction_counts = filtered_data["Prediction"].value_counts()
            fig_bar = px.bar(x=prediction_counts.index, y=prediction_counts.values, title="Prediction Distribution",
                             color=prediction_counts.index, color_discrete_sequence=["#4caf50", "#ff9800"])
            st.plotly_chart(fig_bar)
            st.progress(int(prediction_counts.get("High Risk", 0) / len(filtered_data) * 100) if len(filtered_data) > 0 else 0)
            fig_pie = px.pie(names=prediction_counts.index, values=prediction_counts.values, title="Risk Distribution",
                             color_discrete_sequence=["#4caf50", "#ff9800"])
            st.plotly_chart(fig_pie)

        with tab2:
            st.markdown("### Age vs Maximum Heart Rate by Prediction")
            fig_scatter = px.scatter(filtered_data, x="age", y="thalach", color="Prediction", title="Age vs Maximum Heart Rate",
                                     trendline="ols", hover_data=["Probability of Heart Disease", "source"],
                                     color_discrete_sequence=["#4caf50", "#ff9800"])
            st.plotly_chart(fig_scatter)

        with tab3:
            st.markdown("### Cholesterol Distribution by Prediction")
            fig_box = px.box(filtered_data, x="Prediction", y="chol", title="Cholesterol Levels",
                             color="Prediction", color_discrete_sequence=["#4caf50", "#ff9800"])
            st.plotly_chart(fig_box)

        with tab4:
            st.markdown("### Correlation Heatmap")
            numeric_data = filtered_data[expected_columns].corr()
            fig_heatmap = px.imshow(numeric_data, text_auto=True, aspect="auto", title="Feature Correlations",
                                    color_continuous_scale="YlOrRd")
            st.plotly_chart(fig_heatmap)

        with tab5:
            st.markdown("### PCA Visualization")
            if "pca_data" in st.session_state:
                pca_data = st.session_state["pca_data"]
                fig_pca = px.scatter(pca_data, x="PC1", y="PC2", color="target", title="PCA Visualization",
                                     color_continuous_scale="Viridis",
                                     labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2", "target": "Heart Disease"})
                st.plotly_chart(fig_pca)
            else:
                st.markdown("<div class='warning-box'>PCA data not available.</div>", unsafe_allow_html=True)

        with tab6:
            st.markdown("### Comparison Across Datasets")
            fig_source_target = px.histogram(filtered_data, x="source", color="target",
                                             title="Heart Disease Prevalence by Dataset Source",
                                             color_discrete_sequence=["#4caf50", "#ff9800"])
            st.plotly_chart(fig_source_target)
            fig_source_age = px.box(filtered_data, x="source", y="age", color="target",
                                    title="Age Distribution by Dataset Source",
                                    color_discrete_sequence=["#4caf50", "#ff9800"])
            st.plotly_chart(fig_source_age)

        with tab7:
            st.markdown("### Feature Distributions Across Datasets")
            feature_to_plot = st.selectbox("Select Feature to Visualize", options=expected_columns)
            fig_feature_dist = px.histogram(filtered_data, x=feature_to_plot, color="source", facet_col="source",
                                            title=f"Distribution of {feature_to_plot} by Dataset Source",
                                            color_discrete_sequence=px.colors.qualitative.Plotly)
            st.plotly_chart(fig_feature_dist)

        with tab8:
            st.markdown("### Model Performance Metrics")
            if "X_test" in st.session_state and "y_test" in st.session_state:
                X_test = st.session_state["X_test"]
                y_test = st.session_state["y_test"]
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1]

                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig_cm)

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {roc_auc:.2f})",
                                  labels={"x": "False Positive Rate", "y": "True Positive Rate"})
                fig_roc.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig_roc)

                rf_model = pipeline.named_steps['classifier'].estimators_[0]
                n_components = pipeline.named_steps['pca'].n_components_
                pca_feature_names = [f"PC{i+1}" for i in range(n_components)]
                feature_importance = pd.DataFrame({
                    'feature': pca_feature_names,
                    'importance': rf_model.feature_importances_
                }).sort_values(by='importance', ascending=False)
                fig_fi = px.bar(feature_importance, x='importance', y='feature', title="Feature Importance (PCA Components)",
                                color='importance', color_continuous_scale="YlOrRd")
                st.plotly_chart(fig_fi)

                st.markdown("### Classification Report")
                st.text(classification_report(y_test, y_pred))
            else:
                st.markdown("<div class='warning-box'>Model evaluation data not available.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Detailed Summary")
        col1, col2, col3 = st.columns([1, 1, 1])
        total_records = len(filtered_data)
        high_risk_count = prediction_counts.get("High Risk", 0)
        low_risk_count = prediction_counts.get("Low Risk", 0)
        high_risk_pct = (high_risk_count / total_records * 100) if total_records > 0 else 0
        low_risk_pct = (low_risk_count / total_records * 100) if total_records > 0 else 0
        
        with col1:
            st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
            st.write("**Total Records**:", total_records)
            st.write("**High Risk**:", f"{high_risk_count} ({high_risk_pct:.1f}%)")
            st.write("**Low Risk**:", f"{low_risk_count} ({low_risk_pct:.1f}%)")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
            st.write("**Average Age**:", f"{filtered_data['age'].mean():.1f} years")
            st.write("**Average Cholesterol**:", f"{filtered_data['chol'].mean():.1f} mg/dl")
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
            st.write("**Avg Max Heart Rate**:", f"{filtered_data['thalach'].mean():.1f} bpm")
            st.write("**Avg Oldpeak**:", f"{filtered_data['oldpeak'].mean():.1f}")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Statistical Summary")
        st.dataframe(filtered_data.describe())
        st.markdown("</div>", unsafe_allow_html=True)

# Prediction
elif app_mode == "💡 Prediction":
    st.markdown("<h2 class='sub-header'>Heart Disease Risk Prediction</h2>", unsafe_allow_html=True)
    if pipeline:
        with st.form("prediction_form"):
            col1, col2 = st.columns([1, 1])
            with col1:
                age = st.slider("Age", 18, 100, 50, help="Your age in years.")
                sex = st.selectbox("Sex", ["Female", "Male"], help="Your biological sex.")
                cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
                                  help="Type of chest discomfort.")
                trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 120, help="Blood pressure at rest.")
                chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200, help="Cholesterol level from a blood test.")
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], help="Whether your blood sugar is high after fasting.")
                restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                                       help="Heart electrical activity result.")
            with col2:
                thalach = st.slider("Maximum Heart Rate", 60, 200, 150, help="Highest heart rate during exercise.")
                exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"], help="Chest pain during exercise.")
                oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.2, 1.0, step=0.1, help="Change in heart tracing after exercise.")
                slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"], help="Slope of heart tracing.")
                ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0, help="Number of blocked heart vessels.")
                thal = st.selectbox("Thalassemia", [3, 6, 7], help="Blood disorder type.")
            submit_button = st.form_submit_button("Predict Risk")

        if submit_button:
            input_data = pd.DataFrame([[age, 1 if sex == "Male" else 0, {"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal Pain": 3, "Asymptomatic": 4}[cp],
                                       trestbps, chol, 1 if fbs == "Yes" else 0, {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg],
                                       thalach, 1 if exang == "Yes" else 0, oldpeak, {"Upsloping": 1, "Flat": 2, "Downsloping": 3}[slope], ca, thal]],
                                      columns=expected_columns)
            prediction = pipeline.predict(input_data)[0]
            probability = pipeline.predict_proba(input_data)[0][1]

            components.html("""
            <audio autoplay>
                <source src="https://www.soundjay.com/buttons/sounds/beep-01a.mp3" type="audio/mpeg">
            </audio>
            """, height=0)
            time.sleep(0.5)

            if prediction == 1:
                st.markdown("<div class='warning-box'>Prediction: High Risk of Heart Disease (Probability: {:.2%})</div>".format(probability), unsafe_allow_html=True)
                components.html("""
                <div style="text-align: center;">
                    <span class='heartbeat'>❤️</span>
                </div>
                <audio autoplay>
                    <source src="https://www.soundjay.com/buttons/sounds/beep-07.mp3" type="audio/mpeg">
                </audio>
                """, height=50)
                time.sleep(1.5)
                st.markdown("<div class='info-box'>We're here for you. Consider sharing your concerns with a doctor or a loved one for support.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='success-box'>Prediction: Low Risk of Heart Disease (Probability: {:.2%})</div>".format(probability), unsafe_allow_html=True)
                components.html("""
                <script>
                    setTimeout(function() {
                        document.getElementById('heartbeat-audio').play();
                    }, 0);
                </script>
                <audio id='heartbeat-audio' autoplay>
                    <source src="https://www.soundjay.com/human/sounds/heartbeat-01.mp3" type="audio/mpeg">
                </audio>
                """, height=0)
                st.balloons()
    else:
        st.markdown("<div class='warning-box'>Prediction model not available. Please upload data and train the model.</div>", unsafe_allow_html=True)

# Data Upload and Processing
elif app_mode == "📝 Data Upload":
    st.markdown("<h2 class='sub-header'>Upload and Process Patient Data</h2>", unsafe_allow_html=True)
    
    st.markdown("### Load Default Heart Disease Datasets")
    if st.button("Load Cleveland, Switzerland, Hungary, and VA Long Beach Datasets"):
        data = load_heart_disease_datasets()
        if data is not None:
            st.session_state["raw_data"] = data
            st.markdown("<div class='success-box'>Successfully loaded {} records from 4 datasets.</div>".format(data.shape[0]), unsafe_allow_html=True)
            st.markdown("### Preview of Combined Data")
            st.dataframe(data.head())
        else:
            st.markdown("<div class='warning-box'>Failed to load default datasets. Please upload your own data.</div>", unsafe_allow_html=True)

    st.markdown("### Or Upload Your Own Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            if "source" not in data.columns:
                data["source"] = "Custom"
            if "target" not in data.columns:
                st.markdown("<div class='warning-box'>Uploaded file must include a 'target' column.</div>", unsafe_allow_html=True)
            else:
                st.session_state["raw_data"] = data
                st.markdown("<div class='success-box'>Successfully uploaded {} records.</div>".format(data.shape[0]), unsafe_allow_html=True)
                st.markdown("### Preview of Uploaded Data")
                st.dataframe(data.head())
        except Exception as e:
            st.markdown("<div class='warning-box'>File error: {}</div>".format(str(e)), unsafe_allow_html=True)

    if "raw_data" in st.session_state:
        data = st.session_state["raw_data"]
        missing_columns = [col for col in expected_columns + ["target"] if col not in data.columns]
        if missing_columns:
            st.markdown("<div class='warning-box'>The dataset is missing: {}. Please include: {}</div>".format(', '.join(missing_columns), ', '.join(expected_columns + ["target"])), unsafe_allow_html=True)
        else:
            for col in expected_columns:
                if data[col].dtype not in [np.float64, np.int64] and not pd.api.types.is_numeric_dtype(data[col]):
                    st.markdown("<div class='warning-box'>Column {} should be numeric. Current type: {}</div>".format(col, data[col].dtype), unsafe_allow_html=True)
                    if st.button(f"Convert {col} to numeric (replaces non-numeric with NaN)"):
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        st.session_state["raw_data"] = data

            st.markdown("### Data Cleaning Options")
            impute_missing = st.checkbox("Impute missing values with averages", value=True)
            if impute_missing:
                data = data.fillna(data.mean(numeric_only=True))
            else:
                data = data.dropna()
            st.session_state["raw_data"] = data

            if st.button("Process with PCA and Train Model"):
                try:
                    pipeline, data_processed, X_test, y_test, pca_data = process_and_train_model(data)
                    st.session_state["pipeline"] = pipeline
                    st.session_state["uploaded_data"] = data_processed
                    st.session_state["X_test"] = X_test
                    st.session_state["y_test"] = y_test
                    st.session_state["pca_data"] = pca_data

                    st.markdown("<div class='success-box'>Model trained and pipeline saved successfully!</div>", unsafe_allow_html=True)
                    st.markdown("### Processed Data with Predictions")
                    st.dataframe(data_processed)
                except Exception as e:
                    st.markdown("<div class='warning-box'>Error processing data: {}</div>".format(str(e)), unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>No data loaded yet. Please load default datasets or upload your own.</div>", unsafe_allow_html=True)

# Prediction Inputs Guide
elif app_mode == "📋 Prediction Inputs Guide":
    st.markdown("<h2 class='sub-header'>Prediction Inputs Guide</h2>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>This guide is for everyone! It explains each piece of information you need to predict your heart disease risk, what it means in simple terms, and how to find it—no medical degree required!</div>", unsafe_allow_html=True)

    with st.expander("📏 Age"):
        st.markdown("""
        <h3>What it is</h3>
        <p>Your age in years—the total number of years you have lived since birth. This is a key demographic factor in medical assessments.</p>
        <h3>Why it matters</h3>
        <p>Age is a significant risk factor for heart disease. As you get older, your heart and blood vessels experience natural wear and tear, which can lead to:</p>
        <ul>
            <li><b>Hardening or narrowing of arteries</b> (atherosclerosis)</li>
            <li><b>Increased blood pressure</b></li>
            <li><b>Reduced heart efficiency</b></li>
        </ul>
        <p>Research shows that individuals above <b>45 years (men)</b> and <b>55 years (women)</b> have a higher likelihood of developing heart-related conditions.</p>
        <h3>How to find it</h3>
        <p>You can determine your age using official documents such as:</p>
        <ul>
            <li>✅ Birth certificate</li>
            <li>✅ Passport or driver’s license</li>
            <li>✅ Government-issued ID</li>
        </ul>
        <p>Alternatively, calculate it manually:</p>
        <p>📌 <b>Formula:</b> <code>Current Year - Your Birth Year</code></p>
        <h3>Example</h3>
        <p>If you were born in <b>1980</b> and the current year is <b>2025</b>:</p>
        <p>👉 <code>2025 - 1980 = 45</code></p>
        <p>Enter <b>45</b> in the input field.</p>
        """, unsafe_allow_html=True)

    with st.expander("👤 Sex"):
        st.markdown("""
        <h3>What it is</h3>
        <p>Your biological sex, categorized as <b>Male</b> or <b>Female</b>, based on physical and genetic characteristics.</p>
        <h3>Why it matters</h3>
        <p>Biological sex plays a crucial role in heart disease risk due to differences in:</p>
        <ul>
            <li><b>Hormones:</b> Estrogen in women may offer some protection against heart disease before menopause, while testosterone in men can influence cholesterol levels.</li>
            <li><b>Heart Structure & Function:</b> Men's hearts are typically larger, and their arteries may stiffen earlier, while women’s arteries tend to be smaller and more flexible.</li>
            <li><b>Symptoms & Risk Factors:</b> Women often experience atypical symptoms (such as nausea and fatigue) during heart attacks, while men are more likely to have classic chest pain.</li>
        </ul>
        <p>On average:</p>
        <ul>
            <li><b>Men</b> have a higher risk of heart disease at an earlier age.</li>
            <li><b>Women</b> may develop heart disease later, but their symptoms can be harder to diagnose.</li>
        </ul>
        <h3>How to determine your sex</h3>
        <p>Your biological sex is typically listed on official documents such as:</p>
        <ul>
            <li>✅ Government-issued ID (e.g., Passport, Driver’s License)</li>
            <li>✅ Birth Certificate</li>
            <li>✅ Medical Records</li>
        </ul>
        <h3>Example Selection</h3>
        <p>If your document says <b>‘M’</b>, select <b>Male</b>.<br>If your document says <b>‘F’</b>, select <b>Female</b>.</p>
        """, unsafe_allow_html=True)

    with st.expander("💔 Chest Pain Type"):
        st.markdown("""
        <h3>What it is</h3>
        <p>The type of discomfort or pain you experience in your chest, if any. Chest pain can vary in sensation, intensity, and cause.</p>
        <h3>Why it matters</h3>
        <p>Chest pain is a key indicator of heart health. Certain types of chest pain can signal heart disease, especially if they occur during physical activity or emotional stress. Identifying the correct type helps in accurate diagnosis and treatment.</p>
        <h3>Options</h3>
        <ul>
            <li><b>🩺 Typical Angina:</b> A <b>pressure, squeezing, or tightness</b> in the chest that often occurs during exercise or stress and improves with rest or medication.</li>
            <li><b>⚠️ Atypical Angina:</b> Unusual chest pain that may feel like burning, stabbing, or discomfort in areas other than the chest. It can happen at rest or without a clear trigger.</li>
            <li><b>💡 Non-Anginal Pain:</b> Chest discomfort <b>not related to the heart</b>, such as pain from muscle strain, acid reflux, or anxiety.</li>
            <li><b>✅ Asymptomatic:</b> No chest pain or discomfort at all.</li>
        </ul>
        <h3>How to determine your chest pain type</h3>
        <p>Think about any chest discomfort you've had recently. Consider:</p>
        <ul>
            <li>✔️ When does the pain happen? (During exercise, at rest, randomly?)</li>
            <li>✔️ How does it feel? (Tight, sharp, burning, mild?)</li>
            <li>✔️ Does it get better with rest or medication?</li>
        </ul>
        <p>If you are unsure, consult a doctor or check past medical records.</p>
        <h3>Example Selection</h3>
        <p>If you feel <b>tight chest pain when climbing stairs</b>, choose <b>Typical Angina</b>.<br>If you have <b>random chest discomfort that feels different each time</b>, choose <b>Atypical Angina</b>.<br>If you experience <b>heartburn or muscle pain</b>, choose <b>Non-Anginal Pain</b>.<br>If you have <b>no chest pain at all</b>, choose <b>Asymptomatic</b>.</p>
        """, unsafe_allow_html=True)

    with st.expander("🩺 Resting Blood Pressure (mmHg)"):
        st.markdown("""
        <h3>What it is</h3>
        <p>The pressure of your blood against your artery walls while you’re at rest, measured in <b>millimeters of mercury (mmHg)</b>.</p>
        <h3>Why it matters</h3>
        <p>High blood pressure (hypertension) makes your heart work harder, increasing the risk of heart disease, stroke, and other complications.<br>Normal blood pressure is typically around <b>120 mmHg (systolic)</b>, but anything above <b>130 mmHg</b> may require medical attention.<br>Elevated blood pressure can lead to <b>heart strain</b>, affecting circulation and overall health.</p>
        <h3>How to find it</h3>
        <p>✔️ Use a <b>home blood pressure monitor</b> (look at the first/top number, called systolic pressure).<br>✔️ Check your <b>medical records</b> or the <b>latest doctor’s visit report</b>.<br>✔️ If unsure, visit a <b>pharmacy or clinic</b> where they offer free blood pressure checks.</p>
        <h3>Example</h3>
        <p>If your last reading was <b>130/80 mmHg</b>, enter <b>130</b> (the first/top number).</p>
        """, unsafe_allow_html=True)

    with st.expander("🩺 Cholesterol (mg/dl)"):
        st.markdown("""
        <h3>What it is</h3>
        <p>Cholesterol is a <b>fatty substance (lipid)</b> in your blood that helps build cells but can be harmful in excess. It is measured in <b>milligrams per deciliter (mg/dl)</b>.</p>
        <h3>Why it matters</h3>
        <p>Your body needs some cholesterol, but <b>high levels can cause artery blockages</b>, increasing heart disease risk.<br>There are two main types:<ul><li><b>LDL (bad cholesterol):</b> Can clog arteries if too high.</li><li><b>HDL (good cholesterol):</b> Helps remove bad cholesterol from your blood.</li></ul><b>Total cholesterol below 200 mg/dl</b> is considered healthy, while higher levels may indicate a risk for heart disease.</p>
        <h3>How to find it</h3>
        <p>✔️ Get a <b>cholesterol blood test</b> from a doctor or health clinic.<br>✔️ Ask for your <b>total cholesterol</b> level (combining LDL, HDL, and other fats).<br>✔️ Some home cholesterol test kits are available, but lab tests are more accurate.</p>
        <h3>Example</h3>
        <p>If your test result is <b>220 mg/dl</b>, enter <b>220</b>.</p>
        """, unsafe_allow_html=True)

    with st.expander("🍬 Fasting Blood Sugar (> 120 mg/dl)"):
        st.markdown("""
        <h3>What it is</h3>
        <p>Fasting blood sugar (FBS) measures the <b>amount of glucose (sugar) in your blood</b> after fasting (not eating or drinking anything except water) for <b>8-12 hours</b>.</p>
        <h3>Why it matters</h3>
        <p><b>High blood sugar (above 120 mg/dl)</b> can indicate <b>prediabetes or diabetes</b>.<br><b>Consistently high levels</b> can damage blood vessels, increasing the risk of <b>heart disease, stroke, and kidney problems</b>.<br><b>Healthy fasting blood sugar</b> is typically <b>below 100 mg/dl</b>. A reading between <b>100-125 mg/dl</b> may indicate prediabetes, while <b>126 mg/dl or higher</b> suggests diabetes.</p>
        <h3>How to find it</h3>
        <p>✔️ Get a <b>fasting blood sugar test</b> at a clinic or doctor's office.<br>✔️ Use a <b>home glucose meter</b> (if available) by checking your blood sugar after fasting.<br>✔️ If your fasting blood sugar is <b>above 120 mg/dl</b>, select 'Yes'. If it’s <b>120 mg/dl or below</b>, select 'No'.</p>
        <h3>Example</h3>
        <p>If your test result is <b>130 mg/dl</b> after fasting, choose <b>'Yes'</b>.<br>If your result is <b>90 mg/dl</b>, choose <b>'No'</b>.</p>
        """, unsafe_allow_html=True)

    with st.expander("📈 Resting ECG (Electrocardiogram)"):
        st.markdown("""
        <h3>What it is</h3>
        <p>An <b>electrocardiogram (ECG or EKG)</b> is a medical test that records your heart’s <b>electrical activity</b> while you're at rest. It helps doctors check your heart rhythm, detect irregularities, and identify signs of heart disease.</p>
        <h3>Why it matters</h3>
        <p>A resting ECG can <b>reveal abnormalities</b> in heart function before symptoms appear.<br>Irregular electrical patterns may indicate <b>heart strain, past heart attacks, or heart disease</b>.<br>It’s a quick, painless test that helps assess your <b>heart health</b>.</p>
        <h3>Options</h3>
        <ul>
            <li><b>Normal 🟢</b> – A healthy, steady heart rhythm with no detected issues.</li>
            <li><b>ST-T Wave Abnormality 🟡</b> – Irregular wave patterns that could indicate <b>heart strain, lack of oxygen, or past heart damage</b>.</li>
            <li><b>Left Ventricular Hypertrophy (LVH) 🔴</b> – Thickening of the left side of the heart, often due to <b>high blood pressure or heart disease</b>.</li>
        </ul>
        <h3>How to find it</h3>
        <p>✔️ Check your last <b>ECG report</b> from a doctor or hospital visit.<br>✔️ If you’ve had an ECG, your doctor will classify your results as one of the options above.<br>✔️ If unsure, consult your doctor for a recommendation.</p>
        <h3>Example</h3>
        <p>If your ECG report states <b>'normal sinus rhythm'</b>, select <b>'Normal'</b>.<br>If it mentions <b>'ST-T wave changes'</b>, select <b>'ST-T Wave Abnormality'</b>.<br>If it reports <b>'Left Ventricular Hypertrophy'</b>, select <b>'Left Ventricular Hypertrophy'</b>.</p>
        """, unsafe_allow_html=True)

    with st.expander("❤️ Maximum Heart Rate (BPM)"):
        st.markdown("""
        <h3>What it is</h3>
        <p>Your <b>maximum heart rate (MHR)</b> is the highest number of beats per minute (BPM) your heart can reach during intense physical activity. It reflects your cardiovascular fitness and endurance.</p>
        <h3>Why it matters</h3>
        <p>A <b>higher MHR</b> usually means your heart is working efficiently to pump blood.<br>A <b>lower-than-expected MHR</b> could suggest reduced heart function or poor cardiovascular fitness.<br>Monitoring MHR helps in <b>assessing heart health, exercise intensity, and potential risks</b>.</p>
        <h3>How to find it</h3>
        <p>✔️ <b>Doctor’s stress test:</b> The most accurate way is through a supervised <b>exercise stress test</b> in a medical setting.<br>✔️ <b>Estimation formula:</b> You can also estimate your MHR using the formula: <br><b>220 – your age</b><br>(For example, if you’re 45 years old: <b>220 – 45 = 175 bpm</b>)</p>
        <h3>Example</h3>
        <p>If a <b>stress test</b> measured your max heart rate at <b>160 bpm</b>, enter <b>160</b>.<br>If you’re <b>30 years old</b> and using the formula, your estimated MHR is <b>190 bpm</b>.<br>If unsure, use the <b>formula-based estimate</b> or consult your doctor.</p>
        """, unsafe_allow_html=True)

    with st.expander("🏃 Exercise-Induced Angina (Chest Pain During Activity)"):
        st.markdown("""
        <h3>What it is</h3>
        <p>Exercise-Induced Angina refers to <b>chest pain, tightness, or discomfort</b> that occurs specifically during <b>physical activity</b>, such as walking, running, or climbing stairs.</p>
        <h3>Why it matters</h3>
        <p><b>Possible sign of heart disease:</b> Chest pain during exertion may indicate that your heart isn't getting enough oxygen due to <b>narrowed or blocked arteries</b>.<br><b>Severity can vary:</b> Some people experience mild discomfort, while others feel sharp or squeezing pain that stops them from continuing an activity.<br><b>A warning sign:</b> If left unchecked, it could lead to more serious heart conditions like a heart attack.</p>
        <h3>How to find it</h3>
        <p>✔️ Pay attention to your <b>chest sensations</b> during physical activity.<br>✔️ If you feel <b>tightness, pressure, or pain</b> that eases when you rest, it may be angina.<br>✔️ Consult your doctor if you have experienced these symptoms during exercise or if you’re unsure.</p>
        <h3>Example</h3>
        <p>If you <b>feel chest pain or tightness while jogging</b> but it <b>goes away when you stop</b>, select <b>'Yes'</b>.<br>If you <b>never feel chest pain during exercise</b>, select <b>'No'</b>.<br>If you're unsure, consult a doctor before making a selection.</p>
        """, unsafe_allow_html=True)

    with st.expander("📉 Oldpeak (ST Depression)"):
        st.markdown("""
        <h3>What it is</h3>
        <p>Oldpeak, also known as <b>ST Depression</b>, is a measurement taken from an <b>Electrocardiogram (ECG)</b> during a <b>stress test</b>. It shows how much the ST segment of your heart’s electrical activity <b>drops</b> when you exercise.</p>
        <h3>Why it matters</h3>
        <p><b>Indicator of heart stress:</b> A larger ST Depression may indicate that your heart <b>isn’t getting enough oxygen</b> during physical activity.<br><b>Possible sign of coronary artery disease:</b> It can suggest reduced blood flow due to <b>blocked or narrowed arteries</b>.<br><b>Higher values may signal risk:</b> A depression greater than <b>1 mm</b> is often linked to heart issues, while <b>0 to 1 mm</b> is usually considered normal.</p>
        <h3>How to find it</h3>
        <p>✔️ Check your <b>stress test ECG report</b> for the 'ST Depression' value.<br>✔️ Ask your <b>doctor or cardiologist</b> if you’re unsure about your results.<br>✔️ If you haven’t had an ECG, a <b>safe estimate</b> is around <b>0 to 0.5 mm</b> (normal range).</p>
        <h3>Example</h3>
        <p>If your <b>ECG report</b> states an <b>ST Depression of 2.0</b>, enter <b>2.0</b>.<br>If you <b>haven’t had a test</b>, start with a <b>low estimate (e.g., 0.5)</b>.</p>
        """, unsafe_allow_html=True)

    with st.expander("📈 ST Slope"):
        st.markdown("""
        <h3>What it is</h3>
        <p>ST Slope refers to the <b>direction</b> of the ST segment on an <b>Electrocardiogram (ECG)</b> taken during a <b>stress test</b>. It indicates how your heart reacts under physical exertion.</p>
        <h3>Why it matters</h3>
        <p>A healthy heart usually shows an <b>upward (positive) slope</b>.<br>A <b>flat or downward slope</b> may suggest reduced blood flow to the heart, potentially indicating heart disease.<br>Doctors use this to assess <b>heart strain</b> and possible blockages in coronary arteries.</p>
        <h3>Options & Meaning</h3>
        <ul>
            <li><b>✅ Upsloping:</b> The ECG line <b>rises</b>—usually a <b>normal, healthy sign</b>.</li>
            <li><b>⚠️ Flat:</b> The ECG line is <b>level</b>—may suggest <b>mild heart stress</b>.</li>
            <li><b>🚨 Downsloping:</b> The ECG line <b>drops</b>—often a <b>warning sign</b> of significant heart issues.</li>
        </ul>
        <h3>How to find it</h3>
        <p>✔️ Look at your <b>stress test ECG report</b> under the 'ST Segment' section.<br>✔️ Ask your <b>doctor or cardiologist</b> if you're unsure.<br>✔️ If you <b>haven’t had a test</b>, selecting <b>Upsloping</b> is a safer default.</p>
        <h3>Example</h3>
        <p>If your <b>ECG report</b> mentions a <b>flat slope</b>, select <b>'Flat'</b>.<br>If it shows a <b>downward slope</b>, select <b>'Downsloping'</b>.</p>
        """, unsafe_allow_html=True)

    with st.expander("🩺 Number of Major Vessels (0-3)"):
        st.markdown("""
        <h3>What it is</h3>
        <p>This represents the <b>number of major blood vessels</b> (ranging from 0 to 3) that are <b>narrowed or blocked</b> due to plaque buildup, as seen in a <b>coronary angiogram</b>.</p>
        <h3>Why it matters</h3>
        <p><b>💓 Healthy arteries (0 blockages)</b> allow normal blood flow to the heart.<br><b>⚠️ 1-2 blocked vessels</b> can indicate moderate heart disease risk.<br><b>🚨 3 blocked vessels</b> may signal <b>severe coronary artery disease</b> and a higher chance of heart complications.</p>
        <h3>How to find it</h3>
        <p>✔️ This information comes from a <b>coronary angiogram</b>, a special heart scan using dye and X-rays.<br>✔️ Ask your <b>doctor or cardiologist</b> for your angiogram results.<br>✔️ If you <b>haven’t had a scan</b>, selecting <b>0</b> is a reasonable default.</p>
        <h3>Example</h3>
        <p>If your <b>angiogram report</b> says <b>1 vessel is blocked</b>, enter <b>1</b>.<br>If you've <b>never had a scan</b>, enter <b>0</b> as the safest assumption.</p>
        """, unsafe_allow_html=True)

    with st.expander("🩺 Thalassemia"):
        st.markdown("""
        <h3>What it is</h3>
        <p>Thalassemia is a <b>genetic blood disorder</b> that affects your body's ability to produce healthy red blood cells, which carry oxygen. It can impact your heart if oxygen delivery is reduced.</p>
        <h3>Why it matters</h3>
        <p><b>🩸 Normal (3):</b> No significant issues with oxygen transport.<br><b>⚠️ Fixed Defect (6):</b> A past problem affecting blood flow, possibly permanent damage.<br><b>🔄 Reversible Defect (7):</b> A temporary issue where oxygen supply improves under certain conditions.</p>
        <h3>How to find it</h3>
        <p>✔️ You can determine this from:<ul><li>A <b>thalassemia blood test</b> (common for diagnosing genetic blood conditions)</li><li>A <b>thallium stress test (heart scan)</b>, where doctors check how well blood flows to your heart.</li></ul>✔️ <b>Ask your doctor</b> for your test results and the corresponding code (3, 6, or 7).</p>
        <h3>Example</h3>
        <p>If your doctor says <b>you have no issues</b>, select <b>3 (Normal)</b>.<br>If you <b>don’t know</b>, select <b>3</b> as the safest default.</p>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='success-box'>
        <h3>Tips for Everyone:</h3>
        <ul>
            <li><b>No clue? Ask!</b> Call your doctor or check old health records.</li>
            <li><b>Guessing? Start simple:</b> Use typical values (like 120 for blood pressure) and adjust later.</li>
            <li><b>Try it out:</b> Go to the 'Prediction' section, enter your numbers, and see your risk!</li>
            <li><b>Stay curious:</b> Not sure about a term? Look it up or ask a friend!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# About
elif app_mode == "ℹ️ About":
    st.markdown("<h2 class='sub-header'>💙 About This Application</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <p>Welcome! This application is designed to <b>help assess your heart health risk</b> using smart AI-based predictions.</p>
        <h3>🔍 How It Works</h3>
        <ul>
            <li>✔️ You enter key health details like blood pressure, cholesterol, and heart rate.</li>
            <li>💡 The model analyzes your data based on medical research and past patient records.</li>
            <li>📊 You get an easy-to-understand prediction about potential heart disease risk.</li>
        </ul>
        <h3>💡 Who Can Use It?</h3>
        <p>Anyone! Whether you're tracking your own health, assisting a loved one, or just curious, this tool is built for <b>everyone</b>—no medical expertise needed.</p>
        <h3>⚠️ Disclaimer</h3>
        <p>This tool is <b>not a substitute for professional medical advice</b>. If you're concerned about your heart health, please consult a doctor.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🛠️ How It Was Made"):
        st.markdown("""
        <h3>🔬 The Science Behind It</h3>
        <h4>📊 The Data</h4>
        <p>This tool is powered by real medical records from <b>920 individuals</b> across four locations: <b>Cleveland, Switzerland, Hungary, and VA Long Beach</b>. Each record represents a unique heart health profile, helping us build a well-rounded prediction model.</p>
        <h4>🏥 The Key Health Factors</h4>
        <p>Our model analyzes <b>13 critical health indicators</b>, including:</p>
        <ul>
            <li><b>Age & Gender:</b> Basic demographics influencing heart risk.</li>
            <li><b>Cholesterol & Blood Pressure:</b> Major cardiovascular health markers.</li>
            <li><b>ECG Readings & Exercise Response:</b> Insights into heart function and stress tolerance.</li>
            <li><b>Blood Sugar, Angina & Vessel Blockage:</b> Additional risk factors for heart disease.</li>
        </ul>
        <h4>🤖 The AI Behind the Predictions</h4>
        <ul>
            <li><b>🔍 Data Processing:</b> We use <b>Principal Component Analysis (PCA)</b> to reduce dimensionality and remove noise, ensuring efficient model performance.</li>
            <li><b>🧹 Data Cleaning & Transformation:</b> Outliers and missing values are handled through <b>mean imputation and standardization</b>, improving data quality.</li>
            <li><b>🌲 Smart Machine Learning Models:</b> Our ensemble method blends <b>Random Forest</b> and <b>Gradient Boosting</b>, leveraging strengths from each for highly accurate predictions.</li>
        </ul>
        <h4>📈 Performance & Accuracy</h4>
        <p><b>Accuracy:</b> Our model’s accuracy depends on the trained data—check the "Model Performance" tab for specific metrics.<br><b>Precision & Recall:</b> Balanced to minimize false negatives, ensuring reliable risk assessment.</p>
        <h4>🎯 Our Mission</h4>
        <p>We designed this tool to be <b>fast, accurate, and easy to use</b>, making heart health assessment accessible to everyone. No medical expertise needed—just input your details and get insights instantly!</p>
        """, unsafe_allow_html=True)

    with st.expander("📖 How to Use It"):
        st.markdown("""
        <h3>🚀 Quick Start Guide</h3>
        <h4>🏗️ Step 1 - Load Data</h4>
        <p>Navigate to <b>'Data Upload'</b> to load the default dataset or upload your own medical data.<br>The model will learn from this data, refining its understanding of heart health trends.</p>
        <h4>🎯 Step 2 - Make Predictions</h4>
        <p>Go to <b>'Prediction'</b>, input your health metrics (age, blood pressure, cholesterol, etc.), and get an instant risk analysis.<br>The AI-powered model will assess your details and predict your likelihood of heart disease.</p>
        <h4>📊 Step 3 - Explore Insights</h4>
        <p>Visit the <b>'Dashboard'</b> for interactive visualizations showcasing heart disease trends and risk factors.<br>Analyze how different factors, such as cholesterol levels and exercise-induced angina, impact heart disease risk.</p>
        <h4>🎮 Advanced Exploration</h4>
        <p>Use filters in the <b>Dashboard</b> to compare different age groups, cholesterol ranges, and exercise responses.<br>Gain deeper insights into heart health patterns and refine your understanding of cardiovascular risks.</p>
        """, unsafe_allow_html=True)

    with st.expander("✨ What It Can Do"):
        st.markdown("""
        <ul>
            <li><b>🔍 Smart Risk Assessment:</b> Instantly calculates whether you're at 'Low Risk' or 'High Risk,' along with a confidence score.</li>
            <li><b>📊 Interactive Visuals:</b> Explore your data through <b>bar charts, scatter plots, and heatmaps</b> for a clear understanding of trends.</li>
            <li><b>🎯 Deep Insights:</b> Use filters to analyze patterns based on age, region, or other factors—see how risks vary across different groups.</li>
            <li><b>📖 Built-in Guidance:</b> The <b>Prediction Inputs Guide</b> ensures you know exactly what each health factor means and how it affects your results.</li>
        </ul>
        """, unsafe_allow_html=True)

    with st.expander("📬 Contact the Creator"):
        st.markdown("""
        <h3>👋 Meet the Creator</h3>
        <p>Hi, I’m <b>Manish</b>—a passionate developer who loves blending technology with real-world impact!</p>
        <h3>🔗 Connect With Me</h3>
        <p><b>💻 GitHub:</b> Explore my projects at <a href='https://github.com/rixscx'>github.com/rixscx</a>—contributions and ideas are always welcome!<br><b>📧 Email:</b> Reach out at <a href='mailto:manishp.73codestop@gmail.com'>manishp.73codestop@gmail.com</a> for questions, feedback, or collaborations.</p>
        <h3>💡 Why I Built This</h3>
        <p>I created this tool to make heart health insights accessible to everyone. By combining AI with real medical data, my goal is to empower users with meaningful predictions that are easy to understand and use. Hope you find it helpful!</p>
        """, unsafe_allow_html=True)

    with st.expander("⚠️ Important Note"):
        st.markdown("""
        <h3>🚨 Disclaimer</h3>
        <ul>
            <li><b>For Educational Purposes Only:</b> This tool is designed to explore heart health risks using AI but <b>does not provide medical advice</b>.</li>
            <li><b>Not a Replacement for a Doctor:</b> Always consult a healthcare professional for accurate diagnosis and treatment.</li>
            <li><b>Stay Informed, Stay Safe:</b> Use this as a learning tool, but rely on medical experts for critical health decisions.</li>
        </ul>
        """, unsafe_allow_html=True)