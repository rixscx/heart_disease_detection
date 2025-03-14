import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import base64
import streamlit as st
import sys
import subprocess

# Add this at the top of your app to debug
st.write("Python Version:", sys.version)
st.write("Pip List:")
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
st.code(result.stdout)

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .card {
        border-radius: 5px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .prediction-positive {
        background-color: #ffcccb;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
        font-weight: bold;
    }
    .prediction-negative {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #0D47A1;
    }
    .sidebar-info {
        font-size: 0.9rem;
    }
    .feature-explanation {
        font-size: 0.85rem;
        color: #555;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model_pipeline.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'heart_disease_model_pipeline.pkl' exists in the app directory.")
        return None

# Helper functions
def get_feature_description():
    return {
        'age': 'Age of the patient in years',
        'sex': 'Sex (1 = male, 0 = female)',
        'cp': 'Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic)',
        'trestbps': 'Resting blood pressure in mm Hg',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
        'restecg': 'Resting electrocardiographic results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes, 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (0: Normal, 1: Fixed defect, 2: Reversible defect)'
    }

def create_feature_input(feature_descriptions):
    input_data = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Demographics & Vitals</p>', unsafe_allow_html=True)
        
        input_data['age'] = st.slider(
            "Age (years)", min_value=20, max_value=100, value=50,
            help=feature_descriptions['age']
        )
        
        input_data['sex'] = st.radio(
            "Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male",
            horizontal=True, help=feature_descriptions['sex']
        )
        
        input_data['trestbps'] = st.slider(
            "Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120,
            help=feature_descriptions['trestbps']
        )
        
        input_data['chol'] = st.slider(
            "Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200,
            help=feature_descriptions['chol']
        )
        
        input_data['fbs'] = st.radio(
            "Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes",
            horizontal=True, help=feature_descriptions['fbs']
        )
        
        input_data['thalach'] = st.slider(
            "Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150,
            help=feature_descriptions['thalach']
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Heart Condition Parameters</p>', unsafe_allow_html=True)
        
        input_data['cp'] = st.selectbox(
            "Chest Pain Type", 
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x],
            help=feature_descriptions['cp']
        )
        
        input_data['restecg'] = st.selectbox(
            "Resting ECG Results", 
            options=[0, 1, 2],
            format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x],
            help=feature_descriptions['restecg']
        )
        
        input_data['exang'] = st.radio(
            "Exercise Induced Angina", options=[0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes",
            horizontal=True, help=feature_descriptions['exang']
        )
        
        input_data['oldpeak'] = st.slider(
            "ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1,
            help=feature_descriptions['oldpeak']
        )
        
        input_data['slope'] = st.selectbox(
            "Slope of Peak Exercise ST Segment", 
            options=[0, 1, 2],
            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
            help=feature_descriptions['slope']
        )
        
        input_data['ca'] = st.selectbox(
            "Number of Major Vessels Colored by Fluoroscopy", 
            options=[0, 1, 2, 3],
            help=feature_descriptions['ca']
        )
        
        input_data['thal'] = st.selectbox(
            "Thalassemia", 
            options=[0, 1, 2],
            format_func=lambda x: ["Normal", "Fixed defect", "Reversible defect"][x],
            help=feature_descriptions['thal']
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    return pd.DataFrame([input_data])

def make_prediction(model, input_data):
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]
        return prediction[0], probability[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def get_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probability of Heart Disease", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1E88E5"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ccffcc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ffcccb'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def show_risk_factors(input_data, feature_descriptions):
    # Define normal ranges and risk thresholds
    risk_factors = {}
    
    # Check age risk
    if input_data['age'][0] > 65:
        risk_factors['Age'] = f"{input_data['age'][0]} (High risk: > 65 years)"
    
    # Check blood pressure
    if input_data['trestbps'][0] >= 140:
        risk_factors['Blood Pressure'] = f"{input_data['trestbps'][0]} mm Hg (High risk: ≥ 140 mm Hg)"
    
    # Check cholesterol
    if input_data['chol'][0] >= 240:
        risk_factors['Cholesterol'] = f"{input_data['chol'][0]} mg/dl (High risk: ≥ 240 mg/dl)"
    
    # Check heart rate
    if input_data['thalach'][0] < 100:
        risk_factors['Max Heart Rate'] = f"{input_data['thalach'][0]} (Risk: < 100 for age)"
    
    # Check chest pain
    if input_data['cp'][0] == 3:
        risk_factors['Chest Pain'] = "Asymptomatic (High risk indicator)"
    
    # Check ST depression
    if input_data['oldpeak'][0] >= 2:
        risk_factors['ST Depression'] = f"{input_data['oldpeak'][0]} (High risk: ≥ 2.0)"
    
    # Check number of vessels
    if input_data['ca'][0] >= 1:
        risk_factors['Major Vessels'] = f"{input_data['ca'][0]} (Risk increases with number)"
    
    if risk_factors:
        st.markdown("### Key Risk Factors Identified")
        for factor, value in risk_factors.items():
            st.warning(f"**{factor}**: {value}")
    else:
        st.success("No major risk factors identified from your inputs.")

def create_recommendation_section(prediction):
    st.markdown("### Recommendations")
    
    if prediction == 1:
        st.markdown("""
        Based on your results, consider the following actions:
        
        1. **Consult a cardiologist** - Schedule an appointment to discuss these results
        2. **Follow-up tests** - Ask about ECG, stress tests, or angiography
        3. **Review lifestyle factors** - Diet, exercise, and stress management
        4. **Medication review** - Bring current medications to your doctor's appointment
        
        _Note: This tool provides guidance but is not a substitute for professional medical advice._
        """)
    else:
        st.markdown("""
        Maintaining heart health is always important:
        
        1. **Regular check-ups** - Continue annual physical exams
        2. **Heart-healthy diet** - Rich in fruits, vegetables, whole grains, and lean proteins
        3. **Regular exercise** - Aim for at least 150 minutes of moderate activity weekly
        4. **Monitor blood pressure and cholesterol** - Keep track of your numbers
        5. **Avoid smoking** - Quit if you currently smoke
        
        _Note: Heart disease risk factors can change over time. Regular screenings are recommended._
        """)

def visualize_input_data(input_data, model):
    st.markdown("### Visualization")
    
    # Extract PCA and scaler from the pipeline if available
    pca = None
    scaler = None
    
    try:
        for name, step in model.named_steps.items():
            if isinstance(step, PCA):
                pca = step
            if isinstance(step, StandardScaler):
                scaler = step
    except:
        pass
    
    # Create a radar chart of the input features
    categories = list(input_data.columns)
    
    # Get min-max values for scaling
    min_vals = {
        'age': 20, 'sex': 0, 'cp': 0, 'trestbps': 90, 'chol': 100,
        'fbs': 0, 'restecg': 0, 'thalach': 60, 'exang': 0,
        'oldpeak': 0, 'slope': 0, 'ca': 0, 'thal': 0
    }
    
    max_vals = {
        'age': 100, 'sex': 1, 'cp': 3, 'trestbps': 200, 'chol': 600,
        'fbs': 1, 'restecg': 2, 'thalach': 220, 'exang': 1,
        'oldpeak': 6, 'slope': 2, 'ca': 3, 'thal': 2
    }
    
    # Normalize values between 0 and 1 for radar chart
    normalized_values = []
    for col in categories:
        val = input_data[col].values[0]
        min_val = min_vals.get(col, 0)
        max_val = max_vals.get(col, 1)
        normalized = (val - min_val) / (max_val - min_val)
        normalized_values.append(normalized)
    
    # Add the first value again to close the loop
    categories.append(categories[0])
    normalized_values.append(normalized_values[0])
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(30, 136, 229, 0.5)',
        line=dict(color='#1E88E5', width=2),
        name='Patient Data'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Feature Profile",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main application
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This application predicts the likelihood of heart disease based on 
        clinical and demographic features. The underlying model was trained on 
        multiple heart disease datasets and uses machine learning to identify patterns 
        associated with heart disease risk.
        
        **Data Sources:**
        - Cleveland Heart Disease Dataset
        - Hungarian Heart Disease Dataset
        - Switzerland Heart Disease Dataset
        - VA Long Beach Heart Disease Dataset
        
        **Model:** Ensemble of multiple classifiers (Random Forest, Gradient Boosting, etc.) 
        with PCA for feature engineering.
        """
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        """
        1. Enter patient information using the form
        2. Click the "Predict" button
        3. View the prediction results and risk analysis
        4. Review recommendations based on prediction
        
        **Note:** This tool is for educational purposes and should not replace 
        professional medical advice.
        """
    )
    
    # Main content
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Get feature descriptions
    feature_descriptions = get_feature_description()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Prediction", "About Heart Disease", "How It Works"])
    
    with tab1:
        st.markdown("### Enter Patient Information")
        
        # Create input form
        input_data = create_feature_input(feature_descriptions)
        
        # Add predict button
        if st.button("Predict", key="predict_button", use_container_width=True):
            # Make prediction
            prediction, probability = make_prediction(model, input_data)
            
            if prediction is not None:
                st.markdown("### Prediction Results")
                
                # Display prediction
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown(
                            f'<div class="prediction-positive">Heart Disease Detected</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-negative">No Heart Disease Detected</div>',
                            unsafe_allow_html=True
                        )
                
                with col2:
                    gauge_chart = get_gauge_chart(probability)
                    st.plotly_chart(gauge_chart, use_container_width=True)
                
                # Display risk factor analysis
                show_risk_factors(input_data, feature_descriptions)
                
                # Create recommendations section
                create_recommendation_section(prediction)
                
                # Visualize input data
                visualize_input_data(input_data, model)
    
    with tab2:
        st.markdown("## About Heart Disease")
        
        st.markdown("""
        ### What is Heart Disease?
        
        Heart disease refers to various conditions that affect the heart's structure and function. 
        The most common type is coronary artery disease, which affects blood flow to the heart and 
        can lead to a heart attack.
        
        ### Key Risk Factors
        
        - **Age:** Risk increases with age
        - **Sex:** Men are generally at higher risk than pre-menopausal women
        - **Family history:** Increased risk if close relatives had heart disease
        - **Smoking:** Damages blood vessels and reduces oxygen in blood
        - **High blood pressure:** Forces heart to work harder, thickening heart muscle
        - **High cholesterol:** Builds up in arteries, restricting blood flow
        - **Diabetes:** Increases risk of heart disease by affecting blood vessels
        - **Obesity:** Linked to higher blood pressure, diabetes risk
        - **Physical inactivity:** Contributes to obesity and weakens heart muscle
        - **Stress:** May contribute to high blood pressure and other risk factors
        
        ### Warning Signs
        
        - Chest pain or discomfort (angina)
        - Shortness of breath
        - Pain in the neck, jaw, throat, upper abdomen, or back
        - Numbness, weakness, coldness in legs or arms
        - Fluttering in chest (palpitations)
        - Racing heartbeat
        - Slow heartbeat
        - Lightheadedness or dizziness
        - Fatigue during physical activity
        
        ### Prevention Strategies
        
        - Regular check-ups with your doctor
        - Heart-healthy diet (Mediterranean or DASH diet)
        - Regular physical activity (150+ minutes/week)
        - Maintain healthy weight
        - Quit smoking
        - Limit alcohol consumption
        - Manage stress
        - Get adequate sleep
        - Control blood pressure and cholesterol
        """)
        
        # Create two columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Create heart disease risk factors chart
            risk_labels = ['Smoking', 'High BP', 'High Cholesterol', 'Obesity', 'Inactivity', 'Diabetes']
            risk_values = [68, 62, 55, 45, 40, 35]
            
            fig = px.bar(
                x=risk_values,
                y=risk_labels,
                orientation='h',
                labels={'x': 'Risk Increase (%)', 'y': 'Risk Factor'},
                title='Impact of Risk Factors on Heart Disease',
                color=risk_values,
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create age distribution chart
            age_groups = ['30-39', '40-49', '50-59', '60-69', '70-79', '80+']
            male_rates = [3, 7, 14, 22, 31, 38]
            female_rates = [1, 4, 8, 15, 24, 32]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=age_groups,
                y=male_rates,
                name='Men',
                marker_color='#1E88E5'
            ))
            
            fig.add_trace(go.Bar(
                x=age_groups,
                y=female_rates,
                name='Women',
                marker_color='#FF4B4B'
            ))
            
            fig.update_layout(
                title='Heart Disease Prevalence by Age and Sex',
                xaxis_title='Age Group',
                yaxis_title='Prevalence (%)',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## How the Prediction Model Works")
        
        st.markdown("""
        ### Data Collection
        
        The model was trained on four datasets from the UCI Machine Learning Repository:
        - Cleveland Heart Disease Dataset
        - Hungarian Heart Disease Dataset
        - Switzerland Heart Disease Dataset
        - VA Long Beach Heart Disease Dataset
        
        Combined, these datasets provide a diverse range of patient profiles and heart disease presentations.
        
        ### Feature Engineering
        
        The model uses 13 key clinical features that have been found to be predictive of heart disease:
        """)
        
        # Create a table showing the features
        feature_df = pd.DataFrame({
            'Feature': feature_descriptions.keys(),
            'Description': feature_descriptions.values()
        })
        
        st.table(feature_df)
        
        st.markdown("""
        ### Model Architecture
        
        The prediction system uses a machine learning pipeline with these components:
        
        1. **Data Preprocessing**: Standardizing numerical features for consistent scale
        
        2. **Dimensionality Reduction**: Principal Component Analysis (PCA) to capture the most important patterns in the data
        
        3. **Classification Model**: An ensemble of multiple algorithms including:
           - Random Forest
           - Gradient Boosting
           - Logistic Regression
           - Support Vector Machine
           - Neural Network
        
        4. **Model Evaluation**: Cross-validation across different datasets to ensure robust performance
        
        ### Performance
        
        The model achieves over 85% accuracy in predicting heart disease presence, with balanced precision and recall to minimize both false positives and false negatives.
        """)
        
        # Create visualization of model architecture
        st.markdown("### Model Pipeline")
        
        # Pipeline visualization using Plotly
        stages = ['Patient Data', 'Preprocessing', 'Feature Engineering', 'Ensemble Model', 'Prediction']
        
        fig = go.Figure(data=[
            go.Scatter(
                x=[1, 2, 3, 4, 5],
                y=[1, 1, 1, 1, 1],
                mode='markers+text',
                marker=dict(size=30, color=['#C5CAE9', '#9FA8DA', '#7986CB', '#5C6BC0', '#3F51B5']),
                text=stages,
                textposition="bottom center"
            )
        ])
        
        # Add arrows connecting the points
        for i in range(1, 5):
            fig.add_annotation(
                x=i+0.5,
                y=1,
                ax=i,
                ay=1,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#5C6BC0'
            )
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=100),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
