# Heart Disease Prediction App

## Overview
This Streamlit application provides a user-friendly interface for predicting heart disease risk using machine learning. The app allows users to input patient data and receive instant predictions along with risk analysis and personalized recommendations.

## Features
- Interactive form for entering patient data
- Real-time heart disease risk prediction
- Visual risk probability gauge
- Automatic identification of key risk factors
- Personalized recommendations based on prediction results
- Educational content about heart disease
- Visualization of the prediction model workflow

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone this repository or download the files
2. Create and activate a virtual environment (recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages
   ```
   pip install -r requirements.txt
   ```

### Model Setup
Ensure the heart disease prediction model (`heart_disease_model_pipeline.pkl`) is in the same directory as the app.py file.

## Running the App
Run the Streamlit app with:
```
streamlit run app.py
```

The app will open in your default web browser at http://localhost:8501

## Data Source
The model was trained on heart disease datasets from the UCI Machine Learning Repository:
- Cleveland Heart Disease Dataset
- Hungarian Heart Disease Dataset
- Switzerland Heart Disease Dataset
- VA Long Beach Heart Disease Dataset

## Model Information
The prediction system uses an ensemble of multiple machine learning models including:
- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machine
- Neural Network

Feature engineering techniques such as PCA (Principal Component Analysis) are applied to improve prediction accuracy.

## Disclaimer
This application is intended for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.
