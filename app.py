import streamlit as st

st.title("Heart Disease Prediction System")

try:
    # Try importing required packages
    import pandas as pd
    import numpy as np
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        matplotlib_available = True
    except ImportError:
        matplotlib_available = False
        st.error("Matplotlib/Seaborn packages are not available. Some visualizations will not be displayed.")
    
    # Check if sklearn is available
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import joblib
        sklearn_available = True
    except ImportError:
        sklearn_available = False
        st.error("Scikit-learn packages are not available. Prediction functionality will be limited.")
    
    # Check if plotly is available
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        plotly_available = True
    except ImportError:
        plotly_available = False
        st.error("Plotly package is not available. Interactive visualizations will not be displayed.")
    
    # Check for PIL
    try:
        from PIL import Image
        pil_available = True
    except ImportError:
        pil_available = False
    
    import os
    import base64
    
    # Continue with your app, but check availability flags before using packages
    
    # For example, in your visualization section:
    # if matplotlib_available and plotly_available:
    #     # Create and display visualizations
    # else:
    #     st.warning("Some visualization packages are missing. Please contact the app administrator.")
    
except Exception as e:
    st.error(f"Error initializing app: {str(e)}")
    st.info("This app requires several Python packages that appear to be unavailable in this environment. Please contact the app administrator for assistance.")
