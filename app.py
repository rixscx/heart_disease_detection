import streamlit as st
import sys
import subprocess
import importlib

# Function to check if a package is installed
def is_package_installed(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

# Install required packages if not already installed
packages_to_check = [
    "matplotlib",
    "seaborn",
    "sklearn",
    "joblib",
    "plotly"
]

missing_packages = [pkg for pkg in packages_to_check if not is_package_installed(pkg)]

if missing_packages:
    st.warning(f"Installing missing packages: {', '.join(missing_packages)}")
    for package in missing_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    st.success("Packages installed! Rerunning the app...")
    st.experimental_rerun()

# Regular imports
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

# Continue with the rest of your app...
