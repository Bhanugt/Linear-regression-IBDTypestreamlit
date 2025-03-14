import os
import streamlit as st
import joblib

# ✅ Ensure correct file paths
MODEL_PATH = os.path.join(os.getcwd(), "linear_regression.pkl")
SCALER_PATH = os.path.join(os.getcwd(), "scaler.pkl")
ENCODERS_PATH = os.path.join(os.getcwd(), "label_encoders.pkl")

# ✅ Load model, scaler, and encoders
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    st.success("✅ Model and encoders loaded successfully!")
except FileNotFoundError:
    st.error("❌ One or more files are missing. Please check your GitHub repository.")
