import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ✅ Ensure correct file paths
MODEL_PATH = os.path.join(os.getcwd(), "linear_regression.pkl")
SCALER_PATH = os.path.join(os.getcwd(), "scaler.pkl")
ENCODERS_PATH = os.path.join(os.getcwd(), "label_encoders.pkl")

# ✅ Check if files exist before loading
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Make sure `linear_regression.pkl` is uploaded to your GitHub repository.")

if not os.path.exists(SCALER_PATH):
    st.error("❌ Scaler file not found. Make sure `scaler.pkl` is uploaded to GitHub.")

if not os.path.exists(ENCODERS_PATH):
    st.error("❌ Label Encoders file not found. Make sure `label_encoders.pkl` is uploaded to GitHub.")

# ✅ Load the trained model, scaler, and encoders
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

# ✅ Streamlit UI
st.title("IBD Prediction App (Linear Regression)")

st.write("Enter patient details to predict IBD outcome:")

# Create input fields for user input
feature_inputs = {}
for col in ["Feature1", "Feature2"]:  # Replace with actual feature names
    feature_inputs[col] = st.number_input(f"Enter {col}", min_value=0, max_value=100, value=50)

# ✅ Make Prediction
if st.button("Predict"):
    # Convert user input to a DataFrame
    input_df = pd.DataFrame([feature_inputs])

    # Encode categorical features if necessary
    for col, le in label_encoders.items():
        if col in input_df:
            input_df[col] = le.transform(input_df[col])

    # Standardize input features
    input_scaled = scaler.transform(input_df)

    # Predict using the model
    prediction = model.predict(input_scaled)
    
    # Show result
    st.write(f" Predicted IBD Type: {prediction[0]:.2f}")
