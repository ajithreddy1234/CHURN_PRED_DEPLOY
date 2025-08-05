import streamlit as st
import joblib
import numpy as np
import pandas as pd
import xgboost

# Load model and scaler
model = joblib.load("XGBOOSTT.pkl")
scaler = joblib.load("scaler.pkl")
techsupport_encoder = joblib.load("techsupport.pkl")
# Assume get_dummies column structure saved
expected_columns = joblib.load("expected_columns.pkl")  # List of all column names after one-hot encoding

st.title("Churn Prediction App")
st.write("Please enter the details below:")
st.divider()

# Inputs
age = st.number_input("Age", min_value=12, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=130, value=10)
monthlycharge = st.number_input("Monthly Charges", min_value=30.0, max_value=150.0)
contract = st.selectbox("Contract Type", ["Month-to-Month", "One-Year", "Two-Year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No", "Unknown"])
techsupport = st.selectbox("Tech Support", ["Yes", "No"])

# Predict button
if st.button("Predict"):
    gender_encoded = 1 if gender == "Female" else 0
    tech_encoded = techsupport_encoder.transform([techsupport])[0]

    # Prepare base DataFrame
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender_encoded,
        "Tenure": np.log1p(tenure),
        "MonthlyCharges": monthlycharge,
        "TechSupport": tech_encoded,
        "ContractType": contract,
        "InternetService": internet
    }])

    # One-hot encode Contract and Internet
    input_df = pd.get_dummies(input_df, columns=["ContractType", "InternetService"])

    # Add missing columns (to match training data)
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure column order matches model training
    input_df = input_df[expected_columns]

    # Scale
    input_df["Age"] = scaler.transform(input_df[["Age"]])


    # Predict
    prediction = model.predict(input_df)[0]
    result = "Churn" if prediction == 1 else "Not Churn"
    st.success(f"Prediction: {result}")

