import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def main():
    st.title("Churn Prediction")

    # Input widgets for your features
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    tenure = st.number_input("Tenure (months)", min_value=0, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    contract_type = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", options=["Yes", "No"])

    if st.button("Predict"):
        data = CustomData(
            Age=age,
            Gender=gender,
            Tenure=tenure,
            MonthlyCharges=monthly_charges,
            ContractType=contract_type,
            InternetService=internet_service,
            TechSupport=tech_support
        )
        pred_df = data.get_data_as_data_frame()
        st.write("Input Data:")
        st.dataframe(pred_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        label = "Churn" if result[0] == 1 else "No Churn"
        st.success(f"Prediction result: {label}")


if __name__ == '__main__':
    main()
