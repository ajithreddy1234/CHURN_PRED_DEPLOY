from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application= Flask(__name__)

@application.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_data = None

    if request.method == "POST":
        # Extract form data
        age = int(request.form.get("age"))
        gender = request.form.get("gender")
        tenure = int(request.form.get("tenure"))
        monthly_charges = float(request.form.get("monthly_charges"))
        contract_type = request.form.get("contract_type")
        internet_service = request.form.get("internet_service")
        tech_support = request.form.get("tech_support")

        # Prepare data
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
        input_data = pred_df.to_dict(orient="records")[0]

        # Predict
        pipeline = PredictPipeline()
        prediction = pipeline.predict(pred_df)
        result = "Churn" if prediction[0] == 1 else "No Churn"

    return render_template("index.html", result=result, input_data=input_data)


if __name__ == "__main__":
    application.run(debug=True)
