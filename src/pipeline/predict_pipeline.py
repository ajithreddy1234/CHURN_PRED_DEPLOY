import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Encode Gender if it's still string
            if "Gender" in features.columns and features["Gender"].dtype == "object":
                features["Gender"] = features["Gender"].map({"Male": 1, "Female": 0})

            # Encode TechSupport if it's still string
            if "TechSupport" in features.columns and features["TechSupport"].dtype == "object":
                features["TechSupport"] = features["TechSupport"].map({"Yes": 1, "No": 0})

            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "churn_preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Age: int,
        Gender: str,
        Tenure: int,
        MonthlyCharges: float,
        ContractType: str,
        InternetService: str,
        TechSupport: str
    ):
        self.Age = Age
        self.Gender = Gender
        self.Tenure = Tenure
        self.MonthlyCharges = MonthlyCharges
        self.ContractType = ContractType
        self.InternetService = InternetService
        self.TechSupport = TechSupport

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Tenure": [self.Tenure],
                "MonthlyCharges": [self.MonthlyCharges],
                "ContractType": [self.ContractType],
                "InternetService": [self.InternetService],
                "TechSupport": [self.TechSupport],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
