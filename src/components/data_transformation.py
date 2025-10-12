import os
import sys
from dataclasses import dataclass
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "churn_preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Create sklearn transformation pipeline for numerical (scaled), log transformed,
        and one-hot encoded categorical features.
        '''
        try:
            numerical_cols = ["Age"]
            log_transform_cols = ["Tenure"]
            onehot_cols = ["InternetService", "ContractType"]

            num_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])

            log_pipeline = Pipeline([
                ("log", FunctionTransformer(func=np.log1p, validate=True))
            ])

            cat_pipeline = Pipeline([
                ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_cols),
                ("log_pipeline", log_pipeline, log_transform_cols),
                ("cat_pipeline", cat_pipeline, onehot_cols)
            ], remainder="passthrough")

            logging.info(f"Numerical columns: {numerical_cols}")
            logging.info(f"Log-transform columns: {log_transform_cols}")
            logging.info(f"One-hot columns: {onehot_cols}")

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        Read train and test data, map categorical binaries, apply transformations,
        apply SMOTE on training data, save preprocessor, return transformed arrays and path.
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Mapping Gender and TechSupport to binary")

            drop_cols = ["CustomerID", "TotalCharges"]
            target_column = "Churn"

            # Map binary categorical columns explicitly
            for df in [train_df, test_df]:
                if "Gender" in df.columns:
                    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
                    if df["Gender"].isnull().any():
                        raise ValueError("Unexpected values found in 'Gender' column after mapping.")
                if "TechSupport" in df.columns:
                    df["TechSupport"] = df["TechSupport"].map({"Yes": 1, "No": 0})
                    if df["TechSupport"].isnull().any():
                        raise ValueError("Unexpected values found in 'TechSupport' column after mapping.")

                # Convert boolean columns to int
                bool_cols = df.select_dtypes(include='bool').columns
                df[bool_cols] = df[bool_cols].astype(int)

                # Map target "Churn"
                if target_column in df.columns:
                    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})

                # Drop unwanted columns
                df.drop(columns=drop_cols, inplace=True, errors='ignore')

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessor = self.get_data_transformer_object()

            logging.info("Applying transformations on train and test data")

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            logging.info("Applying SMOTE to balance training data")

            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_arr, y_train)

            train_arr = np.c_[X_train_smote, y_train_smote]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Data transformation completed successfully")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)


