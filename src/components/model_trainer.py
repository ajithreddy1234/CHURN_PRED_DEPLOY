from dataclasses import dataclass
import os
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
import sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):  # Correct constructor name here
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            model = CatBoostClassifier(verbose=False)
            model.fit(X_train, y_train)

            # Save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model,
            )

            # Predict and calculate accuracy on test set
            predicted = model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)

            logging.info(f"Model trained with accuracy: {accuracy}")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
