import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, r2_score

from src.exception import CustomException


# ---------- Serialization ----------
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path, *, allow_dummy=False):
    """
    Load a pickled object. If allow_dummy=True and file is missing,
    return a tiny dummy object with a .predict(X) method to avoid 500s.
    """
    try:
        if not os.path.exists(file_path):
            if allow_dummy:
                class _DummyModel:
                    def predict(self, X):
                        # Return 1 for each row (treat as "Churn")
                        try:
                            return [1] * len(X)
                        except Exception:
                            return [1]
                return _DummyModel()
            # If you prefer to fail when missing, keep the exception:
            raise FileNotFoundError(f"Missing model file: {file_path}")

        with open(file_path, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        raise CustomException(e, sys)


# ---------- Model evaluation ----------
def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """
    Trains multiple models with GridSearchCV, returns a dict: {model_name: f1_test}
    Assumes binary classification. Change average='weighted' for multi-class.
    """
    try:
        report = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})

            gs = GridSearchCV(model, params, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            best = gs.best_estimator_
            best.fit(X_train, y_train)

            y_train_pred = best.predict(X_train)
            y_test_pred  = best.predict(X_test)

            # Explicit average to avoid surprises. If multi-class, use 'weighted'.
            train_f1 = f1_score(y_train, y_train_pred, average='binary')
            test_f1  = f1_score(y_test,  y_test_pred,  average='binary')

            report[model_name] = test_f1

        return report

    except Exception as e:
        raise CustomException(e, sys)
