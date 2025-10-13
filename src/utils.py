import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, r2_score  # keep if used elsewhere

from src.exception import CustomException


# ---------- Serialization ----------
def save_object(file_path: str, obj) -> None:
    """
    Persist any Python object to disk using pickle.
    Creates directories if needed.
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str, *, allow_dummy: bool = False):
    """
    Load a pickled object from disk.

    If allow_dummy=True and file is missing, return a minimal dummy model
    with .predict(X) to prevent runtime 500s during local/dev runs.
    """
    try:
        if not os.path.exists(file_path):
            if allow_dummy:
                class _DummyModel:
                    def predict(self, X):
                        try:
                            return [1] * len(X)
                        except Exception:
                            return [1]
                return _DummyModel()
            raise FileNotFoundError(f"Missing model file: {file_path}")

        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


# ---------- Model evaluation ----------
def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models: dict,
    param: dict,
    *,
    cv: int = 3,
    scoring: str | None = None,
):
    """
    Train multiple models with GridSearchCV.
    Returns a dict: {model_name: f1_test}

    Notes:
    - Assumes binary classification for F1 scoring (average='binary').
      For multi-class, change to average='weighted' or pass a `scoring`.
    - `scoring` lets you provide a sklearn scoring string (e.g., 'f1', 'f1_weighted').
    """
    try:
        report: dict[str, float] = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})

            gs = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=cv,
                n_jobs=-1,
                scoring=scoring,  # None => estimator default; we still compute F1 below
            )
            gs.fit(X_train, y_train)

            best = gs.best_estimator_
            best.fit(X_train, y_train)

            y_train_pred = best.predict(X_train)
            y_test_pred = best.predict(X_test)

            # Explicit averaging to avoid warnings. Adjust if multi-class.
            train_f1 = f1_score(y_train, y_train_pred, average="binary")
            test_f1 = f1_score(y_test, y_test_pred, average="binary")

            report[model_name] = test_f1

        return report

    except Exception as e:
        raise CustomException(e, sys)
