# Standard library imports
import os   # for file and directory operations
import sys  # for system-specific parameters and functions

# Third-party library imports
import numpy as np  # numerical computations
import pandas as pd  # data manipulation
import dill  # (alternative to pickle for more complex objects, though not used here)
import pickle  # to serialize (save) and deserialize (load) Python objects
from sklearn.metrics import r2_score  # to evaluate regression model performance
from sklearn.model_selection import GridSearchCV  # for hyperparameter tuning

# Custom exception class from your project
from src.exception import CustomException

# ===============================
# Function to save a Python object to a file
# ===============================
def save_object(file_path, obj):
    """
    Saves a Python object to the given file path using pickle.

    Args:
        file_path (str): Path where the object will be saved
        obj (any Python object): Object to be saved

    Raises:
        CustomException: if saving fails
    """
    try:
        # Get directory path from file path
        dir_path = os.path.dirname(file_path)

        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open file in write-binary mode and dump object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raise custom exception with system info
        raise CustomException(e, sys)
    

# ===============================
# Function to evaluate multiple models with hyperparameter tuning
# ===============================
def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    """
    Trains and evaluates multiple models using GridSearchCV for hyperparameter tuning.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        models (dict): Dictionary of models {model_name: model_instance}
        param (dict): Dictionary of hyperparameters {model_name: param_grid}

    Returns:
        report (dict): Dictionary with test R2 scores {model_name: score}

    Raises:
        CustomException: if model evaluation fails
    """
    try:
        report = {}  # to store test scores of each model

        # Loop through each model
        for i in range(len(list(models))):
            model = list(models.values())[i]  # get model instance
            para = param[list(models.keys())[i]]  # get corresponding hyperparameters

            #Perform Grid Search with 3-fold cross-validation
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            #Set the model's best parameters and retrain on full training data
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on training and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 score for train and test
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score in the report
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        # Raise custom exception if something goes wrong
        raise CustomException(e, sys)
    

# ===============================
# Function to load a Python object from a file
# ===============================
def load_object(file_path):
    """
    Loads a Python object from the given file path using pickle.

    Args:
        file_path (str): Path of the saved object

    Returns:
        Loaded Python object

    Raises:
        CustomException: if loading fails
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
