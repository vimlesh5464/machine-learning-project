# Standard library imports
import os  # for file and directory operations
import sys  # for system-specific parameters and functions
from dataclasses import dataclass  # to define simple classes for storing configuration

# Machine learning libraries
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score  # to evaluate regression models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom modules
from src.exception import CustomException  # custom exception handling
from src.logger import logging  # custom logging utility
from src.utils import save_object, evaluate_models  # utility functions

# ===============================
# Configuration for model training
# ===============================
@dataclass
class ModelTrainerConfig:
    """
    Stores configuration related to model training,
    such as the file path to save the trained model.
    """
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  # path to save model

# ===============================
# Model Trainer Class
# ===============================
class ModelTrainer:
    def __init__(self):
        """
        Initialize ModelTrainer with its configuration
        """
        self.model_trainer_config = ModelTrainerConfig()

    # ===============================
    # Main function to train, evaluate, and save the best model
    # ===============================
    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple regression models, evaluates them, selects the best model,
        saves it to disk, and returns its R2 score on test data.

        Args:
            train_array (numpy array): Training data including features and target
            test_array (numpy array): Testing data including features and target

        Returns:
            r2_square (float): R2 score of the best model on test data
        """
        try:
            logging.info("Split training and test input data")
            
            # Split features and target for training and testing
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # all columns except last = features
                train_array[:, -1],   # last column = target
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Dictionary of models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameter grid for each model for GridSearch
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate all models and get their test scores
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Get the best model score
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name using its score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Select the best model instance
            best_model = models[best_model_name]

            # If best model is too weak, raise exception
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            # Save the best model to disk
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict on test data
            predicted = best_model.predict(X_test)

            # Calculate R2 score
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Wrap exceptions in CustomException for logging and debugging
            raise CustomException(e, sys)
