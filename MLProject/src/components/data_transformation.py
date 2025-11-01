# System and utility libraries
import sys  # for exception handling context
from dataclasses import dataclass  # for creating configuration classes

# Data manipulation libraries
import numpy as np  # numerical operations
import pandas as pd  # dataframes for reading and manipulating data

# Sklearn modules for preprocessing and pipelines
from sklearn.compose import ColumnTransformer  # apply transformations to specific columns
from sklearn.impute import SimpleImputer  # handle missing values
from sklearn.pipeline import Pipeline  # create sequential preprocessing pipelines
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # encoding and scaling features

# Custom modules for logging, exceptions, and utilities
from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Logging to track progress
import os  # For file path handling
from src.utils import save_object  # Save objects like preprocessing pipelines

# ---------------------------------------------------------------
# Configuration for Data Transformation
# ---------------------------------------------------------------
@dataclass
class DataTransformationConfig:
    # Path to save the preprocessor object (like a pipeline)
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")


# ---------------------------------------------------------------
# Data Transformation class: handles preprocessing of the dataset
# ---------------------------------------------------------------
class DataTransformation:
    def __init__(self):
        # Initialize the configuration
        self.data_transformation_config = DataTransformationConfig()

    # Method to create preprocessing pipelines
    def get_data_transformer_object(self):
        '''
        This function is responsible for creating preprocessing pipelines
        for numerical and categorical features.
        '''
        try:
            # Define columns
            numerical_columns = ["writing_score", "reading_score"]  # numeric features
            categorical_columns = [  # categorical features
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical columns: impute missing values & scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # fill missing values with median
                    ("scaler", StandardScaler())  # scale features to have mean=0, std=1
                ]
            )

            # Pipeline for categorical columns: impute, encode, scale
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # fill missing values with mode
                    ("one_hot_encoder", OneHotEncoder()),  # convert categorical to numerical
                    ("scaler", StandardScaler(with_mean=False))  # scale features (without centering)
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            # Raise a custom exception if anything fails
            raise CustomException(e, sys)

    # Method to perform preprocessing on train and test data
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()  # get pipeline

            # Define target column
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target feature
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframe.")

            # Fit transformer on train and transform both train and test
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target into one array for model training
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object for later use (like inference)
            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,  # processed train data
                test_arr,   # processed test data
                self.data_transformation_config.preprocessor_obj_file_path  # path to saved preprocessor
            )

        except Exception as e:
            # Handle exceptions gracefully
            raise CustomException(e, sys)
