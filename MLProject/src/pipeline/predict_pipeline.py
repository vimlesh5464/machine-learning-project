# Standard library imports
import sys  # for system-specific parameters and exception handling
import os   # for file and directory operations
import pandas as pd  # for handling data in DataFrame format

# Custom modules from your project
from src.exception import CustomException  # custom exception handling class
from src.utils import load_object  # utility function to load saved Python objects


# ===============================
# Pipeline class for making predictions
# ===============================
class PredictPipeline:
    """
    This class loads the trained model and preprocessor,
    transforms input features, and predicts target values.
    """
    def __init__(self):
        pass  # no initialization needed for now

    def predict(self, features):
        """
        Predict target values using trained model and preprocessor.

        Args:
            features (DataFrame or array-like): Input features to predict

        Returns:
            preds (array): Predicted values

        Raises:
            CustomException: If any error occurs during prediction
        """
        try:
            # Paths to saved model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'proprocessor.pkl')

            print("Before Loading")

            # Load trained model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")

            # Transform input features using preprocessor
            data_scaled = preprocessor.transform(features)

            # Make predictions using the trained model
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            # Wrap any exception in CustomException for better logging
            raise CustomException(e, sys)


# ===============================
# Class to handle custom input data
# ===============================
class CustomData:
    """
    Class to structure user input data for prediction.
    Converts input into a pandas DataFrame that can be processed by the pipeline.
    """
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        # Store input values as object attributes
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Converts the input attributes into a pandas DataFrame
        suitable for preprocessing and prediction.

        Returns:
            pandas.DataFrame: Single-row DataFrame with input data

        Raises:
            CustomException: If conversion fails
        """
        try:
            # Create a dictionary from input attributes
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert dictionary to DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Wrap exception for better logging and debugging
            raise CustomException(e, sys)
