# Standard libraries
import os  # for interacting with the operating system (paths, directories)
import sys  # for system-specific parameters and functions

# Custom exception and logging modules
from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Custom logger to track events

# Data handling
import pandas as pd  # For reading and manipulating data

# Machine Learning utilities
from sklearn.model_selection import train_test_split  # For splitting dataset into train and test

# To create configuration classes easily
from dataclasses import dataclass  

# Import data transformation modules
from src.components.data_transformation import DataTransformation, DataTransformationConfig

 # Import model training modules
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

# # ---------------------------------------------------------------
# Configuration for Data Ingestion: stores file paths
# ---------------------------------------------------------------
@dataclass
class DataIngestionConfig:
    # Path to save training dataset
    train_data_path: str = os.path.join('artifacts', "train.csv")
    # Path to save testing dataset
    test_data_path: str = os.path.join('artifacts', "test.csv")
    # Path to save raw/original dataset
    raw_data_path: str = os.path.join('artifacts', "data.csv")


# ---------------------------------------------------------------
# Class for handling data ingestion process
# ---------------------------------------------------------------
class DataIngestion:
    def __init__(self):
        # Initialize configuration for data ingestion
        self.ingestion_config = DataIngestionConfig()

    # Method to start data ingestion
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read dataset from CSV
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create folder if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split the dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training and testing sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return paths of train and test data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Raise a custom exception with system info
            raise CustomException(e, sys)


# ---------------------------------------------------------------
# Main execution: run the data ingestion, transformation, and model training
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Create DataIngestion object
    obj = DataIngestion()
    # Perform data ingestion
    train_data, test_data = obj.initiate_data_ingestion()

     # Perform data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Train the model using the transformed data
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
