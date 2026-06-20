import os
import sys
import pandas as pd

from dotenv import load_dotenv
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


load_dotenv()


os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")


import kaggle



@dataclass
class DataIngestionConfig:


    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join( "artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts","raw.csv")

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            os.makedirs("artifacts",exist_ok=True)
            logging.info( "Downloading dataset from Kaggle")
            kaggle.api.dataset_download_files(
                "blastchar/telco-customer-churn",
                path="artifacts", 
                unzip=True
            )

            logging.info( "Dataset downloaded successfully" )

            df = pd.read_csv("artifacts/WA_Fn-UseC_-Telco-Customer-Churn.csv")

            logging.info( "Dataset loaded successfully")

            # Taking sample data

            df = df.sample( n=1000,random_state=42)

            logging.info( "Selected 1000 records")

            # Saving raw data

            df.to_csv(

                self.ingestion_config.raw_data_path,
                index=False,
                header=True

            )

            logging.info("Raw data saved")

            # Train test split

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info( "Train test split completed" )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            logging.error("Error occurred in data ingestion")
            raise CustomException(e,sys)


