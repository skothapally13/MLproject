import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self,numerical_columns,categorical_columns):
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())                   
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ( "cat_pipeline",cat_pipeline,categorical_columns)                                                             
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path ) 
            logging.info("Train and test data loaded") 

            # ===============================
            # TotalCharges Cleaning
            # ===============================

            train_df["TotalCharges"] = pd.to_numeric(
                train_df["TotalCharges"],
                errors="coerce"
            )

            test_df["TotalCharges"] = pd.to_numeric(
                test_df["TotalCharges"],
                errors="coerce"
            )

            median_value = train_df["TotalCharges"].median()

            train_df["TotalCharges"].fillna( median_value,inplace=True )

            test_df["TotalCharges"].fillna(median_value,inplace=True )

            logging.info(  "Missing values handled")
              
            
            # ===============================
            # Log Transformation
            # ===============================

            train_df["TotalCharges"] = np.log1p(train_df["TotalCharges"])

            test_df["TotalCharges"] = np.log1p(test_df["TotalCharges"])

            logging.info( "Log transformation completed" )

            # ===============================
            # Split X and y
            # ===============================

            target_column = "Churn"

            X_train = train_df.drop( target_column,axis=1)

            y_train = train_df[target_column]

            X_test = test_df.drop(target_column,axis=1)

            y_test = test_df[target_column]

            # ===============================
            # Remove Customer ID
            # ===============================


            if "customerID" in X_train.columns:

                X_train.drop("customerID",axis=1,inplace=True)
                X_test.drop("customerID",axis=1,inplace=True)


            # ===============================
            # Dynamic column selection
            # ===============================


            numerical_columns = X_train.select_dtypes(
                include=["int64","float64"]).columns.tolist()

            categorical_columns = X_train.select_dtypes(include=["object"] ).columns.tolist()

            logging.info(f"Numerical columns: {numerical_columns}")
    
            logging.info(f"Categorical columns: {categorical_columns}")

            # ===============================
            # Encode Target
            # ===============================

            label_encoder = LabelEncoder()

            y_train = label_encoder.fit_transform(y_train )
            y_test = label_encoder.transform(y_test )

            logging.info("Target encoding completed")


            # ===============================
            # Preprocessing
            # ===============================


            preprocessing_obj = self.get_data_transformer_object(numerical_columns,categorical_columns )

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr,y_train]
            test_arr = np.c_[X_test_arr, y_test]

            logging.info("Preprocessing completed")
                
            # ===============================
            # Save Preprocessor
            # ===============================


            save_object(
                file_path=
                self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )



            logging.info( "Preprocessor saved")
               
           




            return (

                train_arr,

                test_arr,

                self.data_transformation_config.preprocessor_obj_file_path

            )


        except Exception as e:
           raise CustomException(e,sys)
        
