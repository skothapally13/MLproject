import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import (
    save_object,
    evaluate_models
)





@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):

        try:

            X_train = train_array[:,:-1]
            y_train = train_array[:,-1]
            X_test = test_array[:,:-1]
            y_test = test_array[:,-1]

            models = {
                "Logistic Regression":LogisticRegression(),
                "Decision Tree":DecisionTreeClassifier(),
                "Random Forest":RandomForestClassifier(),
                "Gradient Boosting":GradientBoostingClassifier(),
                "AdaBoost":AdaBoostClassifier(),
                "KNN":KNeighborsClassifier(),
                "XGBoost": XGBClassifier(),
                "CatBoost":CatBoostClassifier(verbose=False )
             }                 
               
            params = {
            "Logistic Regression":{
                "C":[0.1,1,10], 
                "max_iter":[100,200]
                },
            "Decision Tree":{
                "criterion":["gini","entropy"],
                "max_depth":[3,5,10] 
                },
            "Random Forest":{
                "n_estimators":[100,200],
                "max_depth":[5,10,None]
                },
            "Gradient Boosting":{
                "learning_rate":[0.01,0.1],
                "n_estimators":[100,200]
                },
            "AdaBoost":{
                "learning_rate":[0.01,0.1],
                "n_estimators":[100,200]
                },
            "KNN":{
                "n_neighbors":[3,5,7]
                },
            "XGBoost":{
                "learning_rate":[0.01,0.1],
                "n_estimators":[100,200]
                },
            "CatBoost":{
                "depth":[5,7],
                "learning_rate":[0.01,0.1]}
            }

            model_report = evaluate_models(X_train,y_train, X_test, y_test,models, params)
                
            
            best_score = max( model_report.values() )

            best_model_name = list( model_report.keys())[ list(model_report.values()).index(best_score)]
                
            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name}")

            save_object(
                self.model_trainer_config.trained_model_file_path,
                best_model
            )

            return best_score

        except Exception as e:
            raise CustomException(e,sys)
        
