import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from src.exception import CustomException




def save_object(file_path,obj):
    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)





def evaluate_models(X_train,y_train,X_test,y_test,models,param):

    try:
        report = {}
        for model_name, model in models.items():
            params = param[model_name]
            gs = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=5,
                scoring="roc_auc"
            )
            gs.fit(X_train,y_train)

            model.set_params( **gs.best_params_)
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]
            roc_auc = roc_auc_score(y_test,y_prob)
            report[model_name] = roc_auc

        return report

    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)