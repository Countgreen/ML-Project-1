import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        results={}
        for name,model,param in models:
            gs=GridSearchCV(estimator=model,param_grid=param,cv=3)
            gs.fit(X_train,y_train)
            y_train_pred=gs.predict(X_train)
            y_test_pred=gs.predict(X_test)
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            results[name] = {
                "best_params": gs.best_params_,
                "train_score": train_model_score,
                "test_score": test_model_score,
                "best_estimator": gs.best_estimator_
            }
        return results
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
