import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            Random_param = {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'max_features': ['sqrt', 'log2', None],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
            Decision_param = {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'splitter': ['best', 'random'],
                'max_features': ['sqrt', 'log2'],
            }
            Gradient_param = {
                'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'criterion': ['squared_error', 'friedman_mse'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
            Linear_param = {}
            XGB_param = {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
            Cat_param = {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            }
            Ada_param = {
                'learning_rate': [0.1, 0.01, 0.5, 0.001],
                'loss': ['linear', 'square', 'exponential'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }

            grid_cv = [
                ("Random Forest", RandomForestRegressor(), Random_param),
                ("Decision Tree", DecisionTreeRegressor(), Decision_param),
                ("Gradient Boosting", GradientBoostingRegressor(), Gradient_param),
                ("Linear Regression", LinearRegression(), Linear_param),
                ("XGBRegressor", XGBRegressor(), XGB_param),
                ("CatBoosting Regressor", CatBoostRegressor(verbose=False), Cat_param),
                ("AdaBoost Regressor", AdaBoostRegressor(), Ada_param)
            ]

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=grid_cv
            )

            best_model_name = max(model_report, key=lambda k: model_report[k]["test_score"])
            best_model_score = model_report[best_model_name]["test_score"]
            best_model = model_report[best_model_name]["best_estimator"]

            if best_model_score < 0.6:
                raise CustomException(f"No best model found. Best score: {best_model_score}", sys)

            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)
            return score

        except Exception as e:
            raise CustomException(e, sys)
