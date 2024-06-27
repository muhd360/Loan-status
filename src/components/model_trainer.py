import os
import sys
import numpy as np
import xgboost as xgb
src_path = os.path.abspath(os.path.join("/home/muhd/Desktop/python-proj/mlproject"))
sys.path.append(src_path)
from dataclasses import dataclass
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    # def objective(self,train_array,test_array):
    #     # Split the train and test arrays into features and labels
    #     X_train, y_train = (train_array[:, :-1], train_array[:, -1])
    #     X_test, y_test = (test_array[:, :-1], test_array[:, -1])

    #     # Define hyperparameters to tune88
    #     params = {
    #         'iterations': trial.suggest_int('iterations', 100, 1000),
    #         'depth': trial.suggest_int('depth', 1, 10),
    #         'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
    #         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 1e1, log=True)
    #     }

    #     # Create a new CatBoost model for each trial
    #     model = CatBoostRegressor(**params, verbose=0)

    #     # Evaluate the model using a custom evaluation function
    #     model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=model, param=params)
        
    #     # Return the evaluation metric
    #     return model_report

        

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train=(
                train_array[:,:-1],
                train_array[:,-1],

            )

            X_test,y_test=(
                test_array[:,:-1],
                test_array[:,-1]

            )
            # X_test=np.array(X_test)
            # X_test=np.squeeze(X_test)
          
            #X_test=X_test.reshape(X_test.shape[0], -1)
            print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


            models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(),
            #"CatBoosting Regressor": CatBoostRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            logging.info(f"code worked till here")
            empty_df=pd.DataFrame()
            empty_soln=pd.DataFrame()
            #print(y_train)
            for model_name, model in models.items():
                # Train each model
                model.fit(X_train, y_train)
                
                # Make predictions
                predictions = model.predict(X_test)
                print(predictions)
                
                # Store predictions in dataframe
                empty_df[model_name] = model.predict(X_train)
                empty_soln[model_name] = predictions

            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.2:

                raise CustomException("No best model found")
            
            logging.info(f"working ok")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=LinearRegression()
            )
            logging.info("model is saved")

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)




            
        except Exception as e:
            raise CustomException(e,sys)