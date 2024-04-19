import os
import sys
import pandas as pd
import numpy as np
from src.InsurencepPricePrediction.logger import logging
from src.InsurencepPricePrediction.exception import customexception
from src.InsurencepPricePrediction.utlis.utils import load_object

class model_pred_config:
    preprcessor_path=os.path.join('artifacts',"preprocessor.pkl")
    model_path=os.path.join('artifacts',"model.pkl")
class model_prediction:
    def __init__(self):
        self.model_pred=model_pred_config()
    def model_pred_initiate(self,features):
        try:
            preprocessor=load_object(self.model_pred.preprcessor_path)
            model=load_object(self.model_pred.model_path)
            transform_data=preprocessor.transform(features)
            pred=model.predict(transform_data)
            return pred
        except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)

class custom_data:
    def __init__(self,age,sex,bmi,children,smoker,region):
        self.age=age
        self.sex=sex
        self.bmi=bmi
        self.children=children
        self.smoker=smoker
        self.region=region

    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'age':[self.age],
                    'sex':[self.sex],
                    'bmi':[self.bmi],
                    'children':[self.children],
                    'smoker':[self.smoker],
                    'region':[self.region]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)
    