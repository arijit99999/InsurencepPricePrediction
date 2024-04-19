import os
import sys
import pandas as pd
from src.InsurencepPricePrediction.logger import logging
from src.InsurencepPricePrediction.exception import customexception
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,MinMaxScaler
from src.InsurencepPricePrediction.utlis.utils import save_object

class datatransformationconfig:
    preporcessor_path=os.path.join("artifacts","preprocessor.pkl")
class data_transformation:
    def __init__(self):
        self.prepocessor_path=datatransformationconfig()
    def data_transformation_preparation(self):
        try:
            cat=['sex', 'smoker', 'region']
            num=['age', 'bmi', 'children']
            sex=['female', 'male']
            smoker=['no','yes']
            region=['southwest', 'southeast', 'northwest', 'northeast']
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scale',MinMaxScaler())])
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[sex,smoker,region])),
                ('scale',MinMaxScaler())])
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,num),
            ('cat_pipeline',cat_pipeline,cat)])
            return preprocessor
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise customexception(e,sys)
        
    def data_transform_initiated(self,train_data,test_data):
       try:
          logging.info('trasnformation initiated')
          train_data=pd.read_csv(train_data)
          test_data=pd.read_csv(test_data)
          logging.info("read test and train data")
          target_column_name = 'expenses'
          input_feature_train_df = train_data.drop('expenses',axis=1)
          target_feature_train_df=train_data.iloc[:,-1:]
          input_feature_test_df=test_data.drop('expenses',axis=1)
          target_feature_test_df=test_data.iloc[:,-1:]
          preprocessor_obj=self.data_transformation_preparation()
          input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
          logging.info('transformation has been complited')
          input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
          train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
          test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
          save_object(obj=preprocessor_obj,file_path=self.prepocessor_path.preporcessor_path)
          logging.info('save preprocessor object')
          logging.info("data tranformation has been complited")
          return (train_arr,test_arr)
          
       except Exception as e:
          logging.info("exception during occured at data tarnsformation initiation stage")
          raise customexception(e,sys)