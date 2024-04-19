import os
import sys
from src.InsurencepPricePrediction.logger import logging
from src.InsurencepPricePrediction.exception import customexception
from src.InsurencepPricePrediction.components.data_ingestion import data_ingestion
from src.InsurencepPricePrediction.components.data_transformation import data_transformation
from src.InsurencepPricePrediction.components.model_trainer import ModelTrainer
from src.InsurencepPricePrediction.components.model_evaluation import modelevaluation




obj1=data_ingestion()
train_data,test_data=obj1.initiate_data_ingestion()

obj2=data_transformation()
train_arr,test_arr=obj2.data_transform_initiated(train_data,test_data)

obj3=ModelTrainer()
model=obj3.initate_model_training(train_arr,test_arr)

obj4=modelevaluation()
obj4.modelevaluationint(train_arr,test_arr)

