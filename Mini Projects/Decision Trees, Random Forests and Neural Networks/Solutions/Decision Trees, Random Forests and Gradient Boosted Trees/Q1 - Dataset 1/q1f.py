from operator import contains
import numpy as np
import pandas as pd
import os
import sys
from sklearn import tree
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

train_path = "D:\\ass3\data\\train.csv"
test_path = "D:\\ass3\data\\test.csv"
val_path = "D:\\ass3\data\\val.csv"

#Cleansing the data, removing samples that are missing values

def Cleansing(path):
    df = pd.read_csv(path)
    # i = dataframe[((dataframe.Age == '?') | (dataframe.Shape == '?') | (dataframe.Margin == '?') | (dataframe.Density == '?'))].index
    # df = dataframe.drop(i)
    df = df.replace('?',np.nan)
    inputs = df.drop(['Severity','BI-RADS assessment'], axis='columns')
    target = df['Severity']
    inputs['Age'] = inputs['Age'].astype(float)
    inputs['Shape'] = inputs['Shape'].astype(float)
    inputs['Margin'] = inputs['Margin'].astype(float)
    inputs['Density'] = inputs['Density'].astype(float)
    # target['Severity'] = target['Severity'].astype(float)
    # print(inputs.dtypes)
    return(inputs,target)

train_inputs = Cleansing(train_path)[0]
train_target = Cleansing(train_path)[1]
test_inputs = Cleansing(test_path)[0]
test_target = Cleansing(test_path)[1]
val_inputs = Cleansing(val_path)[0]
val_target = Cleansing(val_path)[1]

model = xgboost.XGBClassifier()
model.fit(train_inputs,train_target)

xgboosted_dtree = GridSearchCV(model,{'n_estimators' : [10,20,30,40,50],'subsample' : [0.1,0.2,0.3,0.4,0.5,0.6],'max_depth' : [4,5,6,7,8,9,10]},cv=5,return_train_score=False,refit=True)
xgboosted_dtree.fit(train_inputs,train_target)
df = pd.DataFrame(xgboosted_dtree.cv_results_)
df.to_csv('CV_Resullts_XGBoost.csv')
print(xgboosted_dtree.best_estimator_,xgboosted_dtree.best_score_)
# bestmodel = xgboost.XGBClassifier(n_estimators=10,max_depth=5,subsample=0.4)
# bestmodel.fit(train_inputs,train_target)
bestmodel=xgboosted_dtree.best_estimator_

#Finding Accuracy

def Accuracy(path):
    if 'train' in path:    
        BestAcc = 100*(bestmodel.score(train_inputs,train_target))
        print(f'Accuracy on Training dataset: {BestAcc} %')
    elif 'test' in path:
        BestAcc = 100*(bestmodel.score(test_inputs,test_target))
        print(f'Accuracy on Testing dataset: {BestAcc} %')
    elif 'val' in path:
        BestAcc = 100*(bestmodel.score(val_inputs,val_target))
        print(f'Accuracy on Validation dataset: {BestAcc} %')

Accuracy(train_path)
Accuracy(test_path)
Accuracy(val_path)