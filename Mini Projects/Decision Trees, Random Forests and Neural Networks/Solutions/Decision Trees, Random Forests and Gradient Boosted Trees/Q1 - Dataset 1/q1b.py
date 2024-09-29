import numpy as np
import pandas as pd
import os
import sys
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

train_path = "D:\\ass3\data\\train.csv"
test_path = "D:\\ass3\data\\test.csv"
val_path = "D:\\ass3\data\\val.csv"

#Cleansing the data, removing samples that are missing values

def Cleansing(path):
    dataframe = pd.read_csv(path)
    i = dataframe[((dataframe.Age == '?') | (dataframe.Shape == '?') | (dataframe.Margin == '?') | (dataframe.Density == '?'))].index
    df = dataframe.drop(i)
    inputs = df.drop(['Severity','BI-RADS assessment'], axis='columns')
    target = df['Severity']
    return(inputs,target)

#Finding Accuracy

model = tree.DecisionTreeClassifier()
train_inputs = Cleansing(train_path)[0]
train_target = Cleansing(train_path)[1]
test_inputs = Cleansing(test_path)[0]
test_target = Cleansing(test_path)[1]
val_inputs = Cleansing(val_path)[0]
val_target = Cleansing(val_path)[1]
model.fit(train_inputs,train_target)

dtree = GridSearchCV(model,{'max_depth' : [5,8,10,12,15,18,20],'min_samples_split' : [2,4,6,8,10],'min_samples_leaf' : [1,2,3]},cv=5,return_train_score=False)
dtree.fit(train_inputs,train_target)
df = pd.DataFrame(dtree.cv_results_)
print(dtree.best_estimator_,dtree.best_score_)
bestmodel = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, min_samples_split=10)
bestmodel.fit(train_inputs,train_target)

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

#Visualising the Decision Tree

plt.figure(figsize = (15,10))
dec_tree = tree.plot_tree(bestmodel, filled =True, rounded= True)
plt.show()