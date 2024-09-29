import numpy as np
import scipy as sp
import pandas as pd
import os
import time
import sys
from sklearn import ensemble
import pickle
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
# import tracemalloc

# sys.stdout = open('outrf.txt','wt')
# start_time = time.time()
# tracemalloc.start()
train_path = "D:\\ass3\COL774_drug_review\DrugsComTrain.csv"
test_path = "D:\\ass3\COL774_drug_review\DrugsComTest.csv"
val_path = "D:\\ass3\COL774_drug_review\DrugsComVal.csv"

print('Started')
dataframe = pd.read_csv(train_path)
# dataframe = dataframe.sample(frac=0.02)
condition = dataframe["condition"]

# print(dataframe['review'].head())
vocab = dict()
def count(str):
    words = set(str.split())
    words=words-set(stopwords.words('english'))
    for i in words:
        vocab.setdefault(i,0)
        vocab[i] += 1 
    return ' '.join(words)     
c=0
def clean_data(path,data=pd.DataFrame()):
    global c
    for i in range(len(data['review'])):
        str = ''.join(letter for letter in data['review'].iloc[i] if letter not in ['.', ',', ':', '@', '[', '!', ']', '=', '&', '(', ')', '?', '"', '#', '$','{', '}', '\\', '/', '%'])
        if 'train' in path:
            data['review'].iloc[i]=count(str)
        else:
            words = set(str.split())
            words=words-set(stopwords.words('english'))
            data['review'].iloc[i]=' '.join(words)
            

        print(c)
        c += 1
    return data

print('Cleaning Begins')
dataframe=clean_data(train_path,data=dataframe)

vocab = dict(sorted(vocab.items(), key= lambda item : item[1], reverse=True))
# for i in vocab.items():
    # print(i)

# REMOVING STOPWORDS

stop = stopwords.words('english')
stop.append("I")
stop.append("It")
keys = list(vocab.keys())
# stop+= keys[8000:]
stop+= keys[250:]

def datastopwordsremover(dataframe):
    dataframe['review'] = dataframe['review'].apply(lambda x : ' '.join([word for word in x.split() if word not in stop]))
    return dataframe


# COUNTVECTORIZER


def vectorizer(dataframe,vocab=None):
    cv = CountVectorizer(vocabulary=vocab)
    array=cv.fit_transform(dataframe['review'])
    vocab=cv.vocabulary_
    # a = condition.unique()
    # index={a[i]:i for i in range(len(a))}
    # dataframe['condition'].replace(index.keys(),index.values(),inplace=True)
    dataframe.dropna()
    # print(dataframe['condition'])
    for i in range(array.shape[0]):
        array[(i,12)]=dataframe['condition'].iloc[i]
    # dataframe['review'][1234] = condition
    rating = dataframe['rating']
    return (array,rating,vocab)

# DECISION TREE IMPLEMENTATION

model = ensemble.RandomForestClassifier()
dataframe=datastopwordsremover(dataframe=dataframe)
a = condition.unique()
index={a[i]:i for i in range(len(a))}
dataframe['condition'].replace(index.keys(),index.values(),inplace=True)
array,rating,vocab=vectorizer(dataframe=dataframe)
model.fit(array,rating)

# randomforest = GridSearchCV(model,{'n_estimators' : [500,1000,1500,2000],'min_samples_split' : [2,4,6,8,10],'max_features' : ['sqrt',2,3,4]},n_jobs=True,return_train_score=False,verbose=True)
randomforest = GridSearchCV(model,{'n_estimators' : [50,100,150,200,250,300,350,400,450],'min_samples_split' : [2,4,6,8,10],'max_features' : [0.4,0.5,0.6,0.7,0.8]},cv=2,n_jobs=-1,return_train_score=False,verbose=True)
op = randomforest.fit(array,rating)
with open('model-randomforest','wb') as file:
    pickle.dump(op,file)
df = pd.DataFrame(randomforest.cv_results_)
# print(df[['param_n_estimators','param_min_samples_split','param_max_features','mean_test_score']])
# print(randomforest.best_estimator_,randomforest.best_score_)
bestmodel = randomforest.best_estimator_


# print(f'Model Training Time: {time.time() - start_time}')
# ACCURACY OF TRAINING SET

# bestmodel = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, min_samples_split=10)
# bestmodel.fit(train_inputs,train_target)

BestAcc = 100*(bestmodel.score(array,rating))
result1 = BestAcc

  
# ACCURACY OF TESTING SET

dataframe = pd.read_csv(test_path)
# dataframe = dataframe.sample(frac=0.02)
condition = dataframe['condition']
a=set(condition.unique())
b=set(index.keys())
rem=a-b
for i in rem:
    index[i]=len(index)+1
dataframe['condition'].replace(index.keys(),index.values(),inplace=True)
dataframe.dropna()
clean_data(test_path,dataframe)
dataframe=datastopwordsremover(dataframe=dataframe)
array,rating,vocab=vectorizer(dataframe=dataframe,vocab=vocab)
BestAcc = 100*(bestmodel.score(array,rating))
result2 = BestAcc

# ACCURACY OF VALIDATION SET

dataframe = pd.read_csv(val_path)
# dataframe = dataframe.sample(frac=0.02)
condition = dataframe['condition']
a=set(condition.unique())
b=set(index.keys())
rem=a-b
for i in rem:
    index[i]=len(index)+1
dataframe['condition'].replace(index.keys(),index.values(),inplace=True)
dataframe.dropna()
clean_data(val_path,dataframe)
dataframe=datastopwordsremover(dataframe=dataframe)
array,rating,vocab=vectorizer(dataframe=dataframe,vocab=vocab)
BestAcc = 100*(bestmodel.score(array,rating))
result3 = BestAcc

print(f'Accuracy on Training dataset: {result1} %')
print(f'Accuracy on Testing dataset: {result2} %')
print(f'Accuracy on Validation dataset: {result3} %')