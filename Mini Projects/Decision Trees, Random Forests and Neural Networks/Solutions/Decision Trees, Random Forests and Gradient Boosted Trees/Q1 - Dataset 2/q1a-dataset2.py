import numpy as np
import scipy as sp
import pandas as pd
import os
import sys
import pickle
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
# import tracemalloc

# tracemalloc.start()
train_path = "D:\\ass3\COL774_drug_review\DrugsComTrain.csv"
test_path = "D:\\ass3\COL774_drug_review\DrugsComTest.csv"
val_path = "D:\\ass3\COL774_drug_review\DrugsComVal.csv"

dataframe = pd.read_csv(train_path)
# dataframe = dataframe.sample(frac=0.05)
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
            

        # print(c)
        c += 1
    return data

dataframe=clean_data(train_path,data=dataframe)
vocab = dict(sorted(vocab.items(), key= lambda item : item[1], reverse=True))
# for i in vocab.items():
    # print(i)

# REMOVING STOPWORDS

stop = stopwords.words('english')
stop.append("I")
stop.append("It")
keys = list(vocab.keys())
stop+= keys[8000:]

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

model = tree.DecisionTreeClassifier()
dataframe=datastopwordsremover(dataframe=dataframe)
a = condition.unique()
index={a[i]:i for i in range(len(a))}
dataframe['condition'].replace(index.keys(),index.values(),inplace=True)
array,rating,vocab=vectorizer(dataframe=dataframe)
model.fit(array,rating)

with open('Modal','wb') as file:
    pickle.dump('Modal',file)

# ACCURACY OF TRAINING SET

BestAcc = 100*(model.score(array,rating))
print(f'Accuracy on Training dataset: {BestAcc} %')
     
# ACCURACY OF TESTING SET

dataframe = pd.read_csv(test_path)
# dataframe = dataframe.sample(frac=0.05)
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
BestAcc = 100*(model.score(array,rating))
print(f'Accuracy on Testing dataset: {BestAcc} %')

# ACCURACY OF VALIDATION SET

dataframe = pd.read_csv(val_path)
# dataframe = dataframe.sample(frac=0.05)
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
BestAcc = 100*(model.score(array,rating))
print(f'Accuracy on Validation dataset: {BestAcc} %')

# dataframe = pd.read_csv(val_path)
# dataframe = dataframe.sample(frac=0.05)
# clean_data(val_path,dataframe)
# dataframe=datastopwordsremover(dataframe=dataframe)
# array,rating,vocab=vectorizer(dataframe=dataframe,vocab=vocab)
# BestAcc = 100*(model.score(array,rating))
# print(f'Accuracy on Validation dataset: {BestAcc} %')



# def Accuracy(path):
#     if 'train' in path:    
#         BestAcc = 100*(model.score(array,rating))
#         print(f'Accuracy on Training dataset: {BestAcc} %')
#     elif 'test' in path:
#         BestAcc = 100*(model.score(dataframe ,test_target))
#         print(f'Accuracy on Testing dataset: {BestAcc} %')
#     elif 'val' in path:
#         BestAcc = 100*(model.score(val_inputs,val_target))
#         print(f'Accuracy on Validation dataset: {BestAcc} %')
