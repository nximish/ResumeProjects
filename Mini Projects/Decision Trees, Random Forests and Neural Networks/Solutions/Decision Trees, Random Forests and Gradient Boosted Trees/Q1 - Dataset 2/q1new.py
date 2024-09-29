import numpy as np
import scipy as sp
import pandas as pd
import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
# import tracemalloc

# tracemalloc.start()
train_path = "D:\\ass3\COL774_drug_review\DrugsComTrain.csv"
test_path = "D:\\ass3\COL774_drug_review\DrugsComTest.csv"
val_path = "D:\\ass3\COL774_drug_review\DrugsComVal.csv"

dataframe = pd.read_csv(train_path)
# dataframe=dataframe.sample(frac=0.05)
# print(tracemalloc.get_traced_memory())
# tracemalloc.stop()
# print(dataframe.head())
condition = dataframe["condition"]
# review = dataframe["review"]
# rating = dataframe["rating"]
print(dataframe['review'].head())
vocab = dict()
def count(str):
    counts = dict()
    words = set(str.split())
    words=words-set(stopwords.words('english'))
    for i in words:
        # if i not in vocab:
        vocab.setdefault(i,1)
        # vocab[j] += 1
    # for j in counts:
    #     if j in counts:
    #         
c=0
def clean_data(data=pd.DataFrame()):
    global c
    for j in range(0,len(data['review']),10000):
        reviewList=[]
        for i in range(j,j+10000):
            str=re.sub('[.,:@[!]=&()?"#$/{/}\\/%]','',data['review'].iloc[i])
            # str = ''.join(letter for letter in data['review'].iloc[i] if letter not in ['.', ',', ':', '@', '[', '!', ']', '=', '&', '(', ')', '?', '"', '#', '$','{', '}', '\\', '/', '%'])
            reviewList.append(str)
            count(str)
            print(c)
            c += 1
        data['review'].iloc[j:j+10000]=reviewList
        reviewList=[]
    return data

dataframe=clean_data(data=dataframe)
# print(dataframe['review'].head())
vocab = dict(sorted(vocab.items(), key= lambda item : item[1], reverse=True))
# print(vocab)
for i in vocab.items():
    print(i)

# REMOVING STOPWORDS
stop = stopwords.words('english')
stop.append("I")
stop.append("It")
keys = list(vocab.keys())
stop+= keys[8000:]
# dataframe=clean_data(data=dataframe)
dataframe['review'] = dataframe['review'].apply(lambda x : ' '.join([word for word in x.split() if word not in stop]))
print(dataframe['review'].head())
# for i,j in vocab.items():
#     print((i,j))
# COUNTVECTORIZER
# cv = TfidfVectorizer(min_df=1,stop_words='english')
# cv_fit = cv.fit_transform(dataframe['review'])
# a = cv_fit.toarray()
# print(a)
a = condition.unique()
print(a.shape)