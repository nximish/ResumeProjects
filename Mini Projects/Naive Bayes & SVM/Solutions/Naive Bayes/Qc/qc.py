import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import re
from wordcloud import WordCloud, STOPWORDS
import random
trainDir=sys.argv[1]
testDir=sys.argv[2]

neg_path = os.path.join(trainDir,'neg')
neg = os.listdir(neg_path)
pos_path = os.path.join(trainDir,'pos')
pos = os.listdir(pos_path)

# print(neg)
# print(pos)
vocab = dict()
def count(str):
    counts = dict()
    words = str.split()
    for i in words:
        if i not in counts:
            counts[i] = 1
    for j in counts:
        if j in counts:
            vocab.setdefault(j,0)
            vocab[j] += 1
c=0
def read_data(path):
    global c
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readlines()
        for i in line:
            str = ''.join(letter for letter in i if letter not in ['.', ',', ':', '@', '[', '!', ']', '=', '&', '(', ')', '?', '"', '#', '$','{', '}', '\\', '/', '%'])
            count(str)
    c += 1


for file_1 in neg:
    if file_1.endswith(".txt"):
        path = f"{neg_path}\{file_1}"
        read_data(path)
vocab_neg = vocab.copy()

vocab=dict()
c=0
for file_2 in pos:
    if file_2.endswith(".txt"):
        path = f"{pos_path}\{file_2}"
        read_data(path)
vocab_pos = vocab.copy()


prob_neg = dict()
for j in vocab_neg:
    prob_j = (1+vocab_neg[j])/ (len(neg)+2)
    prob_neg.update({j : prob_j})

prob_pos = dict()
for j in vocab_pos:
    prob_j = (1+vocab_pos[j])/ (len(pos)+2)
    prob_pos.update({j : prob_j})


p_neg = len(neg)/25000
p_pos = 1 - p_neg


neg_test_path = os.path.join(testDir,'neg')
neg_test = os.listdir(neg_test_path)
pos_test_path = os.path.join(testDir,'pos')
pos_test = os.listdir(pos_test_path)


def prediction(vocab, model = []):
    likelihood = 0.0
    a = 0
    list = [0.0,0.0]
    for i in model:
        for j in vocab:
            likelihood += math.log(i.setdefault(j,(1.0/12502)))
        list[a] = likelihood
        likelihood = 0.0
        a += 1
    return list

def addToCM(p,cl):
    # 0 for positive class
    if (cl==0):
        if p[0] < p[1]:
            CM[0][0] += 1
        else:
            CM[0][1] += 1
    else:
        if p[0] > p[1]:
            CM[1][1] += 1
        else:
            CM[1][0] += 1




CM = [[0,0],[0,0]]


for test_file_1 in neg_test:
    vocab= dict()
    if test_file_1.endswith(".txt"):
        path = f"{neg_test_path}\{test_file_1}"
        read_data(path)
        p = prediction(vocab,[prob_neg,prob_pos])
        addToCM(p,1)

for test_file_2 in pos_test:
    vocab = dict()
    if test_file_2.endswith(".txt"):
        path = f"{pos_test_path}\{test_file_2}"
        read_data(path)
        p = prediction(vocab,[prob_neg,prob_pos])
        addToCM(p,0)


print(f'Confusion Matrix for Test Data for My Model : {CM}')




CM = [[0,0],[0,0]]


for test_file_1 in neg_test:
    vocab= dict()
    if test_file_1.endswith(".txt"):
        path = f"{neg_test_path}\{test_file_1}"
        read_data(path)
        p = prediction(vocab,[prob_neg,prob_pos])
        addToCM(p,random.randint(0,1))

for test_file_2 in pos_test:
    vocab = dict()
    if test_file_2.endswith(".txt"):
        path = f"{pos_test_path}\{test_file_2}"
        read_data(path)
        p = prediction(vocab,[prob_neg,prob_pos])
        addToCM(p, random.randint(0, 1))


print(f'Confusion Matrix for Test Data for Random Model : {CM}')


CM = [[0,0],[0,0]]

for test_file_1 in neg_test:
    vocab= dict()
    if test_file_1.endswith(".txt"):
        path = f"{neg_test_path}\{test_file_1}"
        read_data(path)
        p = prediction(vocab,[prob_neg,prob_pos])
        addToCM(p,0)

for test_file_2 in pos_test:
    vocab = dict()
    if test_file_2.endswith(".txt"):
        path = f"{pos_test_path}\{test_file_2}"
        read_data(path)
        p = prediction(vocab,[prob_neg,prob_pos])
        addToCM(p,0)

print(f'Confusion Matrix for Test Data if all classes are predicted +ve : {CM}')
