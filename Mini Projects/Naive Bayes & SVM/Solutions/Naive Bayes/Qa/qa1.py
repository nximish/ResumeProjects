import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import re
from wordcloud import WordCloud, STOPWORDS
trainDir=sys.argv[1]
testDir=sys.argv[2]

neg_path = os.path.join(trainDir,'neg')
neg = os.listdir(neg_path)
pos_path = os.path.join(trainDir,'pos')
pos = os.listdir(pos_path)

# print(neg)
# print(pos)
vocab = dict()
# def count(str,iterations):
#     counts = dict()
#     words = str.split()
#     for i in words:
#         if i not in counts:
#             counts[i] = 1
#     for j in counts:
#         if j in counts:
#             vocab.setdefault(j,0)
#             vocab[j] += 1

strpos = ''
strneg = ''
def count(str,klass,iterations):
    global strpos
    global strneg
    counts = dict()
    words = str.split()
    for i in words:
        # i = porter.stem(i)
        if int(iterations) <=1000:
            if klass == 0:
                strneg += ' '+i
            else:
                strpos += ' '+i
        if i not in counts:
            counts[i] = 1
    for j in counts:
        if j in counts:
            vocab.setdefault(j,0)
            vocab[j] += 1

c=0
def read_data(path,iteration):
    global c
    global strneg
    global strpos
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readlines()
        for i in line:
            str = ''.join(letter for letter in i if letter not in ['.',',',':','@','[','!',']','=','&','(',')','?','"','#','$','{','}','\\','/','%'])
            j=0
            if re.search("pos*",path):
                j=1
            count(str,j,iteration)
    c+=1            
# c=0
# strpos = ''
# strneg = ''
# def read_data(path):
#     global c
#     global strneg
#     global strpos
#     with open(path, 'r', encoding='utf-8') as f:
#         line = f.readlines()
#         for i in line:
#             str = ''.join(letter for letter in i if letter not in ['.',',',':','@','[','!',']','=','&','(',')','?','"','#','$','{','}','\\','/','%'])
            
#             if re.search("pos*",path):
#                 strpos+=""+str
#             else:
#                 strneg+=""+str
#             count(str)
#     c+=1

c=0
def read_test(path):
    global c
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readlines()
        for i in line:
            str = ''.join(letter for letter in i if letter not in ['.', ',', ':', '@', '[', '!', ']', '=', '&', '(', ')', '?', '"', '#', '$','{', '}', '\\', '/', '%'])
            count(str)
    c += 1

    # print(c)
for file_1 in neg:
    if file_1.endswith(".txt"):
        path = f"{neg_path}\{file_1}"
        read_data(path)
        # print(neg_data)
        # break
vocab_neg = vocab.copy()

# print('frequency of word "the" in negative vocab', vocab_neg['the'])
vocab=dict()
c=0
for file_2 in pos:
    if file_2.endswith(".txt"):
        path = f"{pos_path}\{file_2}"
        read_data(path)
vocab_pos = vocab.copy()
# print('frequency of word "the" in positive vocab', vocab_pos['the'])
# print(len(vocab_neg))


prob_neg = dict()
for j in vocab_neg:
    prob_j = (1+vocab_neg[j])/ (len(neg)+2)
    # print(f'The probability of word {j} is {prob_neg}')
    prob_neg.update({j : prob_j})

prob_pos = dict()
for j in vocab_pos:
    prob_j = (1+vocab_pos[j])/ (len(pos)+2)
    # print(f'The probability of word {j} is {prob_neg}')
    prob_pos.update({j : prob_j})
# print(prob_neg['good'])
# print(prob_pos['good'])


p_neg = len(neg)/25000
p_pos = 1 - p_neg

# print(p_pos,p_neg)


neg_test_path = os.path.join(testDir,'neg')
neg_test = os.listdir(neg_test_path)
pos_test_path = os.path.join(testDir,'pos')
pos_test = os.listdir(pos_test_path)
# print(len(neg_test))
# print(len(pos_test))


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



CM_Tr = [[0,0],[0,0]]

for file_1 in neg:
    vocab= dict()
    if file_1.endswith(".txt"):
        path = f"{neg_path}\{file_1}"
        read_data(path)
        p = prediction(vocab,[prob_neg,prob_pos])
        if p[0]>p[1]:
            CM_Tr[1][1]+=1
        else:
            CM_Tr[1][0]+=1

for file_2 in pos:
    vocab = dict()
    if file_2.endswith(".txt"):
        path = f"{pos_path}\{file_2}"
        read_data(path)
        p = prediction(vocab,[prob_neg,prob_pos])
        if p[0]<p[1]:
            CM_Tr[0][0]+=1
        else:
            CM_Tr[0][1]+=1

Accuracy_Tr = (np.trace(CM_Tr)*100)/25000
print(f'Accuracy over Train Data : {Accuracy_Tr}')



CM = [[0,0],[0,0]]

for test_file_1 in neg_test:
    vocab= dict()
    if test_file_1.endswith(".txt"):
        path = f"{neg_test_path}\{test_file_1}"
        read_test(path)
        p = prediction(vocab,[prob_neg,prob_pos])
        if p[0]>p[1]:
            CM[1][1]+=1
        else:
            CM[1][0]+=1

for test_file_2 in pos_test:
    vocab = dict()
    if test_file_2.endswith(".txt"):
        path = f"{pos_test_path}\{test_file_2}"
        read_test(path)
        p = prediction(vocab,[prob_neg,prob_pos])
        if p[0]<p[1]:
            CM[0][0]+=1
        else:
            CM[0][1]+=1


Accuracy = (np.trace(CM)*100)/15000
print(f'Accuracy over Test Data : {Accuracy}')

NegWC = WordCloud(
    width = 1200, height = 1000,
    background_color='white',
    stopwords=set()
)
# generate the word cloud
NegWC.generate(strneg)
plt.figure(figsize = (12,10), facecolor = None)
plt.imshow(NegWC)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('Negative-WordCount.png')
plt.show()


PosWC = WordCloud(
    width = 1200, height = 1000,
    background_color='white',
    stopwords=set()
)
# generate the word cloud
PosWC.generate(strpos)
plt.figure(figsize = (12,10), facecolor = None)
plt.imshow(PosWC)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('Positive-WordCount.png')
plt.show()
