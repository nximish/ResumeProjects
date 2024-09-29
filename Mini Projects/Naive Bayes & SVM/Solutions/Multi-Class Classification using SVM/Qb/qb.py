from re import S
import numpy as np
import pandas as pd
import time
import os
import sys
import pickle
from itertools import combinations
from sklearn import svm


trainDir=sys.argv[1]
testDir=sys.argv[2]

file = open(os.path.join(trainDir,'train_data.pickle'),'rb')
train_data=pickle.load(file)

file = open(os.path.join(testDir,'test_data.pickle'),'rb')
test_data=pickle.load(file)

start_time = time.time()

images = []
for i in range(len(train_data['data'])):
    flatdata = list(map(lambda p:p/255, train_data['data'][i].reshape(3072)))
    images.append(flatdata)

Model = svm.SVC(kernel='rbf',gamma=0.001,C=1.0,verbose=True, decision_function_shape='ovo')
Model.fit(images,np.array(train_data['labels']))

# Storing models    
with open('ModelGaussian','wb') as f:
    pickle.dump(Model,f)

# Storing models    
with open('ModelGaussian','rb') as f:
    Model = pickle.load(f)


test_images = []
for i in range(len(test_data['data'])):
    flat_data = list(map(lambda p:p/255, test_data['data'][i].reshape(3072)))
    test_images.append(flat_data)


a=0
Prediction = Model.predict(np.array(test_images, dtype= 'float64'))
for i in range(len(Prediction)):
    if Prediction[i] == test_data['labels'][i]:
        a+=1

Acc = (a*100)/len(Prediction)
print(f'Test Data Accuracy = {Acc}')
print(f'Time Taken: {time.time() - start_time}')

