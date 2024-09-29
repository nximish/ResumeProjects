import numpy as np
import pandas as pd
import os
import time
import sys
import pickle
# from cvxopt import matrix,solvers
from sklearn import svm
# from sklearn import metrics

trainDir=sys.argv[1]
testDir=sys.argv[2]

file = open(os.path.join(trainDir,'train_data.pickle'),'rb')
train_data=pickle.load(file)

file = open(os.path.join(testDir,'test_data.pickle'),'rb')
test_data=pickle.load(file)

start_time = time.time()

length=len(train_data['labels'])
trainclass = []
AT=[]
for i in range(0,length):
    if(train_data['labels'][i]==2):
        trainclass.append(np.asarray(list(map(lambda p:p/255,train_data['data'][i].reshape(3072))),dtype='float64'))
        # trainclass.append(np.asarray((train_data['data'][i].reshape(3072)),dtype='int32'))
        AT.append(-1.0)
    if (train_data['labels'][i]==3):
        trainclass.append(np.asarray(list(map(lambda p:p/255,train_data['data'][i].reshape(3072))),dtype='float64'))
        # trainclass.append(np.asarray((train_data['data'][i].reshape(3072)),dtype='int32'))
        AT.append(1.0)
x_i = np.array(trainclass)
y_i=np.array(AT)


#USING SK-LEARN

model = svm.SVC(kernel= 'linear')
model.fit(x_i,y_i)
print(f'Time taken by Linear Model : {time.time() - start_time}')
# Saving the parameters
with open('linearParameter','wb') as file:
    pickle.dump((model.coef_[0],model.intercept_[0]),file)
   
t1 = model.predict(x_i)
# for i in t1:
#     print(i,end=(', '))
# t2 = model2.predict(x_i)
correct = 0
for i in range(len(t1)):
    if t1[i]==int(train_data['labels'][i]):
        correct+=1
Acc = (correct*100)/len(t1)


test_length=len(test_data['labels'])
testclass = []
test_AT=[]
for i in range(0,test_length):
    if(test_data['labels'][i]==2):
        testclass.append(np.asarray(list(map(lambda p:p/255,test_data['data'][i].reshape(3072))),dtype='float64'))
        # testclass.append(np.asarray((test_data['data'][i].reshape(3072)),dtype='int32'))
        test_AT.append(-1.0)
    if (test_data['labels'][i]==3):
        testclass.append(np.asarray(list(map(lambda p:p/255,test_data['data'][i].reshape(3072))),dtype='float64'))
        # testclass.append(np.asarray((test_data['data'][i].reshape(3072)),dtype='int32'))
        test_AT.append(1.0)
x_i_test = np.array(testclass)
y_i_test = np.array(test_AT)



t_test = model.predict(x_i_test)
Test_correct_L = 0
for i in range(len(t_test)):
    if t_test[i]==int(y_i_test[i]):
        Test_correct_L+=1
Test_Acc_L = (Test_correct_L*100)/len(t_test)

#GAUSSIAN KERNEL

start_time_gaussian = time.time()

model2 = svm.SVC(kernel= 'rbf', C=1.0, gamma=0.001)
model2.fit(x_i,y_i)
print(f'Time taken by Gaussian Model : {time.time() - start_time_gaussian}')

tG = model2.predict(x_i)
correct = 0
for i in range(len(tG)):
    if tG[i]==int(train_data['labels'][i]):
        correct+=1
Acc = (correct*100)/len(tG)


test_t_G = model2.predict(x_i_test)

test_correct_G = 0
for i in range(len(test_t_G)):
    if test_t_G[i]==int(y_i_test[i]):
        test_correct_G+=1
Test_Acc_G = (test_correct_G*100.0)/len(test_t_G)


print('\nLINEAR MODEL :')
print(f'Co-eff (w) = {model.coef_[0]}')
print(f'Intercept (b) = {model.intercept_[0]}')
print(f'No. of support vectors = {len(model.support_vectors_)}')
print(f'Correct Predictions : {Test_correct_L}')
print(f'Incorrect Predictions : {len(t_test) - Test_correct_L}')
print(f'Test Data Accuracy : {Test_Acc_L}')

print('\nGAUSSIAN MODEL :')
print(f'No. of support vectors = {len(model2.support_vectors_)}')
print(f'Correct Predictions : {test_correct_G}')
print(f'Incorrect Predictions : {len(test_t_G) - test_correct_G}')
print(f'Test Data Accuracy : {Test_Acc_G}')

