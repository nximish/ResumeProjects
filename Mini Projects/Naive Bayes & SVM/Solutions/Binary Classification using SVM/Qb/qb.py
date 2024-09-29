import numpy as np
import pandas as pd
import os
import sys
import pickle
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers
from sklearn import svm
import time
# from sklearn import metrics

trainDir=sys.argv[1]
testDir=sys.argv[2]

file = open(os.path.join(trainDir,'train_data.pickle'),'rb')
train_data=pickle.load(file)

file = open(os.path.join(testDir,'test_data.pickle'),'rb')
test_data=pickle.load(file)

start_time = time.time()


def Kernel(a,b,gamma):
    Norm = np.linalg.norm(a - b)**2
    G = np.exp(-gamma*Norm)
    return G

def svmGaussian(x_i,y_i):
    gamma = 0.001
    Arr = np.zeros((len(x_i)*len(x_i))).reshape(len(x_i),len(x_i))
    for i in range(0,len(classk)):
        for j in range(0,len(classk)):
            Arr[i][j] = Kernel(classk[i],classk[j],gamma)

    KG = np.outer(y_i,y_i)*Arr
    PG = matrix(KG)
    size = len(y_i)
    q = np.full((size,1),-1,dtype= 'int32')
    # print(q)
    G1 = np.identity(size)
    G2 = -G1
    G = np.append(G2,G1).reshape(2*size,size)
    # print(G)
    H0 = np.zeros(size)
    HC = np.ones(size)*1.0
    H = np.append(H0,HC).T
    # print(H)

    A = y_i.T
    # print(A.shape)
    b=matrix([0.0])


    # print('calculated')
    q = matrix(q, tc= 'd')
    h = matrix(H, tc= 'd')
    G = matrix(G, tc= 'd')
    A = matrix(A.reshape(1,size))
    b = matrix(b, tc= 'd')
    solG = solvers.qp(PG,q,G,h,A,b)
    lG=np.ravel(solG['x'])
    # sv = lG in range(0,1)
    # index = np.arange(len(lG))[sv]
    alpha= sorted(lG)
    top5alpha = alpha[-5:]
    for i in range(len(lG)):
        if lG[i] in top5alpha:
            temp = map(lambda p:int(p*255), x_i[i])
            temp = list(temp)
            display_img(temp,i)

    listSV = []
    indices = []
    outputs = []
    alphas = []
    B=0
    for i in range(len(lG)):
        if(lG[i]>=0 and lG[i]<1.0):
            alphas.append(lG[i])
            listSV.append(x_i[i])
            indices.append(i)
            outputs.append(y_i[i])

    for i in range(len(alphas)):
        a=0
        for j in range(len(alphas)):
            a+= alphas[j]*outputs[j]*Arr[indices[i],indices[j]]
        B+= outputs[i] - a
    B/=len(alphas)
    print(f'B = {B}')
    print(f'No. of Support Vectors : {len(alphas)}')
    return (B,alphas,listSV,outputs)


def display_img(x,index):
    x=np.array(x)
    x = x.reshape(32,32,3)
    plt.imshow(x)
    index = str(index)
    plt.savefig(index)

length=len(train_data['labels'])
classk=[]
AT=[]
for i in range(0,length):
    if(train_data['labels'][i]==2):
        temp=map(lambda p: p/255.0,train_data['data'][i].reshape(3072)) 
        temp=np.asarray(list(temp),dtype='float64')
        classk.append(temp)
        # classk.append(np.asarray(list(map(lambda p:p/255.0,train_data['data'][i].reshape(3072))),dtype='float64'))
        AT.append(-1.0)
    if (train_data['labels'][i]==3):
        temp=map(lambda p: p/255.0,train_data['data'][i].reshape(3072)) 
        temp=np.asarray(list(temp),dtype='float64')
        classk.append(temp)
        # classk.append(np.asarray(list(map(lambda p:p/255.0,train_data['data'][i].reshape(3072))),dtype='float64'))
        AT.append(1.0)
x_i=np.array(classk)
y_i=np.array(AT)
m=svmGaussian(x_i,y_i)
print(f'Time taken: {time.time() - start_time}')


test_length=len(test_data['labels'])
class_k=[]
Test_K=[]
Test_AT=[]
for i in range(0,test_length):
    if(test_data['labels'][i]==2):
        temp=map(lambda p: p/255.0,test_data['data'][i].reshape(3072)) 
        temp=np.asarray(list(temp),dtype='float64')
        class_k.append(temp)
        # class_k.append(np.asarray(list(map(lambda p:p/255.0,test_data['data'][i].reshape(3072))),dtype='float64'))
        Test_AT.append(-1.0)
    if(test_data['labels'][i]==3):
        temp=map(lambda p: p/255.0,test_data['data'][i].reshape(3072)) 
        temp=np.asarray(list(temp),dtype='float64')
        class_k.append(temp)
        # class_k.append(np.asarray(list(map(lambda p:p/255.0,test_data['data'][i].reshape(3072))),dtype='float64'))
        Test_AT.append(1.0)


CM = [[0,0],[0,0]]

def predict(x,y):
    var = 0
    for i in range(len(m[1])):
        var+= m[1][i]*m[3][i]*Kernel(x,m[2][i],0.001)
    var+= m[0]
    if y==-1.0:
        if var<=0:
            CM[1][1]+=1
            return 1
        else:
            CM[0][1]+=1
            return 0    
    if y==1.0:
        if var>0:
            CM[0][0]+=1
            return 1
        else:
            CM[1][0]+=1
            return 0


x_i_test = np.array(class_k)
y_i_test = np.array(Test_AT)

c=0
for i in range(len(x_i_test)):
    c+=predict(x_i_test[i],y_i_test[i])
Acc = (c*100)/len(x_i_test)
print(f'No. of Correct Predictions : {c}')
print(f'No. of Incorrect Predictions : {len(x_i_test)-c}')
print(f'Test Set Accuracy = {Acc}')
print(f'Confusion Matrix : {CM}')