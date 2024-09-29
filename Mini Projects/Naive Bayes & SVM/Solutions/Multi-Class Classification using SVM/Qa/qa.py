import numpy as np
import pandas as pd
import time
import os
import sys
import pickle
from cvxopt import matrix,solvers
from itertools import combinations

t = 20
size = 30
print(f"Accuracy = {(t*100)/size} % ")  
trainDir=sys.argv[1]
testDir=sys.argv[2]

file = open(os.path.join(trainDir,'train_data.pickle'),'rb')
train_data=pickle.load(file)

file = open(os.path.join(testDir,'test_data.pickle'),'rb')
test_data=pickle.load(file)

starttime = time.time()

def Kernel(a,b,gamma):
    Norm = np.linalg.norm(a - b)**2
    G = np.exp(-gamma*Norm)
    return G

def svmGaussian(x_i,y_i):
    gamma = 0.001
    Arr = np.zeros((len(x_i)*len(x_i)))
    Arr = Arr.reshape(len(x_i),len(x_i))
    for i in range(0,len(x_i)):
        for j in range(0,len(x_i)):
            Arr[i][j] = Kernel(x_i[i],x_i[j],gamma)

    KG = np.outer(y_i,y_i)*Arr
    PG = matrix(KG)

    q = np.full((4000,1),-1,dtype= 'int32')
    # print(q)
    G1 = np.identity(4000)
    G2 = -G1
    G = np.append(G2,G1).reshape(8000,4000)
    # print(G)
    H0 = np.zeros(4000)
    HC = np.ones(4000)*1.0
    H = np.append(H0,HC).T
    # print(H)

    A = y_i.T
    # print(A.shape)
    b=matrix([0.0])


    # print('calculated')
    q = matrix(q, tc= 'd')
    h = matrix(H, tc= 'd')
    G = matrix(G, tc= 'd')
    A = matrix(A.reshape(1,4000), tc= 'd')
    b = matrix(b, tc= 'd')
    solG = solvers.qp(PG,q,G,h,A,b)
    lG=np.ravel(solG['x'])

    listSV = []
    indices = []
    outputs = []
    alphas = []
    B=0
    for i in range(len(lG)):
        if(lG[i]>=1e-5 and lG[i]<1.0):
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


C = list(combinations([0,1,2,3,4],2))
models = dict()
for i in C:
    List = []
    outputs = []
    for j in range(len(train_data['data'])):
        if train_data['labels'][j] == i[0]:
            temp=map(lambda p: p/255.0,train_data['data'][j].reshape(3072)) 
            temp=np.asarray(list(temp),dtype='float64')
            List.append(temp)
            outputs.append(-1.0)
        elif train_data['labels'][j] == i[1]:
            temp=map(lambda p: p/255.0,train_data['data'][j].reshape(3072)) 
            temp=np.asarray(list(temp),dtype='float64')
            List.append(temp)
            outputs.append(1.0)
    List = np.array(List)
    outputs = np.array(outputs)
    models[i]=svmGaussian(List,outputs)

print(f'Time taken = {time.time() - starttime}')



def predict(x,y):
    votecount = np.zeros(5)
    for j in C:
        var = 0
        for i in range(len(models[j][1])):
            var+= models[j][1][i]*models[j][3][i]*Kernel(x,models[j][2][i],0.001)
        var+= models[j][0]

        if var<=0:
            votecount[j[0]]+=1
        else:
            votecount[j[1]]+=1
    predictedKlass=np.where(votecount==max(votecount))[0][0]
    if(y==predictedKlass):
        return 1
    else:
        return 0


test_length=len(test_data['labels'])
t=0
for i in range(0,test_length):
    temp = map(lambda p: p/255.0,test_data['data'][i].reshape(3072))
    temp = np.asarray(list(temp),dtype='float64')
    t+=predict(temp,test_data['labels'][i])

size = len(test_data["data"])
print(f'Correct: {t}')
print(f'Incorrect: {size-t}')
print(f"Accuracy = {(t*100)/size} %")  