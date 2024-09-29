import numpy as np
import pandas as pd
import time
import os
import sys
import pickle
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers
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
classk=[]
K=[]
AT=[]
for i in range(0,length):
    if(train_data['labels'][i]==3):
        # classk.append(np.asarray((train_data['data'][i].reshape(3072)),dtype='int32'))
        classk.append(np.asarray(list(map(lambda p:p/255,train_data['data'][i].reshape(3072))),dtype='float64'))
        AT.append(-1.0)
    if (train_data['labels'][i]==2):
        # classk.append(np.asarray((train_data['data'][i].reshape(3072)),dtype='int32'))
        classk.append(np.asarray(list(map(lambda p:p/255,train_data['data'][i].reshape(3072))),dtype='float64'))
        AT.append(1.0)
x_i=np.array(classk)
y_i=np.array(AT)


for i in range(0,len(classk)):
    if y_i[i] == -1.0:
        K.append(-1.0*classk[i])
    else:
        K.append(classk[i])


K = np.array(K)
P = np.matmul(K,K.T)
# print(P.shape)
q = np.full((4000,1),-1,dtype= 'int32')
#print(q)
G1 = np.identity(4000)
G2 = -G1
# print(G1)
# print(G2)
G = np.append(G2,G1).reshape(8000,4000)
#print(G)
H0 = np.zeros(4000)
HC = np.ones(4000)*1.0
H = np.append(H0,HC).T
#print(H)

A = y_i.T
#print(A.shape)
b=matrix([0.0])


#print('calculated')
P = matrix(P, tc= 'd')
q = matrix(q, tc= 'd')
h = matrix(H, tc= 'd')
G = matrix(G, tc= 'd')
A = matrix(A.reshape(1,4000))
b = matrix(b, tc= 'd')
sol = solvers.qp(P,q,G,h,A,b)
#solG = solvers.qp(PG,q,G,h,A,b)
l=np.ravel(sol['x'])
alpha= sorted(l)
top5alpha = alpha[-5:]
def display_img(x,index):
    x=np.array(x)
    x = x.reshape(32,32,3)
    plt.imshow(x)
    index = str(index)
    plt.savefig(index)

for i in range(len(l)):
    if l[i] in top5alpha:
        display_img(list(map(lambda p:int(p*255), x_i[i])),i)

#lG=np.ravel(solG['x'])
w=0
B=0
for i in range(len(l)):
    #print('xxx')
    if(l[i]>1e-10 and l[i]<=1.0):
        w+=K[i]*l[i]
c=0
for i in range(len(l)):
    #print('yyy')
    if(l[i]>1e-10 and l[i]<1.0):
        c+=1
        B+=y_i[i] - x_i[i].dot(w)

B/=len(l)

wmin = min(w)
wmax = max(w)
listofw = []
for i in range(len(w)):
    listofw.append(int((w[i]*255)/(max(w) - min(w))))
listofw = np.array(listofw)
listofw = listofw.reshape(32,32,3)
plt.imshow(listofw)
plt.savefig("W")


# Saving the parameters
with open('linearParameterMyModel','wb') as file:
    pickle.dump((w,b),file)
   
print(f' Time taken: {time.time() - start_time}')
print(f'w = {w}')
print(f'B = {B}')
print(f'No. of Support Vectors : {c}')



test_length=len(test_data['labels'])
class_k=[]
Test_K=[]
Test_AT=[]
for i in range(0,test_length):
    if(test_data['labels'][i]==3):
        class_k.append(np.asarray(list(map(lambda p:p/255,test_data['data'][i].reshape(3072))),dtype='float64'))
        Test_AT.append(-1.0)
    if(test_data['labels'][i]==2):
        class_k.append(np.asarray(list(map(lambda p:p/255,test_data['data'][i].reshape(3072))),dtype='float64'))
        Test_AT.append(1.0)


CM = [[0,0],[0,0]]

def predict(x,y):
    t=np.matmul(w,x.T)+b
    if y==1.0:
        if(t>=0):
            #True Positive
            CM[0][0]+=1
            return 1
        else:
            #False Negative
            CM[1][0]+=1
            return 0
    
    if y==-1.0:
        if(t<=0):
            #True Negative
            CM[1][1]+=1
            return 1
        else:
            #False Positive
            CM[0][1]+=1
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
print(f' Confusion Matrix: {CM}')
