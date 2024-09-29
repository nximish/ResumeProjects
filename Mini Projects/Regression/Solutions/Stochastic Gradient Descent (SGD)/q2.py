import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
# testDataset = pd.read_csv("E:\IITD\COL774\Assignments\Assignment 1\data\q2\q2test.csv")
testDataset = pd.read_csv(sys.argv[1]+'/X.csv',header=None)
st = time.time()
Theta_0 = 3
Theta_1 = 1
Theta_2 = 2
x1 = np.random.normal(3,2,1000000)
x2 = np.random.normal(-1,2,1000000)
epsilon = np.random.normal(0,math.sqrt(2),1000000)
y = Theta_0 + Theta_1*x1 + Theta_2*x2 + epsilon
# print(y)
ita = 0.001
m= 1000000
b = 1
outputThetas=dict()

def sgd(batchsize = 1,fu=0.000000001):
    global outputThetas
    # print(fu)
    st1 = time.time()
    st=time.time()
    # fu = 0.000000001
    prev_cost = 0.0
    prev_theta_0 = 0.0
    prev_theta_1 = 0.0
    prev_theta_2 = 0.0
    theta_0 = 0.0
    theta_1 = 0.0
    theta_2 = 0.0
    count = 0
    i = 0
    t_0=[]
    t_1=[]
    t_2=[]

    while True:
        x_1 = []
        x_2 = []
        y_new = []
        count = count+1

        for j in range(0,batchsize):

            x_1.append(x1[i])
            x_2.append(x2[i])
            y_new.append(y[i])
            i = (i+1)%len(x1)
        x_1 = np.array(x_1)
        x_2 = np.array(x_2)
        y_new = np.array(y_new)
        h_theta = theta_0 + x_1*theta_1 + x_2*theta_2
        temp = h_theta - y_new
        cost_function = (sum(pow(temp,2)))*(0.5/batchsize)
        # print(cost_function)
        ita=1.0/(0.1*count+30)
        #differentiation of cost function gives the values of theta 0, theta 1 and theta_2 initially for the first iteration.
        t_0.append(theta_0)
        t_1.append(theta_1)
        t_2.append(theta_2)
        theta_0 = (theta_0 - ita*(sum(temp)/batchsize))
        theta_1 = (theta_1 - ita*(sum(temp * x_1)/batchsize))
        theta_2 = (theta_2 - ita*(sum(temp * x_2)/batchsize))

        # Th_0.append(theta_0[0])
        # Th_1.append(theta_1[0])
        # # Th_2.append(theta_2[0])
        # Cost.append(cost_function[0])
        # if (abs(prev_theta_0 - theta_0) < 0.25/k) and (abs(prev_theta_1 - theta_1)<0.25/k) and (abs(prev_theta_2 - theta_2)<0.25/k):
        #
        if (time.time()-st1 > 10):
            # print(f'The number of Iterations : {count}')
            # print(f'thetas = {theta_0, theta_1, theta_2}')
            # fu*=10
        #     print("again")
            st1=time.time()
        # if abs(prev_cost - cost_function) < fu*batchsize:
        if (abs(prev_theta_0 - theta_0) < fu*batchsize) and (abs(prev_theta_1 - theta_1) < fu*batchsize) and (abs(prev_theta_2 - theta_2) < fu*batchsize):
            # print(f'\n\nFor batchsize {batchsize}, the values of theta converge to :')
            # print(f'theta_0,theta_1,theta_2 = {theta_0,theta_1,theta_2}')
            # print(f'The number of Iterations : {count}')
            # print(f'Time taken = {time.time() - st}')
            # print(f'Error : {cost_function}')
            # print(f'change in parameters: = {abs(Theta_0 - theta_0)},{abs(Theta_1 - theta_1)},{abs(Theta_2 - theta_2)}')
            outputThetas[batchsize]=[theta_0, theta_1, theta_2]
            break
        prev_theta_0 = theta_0
        prev_theta_1 = theta_1
        prev_theta_2 = theta_2
        # prev_cost = cost_function
    # fig=plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(1,1,1, projection = '3d')
    # ax.plot(t_0,t_1,t_2, marker = '.', color = 'black')
    # plt.show()

# CALCULATING THE ROOT MEAN SQUARE ERROR FOR PREDICTED Y VALUES FOR GIVEN Y IN THE DATASET

    # y_given = dataset.iloc[:,2]
    # z1 = dataset.iloc[:,0]
    # z2 = dataset.iloc[:,1]
    # y_predicted = theta_0 + theta_1*z1 + theta_2*z2
    # abs_error = abs(y_predicted - y_given)
    # sq_error = np.square(abs_error)
    # MSE = sq_error.mean()
    # RMSE = np.sqrt(MSE)
    # print(f'The error in prediction of y using learned model: {RMSE}')
    # fig.savefig(str(batchsize))
# sgd(batchsize=1,fu=0.000001)
sgd(batchsize=100,fu=0.000001)
# sgd(batchsize=10000,fu=0.0000000025)
# sgd(batchsize=1000000,fu=0.000000001)



# test_data = pd.read_csv(sys.argv[2]+"X.csv")
y_test = outputThetas[100][0] + outputThetas[100][1]*testDataset[0] + outputThetas[100][2]*testDataset[1]
a = pd.DataFrame(y_test)
a.to_csv('result_2.txt', header= False, index= False)