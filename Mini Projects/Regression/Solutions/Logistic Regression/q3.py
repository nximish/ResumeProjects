import math
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
# x = pd.read_csv("E:\IITD\COL774\Assignments\Assignment 1\data\q3\logisticX.csv", header=None)
# y = pd.read_csv("E:\IITD\COL774\Assignments\Assignment 1\data\q3\logisticY.csv", header=None)
x = pd.read_csv(sys.argv[1]+"\X.csv", header=None)
y = pd.read_csv(sys.argv[1]+"\Y.csv", header=None)
ita = 0.001
theta = np.array([0.0,0.0,0.0])
theta_0 = theta[0]
theta_1 = theta[1]
theta_2 = theta[2]
x1 = np.array(x.iloc[:,0])
x1 = (x1 - x1.mean())/(x1.std())
x2 = np.array(x.iloc[:,1])
x2 = (x2 - x2.mean())/(x2.std())
x0 = np.ones(len(x1))
X = np.array([x0,x1,x2])
y=np.array(y.iloc[:,0])
prev_cost = 0.0
while True:
# for i in range(5):
    z = theta[0] + x1*theta[1] + x2*theta[2]
    h_theta = 1.0/(1.0 + (math.e**(-z)))
    # print(h_theta.shape)
    log_likelihood = np.multiply(np.log(h_theta),y) + np.multiply(np.log(1-h_theta),1-y)
    # print(log_likelihood)
    cost = sum(log_likelihood)
    # print(cost)
    del_J = np.matmul(X,(y - h_theta))
    # print((y - h_theta).shape)
    # print(del_J)


    #Calculating the Hessian

    W = h_theta*(1-h_theta)
    D = W*X
    Hessian = np.matmul(D,X.T)
    # print(Hessian)
    H_inv = np.linalg.inv(Hessian)
    # print(H_inv)
    theta = theta + np.matmul(H_inv,del_J.T)
    # print(f'\nThetas : {theta}')

    if abs(prev_cost - cost) < 0.000000001:
        # print(theta)
        break
    prev_cost = cost

for i in range(len(x1)):
    if y[i]==1:
        plt.scatter(x1[i],x2[i], marker='.', color='blue', alpha=0.5)
    else:
        plt.scatter(x1[i],x2[i], marker='x', color='red', alpha=0.5)

# z2=theta[0] + theta[2]*x2 # we are fixing the value of x2 as 2 since this is the equation of a plane
z1=-(theta[0] + theta[1]*x1)/(theta[2]) # we are fixing the value of x1 as 1 since this is the equation of a plane
plt.plot(x1,z1, color='green')
# plt.plot(x2,z2)
# plt.show()

testData = pd.read_csv(sys.argv[2]+"\X.csv", header=None)

x1 = np.array(testData.iloc[:,0])
x1 = (x1 - x1.mean())/(x1.std())
x2 = np.array(testData.iloc[:,1])
x2 = (x2 - x2.mean())/(x2.std())
x0 = np.ones(len(x1))
X = np.array([x0,x1,x2])
z = theta[0] + x1 * theta[1] + x2 * theta[2]
h_theta = 1.0 / (1.0 + (math.e ** (-z)))
l = []
for i in h_theta:
    if i > 0.5:
        l.append(1)
    else:
        l.append(0)
l = np.array(l)
a = pd.DataFrame(l)
a.to_csv('result_3.txt', header= False, index= False)