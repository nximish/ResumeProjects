import math
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal
# print(sys.argv)
x_data = pd.read_csv(sys.argv[1] +'/X.csv', header=None)
y_data = pd.read_csv(sys.argv[1] + '/Y.csv', header= None)
# x_data = pd.read_csv("E:\IITD\COL774\Assignments\Assignment 1\data\q4\q4x.dat", sep= '  ', engine='python', header=None)
# y_data = pd.read_csv("E:\IITD\COL774\Assignments\Assignment 1\data\q4\q4y.dat", header= None)
x0 = np.array(x_data.iloc[:,0])
x0 = (x0 - x0.mean())/x0.std()
x1 = np.array(x_data.iloc[:,1])
x1 = (x1 - x1.mean())/x1.std()

X = np.array([x0,x1])

# Phi is the probability of ALaska in y_data
phi = 0
count_Alaska = 0

for i in y_data.iloc[:,0]:
    if i == 'Alaska':
        count_Alaska = count_Alaska + 1
count_Canada = len(y_data) - count_Alaska
phi = count_Alaska/len(y_data)

#Indicator function
#Calculating the means of two Gaussian Distributions

indicator0 = np.array([1 if i == 'Canada' else 0 for i in y_data.iloc[:,0]])
mu_0 = np.matmul(indicator0,X.T)/count_Canada
indicator1 = np.array([1 if i == 'Alaska' else 0 for i in y_data.iloc[:,0]])
mu_1 = np.matmul(indicator1,X.T)/count_Alaska
# print(f'mu_0:\n {mu_0}')
# print(f'mu_1:\n {mu_1}')

a0 = indicator0.reshape(100,1)*(X.T - mu_0)
a0 = np.matmul(a0.T,(X.T - mu_0))/count_Canada
# print(a0)
a1 = indicator1.reshape(100,1)*(X.T - mu_1)
a1 = np.matmul(a1.T,(X.T - mu_1))/count_Alaska
# print(a1)
sig_0 = np.array(a0)
sig_1 = np.array(a1)

# print(f'sig_0 = \n{sig_0}')
# print(f'sig_1 = \n{sig_1}')

sig = (X.T - mu_0).T
cov = np.cov(sig)
# print(f'sigma: \n{cov}')
# PLOTTING SCATTER POINTS

for i in range(len(y_data)):
    if y_data[0][i]=='Alaska':
        plt.scatter(x0[i],x1[i], marker='x', color='red', linewidths=2, alpha=0.5)
    else:
        plt.scatter(x0[i],x1[i], marker='.', color='blue', linewidths=2, alpha=0.5)

# PLOTTING THE LINEAR DECISION BOUNDARY

sig_inv = np.linalg.inv(cov)
w_0 = np.matmul(sig_inv,mu_0.T)
w_1 = np.matmul(sig_inv,mu_1.T)
wk_0 = np.matmul(mu_0,w_0)
wk_1 = np.matmul(mu_1,w_1)
m = -(w_0[0] - w_1[0])/(w_0[1] - w_1[1])
c = 0.5*(wk_0 - wk_1)/(w_0[1] - w_1[1])
y1 = -2*m + c
y2 = 2*m + c
plt.plot([-2,2],[y1,y2], c = 'green')
# plt.show()

# QUADRATIC DISCRIMINANT ANALYSIS

sig_inv_0 = np.linalg.inv(sig_0)
sig_inv_0 = sig_inv_0.reshape(2,2)
sig_inv_1 = np.linalg.inv(sig_1)
sig_inv_1 = sig_inv_1.reshape(2,2)
# print(f'sig_inv_0 = {sig_inv_0}')
# print(f'sig_inv_1 = {sig_inv_1}')
# v_0 = np.matmul(sig_inv_0,mu_0.T)
# print(f'\nv_0 = {v_0}')
# v_1 = np.matmul(sig_inv_1,mu_1.T)
# print(f'\nv_1 = {v_1}')
# vk_0 = np.matmul(mu_0,v_0)
# print(f'\nvk_0 = {vk_0}')
# vk_1 = np.matmul(mu_1,v_1)
# print(f'\nvk_1 = {vk_1}')
# vk = np.matmul(np.matmul((mu_0.T),sig_inv_0),mu_0)
# print(f'\nvk = {vk}')
# ln_sig = math.log(np.linalg.det(sig_0))
# print(f'\nln_sig = {ln_sig}')
# const = vk + ln_sig
# print(f'\nconst = {const}')
#
# for z in range(-4,4):
#     t = (-0.5)*(const + 1 + (sig_inv_0[0][0])*(z**2) + (v_0[0][0])*z)
#     s = v_0[1][0] + 2*sig_inv_0[0][1]*z
#     r = (-0.5)*sig_inv_0[1][1]
#     u = np.roots(r,s,t)
#     print(f'\nRoots : {u}')

x_test = pd.read_csv(sys.argv[2] +'/X.csv', header=None)
x_test = (x_test - x_test.mean())/x_test.std()
a_0 = x_test - mu_0
a_1 = np.matmul(sig_inv_0,a_0.T)
a_2 = np.matmul(a_1,a_0)
# print(a_2)
b_0 = x_test - mu_1
b_1 = np.matmul(sig_inv_1,b_0.T)
b_2 = np.matmul(b_1,b_0)
# print(b_2)
a_3 = np.linalg.det(a_2)
b_3 = np.linalg.det(b_2)
prob_a = -0.5*a_3 - 0.5 * math.log(np.linalg.det(sig_0)) + math.log(phi)
prob_b = -0.5*b_3 - 0.5 * math.log(np.linalg.det(sig_1)) + math.log(1-phi)
l = []
if prob_b > prob_a:
    l.append('Alaska')
else:
    l.append('Canada')
# print(as)
# p0 = -0.5 * np.matmul((x_test - mu_0).T,(np.matmul(sig_inv_0,(x_test - mu_0)))) - 0.5 * math.log(np.linalg.det(sig_0)) + math.log(phi)
# p1 = -0.5 * np.matmul((x_test - mu_1).T,(np.matmul(sig_inv_1,(x_test - mu_1)))) - 0.5 * math.log(np.linalg.det(sig_1)) + math.log(phi)
# l= []
# if p0 > p1:
#     l.append('Canada')
# else:
#     l.append('Alaska')
l = np.array(l)
print(l)
a = pd.DataFrame(l)
a.to_csv('result_4.txt', header= False, index= False)