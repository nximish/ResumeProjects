import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import time
import threading
#x_train is the acidity of the wine
#y_train is the density of the wine
x_train = pd.read_csv(sys.argv[1]+'/X.csv', names=['Acidity'])
y_train = pd.read_csv(sys.argv[1]+'/Y.csv', names=['Density'])
# x_train = pd.read_csv("E:\IITD\COL774\Assignments\Assignment 1\data\q1\linearX.csv", names=['Acidity'])
# y_train = pd.read_csv("E:\IITD\COL774\Assignments\Assignment 1\data\q1\linearY.csv", names=['Density'])
x_train = np.array(x_train)
y_train = np.array(y_train)
count = 0
m = len(x_train)
ita = 0.1
theta_0 = 0.0
theta_1 = 0.0
prev_cost = 0.0
#Normalising x
x_train = (x_train-x_train.mean())/(x_train.std())
Th_0 = [0.0]
Th_1 = [0.0]
Cost = []
st=time.time()




while True:
    h_theta = theta_0 + x_train*theta_1
    temp = h_theta - y_train
    cost_function = (sum(pow(temp,2)))*(0.5/m)
    # print(cost_function)
    #differentiation of cost function gives the values of theta 0 and theta 1 initially for the first iteration.
    theta_0 = (theta_0 - ita*(sum(temp)/m))
    theta_1 = (theta_1 - ita*(sum(temp * x_train)/m))
    Th_0.append(theta_0[0])
    Th_1.append(theta_1[0])
    Cost.append(cost_function[0])
    count = count+1
    if abs(prev_cost - cost_function) < 0.000000001:
        # print('Done')
        # print(f'No. of iterations = {count}')
        # print(f'Theta_0 = {theta_0}\nTheta_1 = {theta_1}\nCost Function = {cost_function}')
        break
    prev_cost = cost_function

x_test = pd.read_csv(sys.argv[2]+'/X.csv',header=None)
y_predicted = theta_0 + theta_1*x_test
#print(y_predicted)
a = pd.DataFrame(y_predicted)
a.to_csv('result_1.txt', header= False, index= False)
# GRADIENT DESCENT COMPLETED
# print(time.time()-st)
# # Creating Linear Regression 2d Plot
#
# plt.plot(x_train,h_theta)
# plt.scatter(x_train, y_train, marker='x', c = 'r')
# plt.title("Density of Wine")
# plt.ylabel('Density of Wine')
# plt.xlabel('Acidity of Wine')
# plt.show()
#
# # Creating a 3d Surface Plot
#
# N = 100
# theta_0_values = np.linspace(-10,10,N)
# theta_1_values = np.linspace(-10,10,N)
# Theta_0, Theta_1 = np.meshgrid(theta_0_values, theta_1_values)
# # print(f'Theta_0 = {Theta_0}\n')
# # print(f'Theta_1 = {Theta_1}\n')
# Arr = []
# for i in range(0,N):
#     for j in range(0,N):
#         H_Theta = Theta_0[i,j] + x_train * Theta_1[i,j]
#         Temp = H_Theta - y_train
#         Cost_Fun = (sum(pow(Temp, 2))) * (0.5 / m)
#         Arr.append(Cost_Fun)
# Arr2 = np.array(Arr)
# J_Theta = Arr2.reshape(N,N)
# # print(J_Theta)
# plt.figure(figsize=(11, 11))
# ax = plt.axes(projection = '3d')
# ax.set_xlabel('Theta_0')
# ax.set_ylabel('Theta_1')
# ax.set_zlabel('J_Theta')
# ax.plot_surface(Theta_0,Theta_1,J_Theta,alpha=0.5, rstride=1, cstride=1, cmap= 'viridis', edgecolor= 'none')
#
#
# def animate():
#     global ax
#     time.sleep(2)
#     for i in range(len(Th_0)):
#         # sleeptime = 0.2 / (i + 1)
#         # time.sleep(sleeptime)
#         ax.plot(Th_0[0:i],Th_1[0:i],Cost[0:i],'black', marker='.')
#         #
#         # fig.savefig("C:\\Users\\Naimish\\PycharmProjects\\pythonProject\\Animesan\\fig" + str(i))
#         plt.show()
#
#         #check if the window has been closed and if yes then break from the loop
# # animate()
#
# t1=threading.Thread(target=animate)
# t1.start()
# plt.show()
#
# t1.join()
#
# #Drawing the contours
#
# ax=plt.axes()
# cs=ax.contour(Theta_0,Theta_1,J_Theta)
# ax.clabel(cs,inline=1)
# # plt.show()
# def animate2():
#     global ax
#     for i in range(len(Th_0)):
#         # sleeptime=0.2/(i+1)
#         # time.sleep(sleeptime)
#         ax.plot(Th_0[0:i], Th_1[0:i], 'black', marker='.')
#         plt.show()
#
# t1=threading.Thread(target=animate2)
# t1.start()
# plt.show()
