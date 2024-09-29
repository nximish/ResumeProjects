import numpy as np
import pandas as pd
import os
import sys
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

train_path = "D:\\ass3\data\\train.csv"
test_path = "D:\\ass3\data\\test.csv"
val_path = "D:\\ass3\data\\val.csv"

#Cleansing the data, removing samples that are missing values

def Cleansing(path):
    dataframe = pd.read_csv(path)
    i = dataframe[((dataframe.Age == '?') | (dataframe.Shape == '?') | (dataframe.Margin == '?') | (dataframe.Density == '?'))].index
    df = dataframe.drop(i)
    inputs = df.drop(['Severity','BI-RADS assessment'], axis='columns')
    target = df['Severity']
    return(inputs,target)

#Finding Accuracy

model = tree.DecisionTreeClassifier()
train_inputs = Cleansing(train_path)[0]
train_target = Cleansing(train_path)[1]
test_inputs = Cleansing(test_path)[0]
test_target = Cleansing(test_path)[1]
val_inputs = Cleansing(val_path)[0]
val_target = Cleansing(val_path)[1]
model.fit(train_inputs,train_target)


def Accuracy(model,path):
    if 'train' in path:    
        BestAcc = 100*(model.score(train_inputs,train_target))
    elif 'test' in path:
        BestAcc = 100*(model.score(test_inputs,test_target))
    elif 'val' in path:
        BestAcc = 100*(model.score(val_inputs,val_target))
    return BestAcc



ccp_path = model.cost_complexity_pruning_path(train_inputs,train_target)
alphas = ccp_path.ccp_alphas
impurities = ccp_path.impurities

plt.plot(alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
plt.xlabel("Effective Alpha")
plt.ylabel("Total Impurity of Leaves")
plt.title("Total Impurity vs Effective Alpha for Training Set")
plt.show()

ccp_alphas = alphas[:-1]
trees = []
train_acc = []
test_acc = []
val_acc = []
for i in ccp_alphas:
    tree_i = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=i)
    tree_i.fit(train_inputs,train_target)
    trees.append(tree_i)
    train_acc.append(Accuracy(tree_i,train_path))
    test_acc.append(Accuracy(tree_i,test_path))
    val_acc.append(Accuracy(tree_i,val_path))


node_counts=[]
max_depth=[]
for tree_i in trees:
    node_counts.append(tree_i.tree_.node_count)
    max_depth.append(tree_i.tree_.max_depth)

plt.plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
plt.xlabel("Alpha")
plt.ylabel("Number of Nodes")
plt.title("Number of Nodes vs Alpha")
plt.tight_layout()
plt.show()

plt.plot(ccp_alphas, max_depth, marker="o", drawstyle="steps-post")
plt.xlabel("Alpha")
plt.ylabel("Depth of Tree")
plt.title("Depth vs Alpha")
plt.tight_layout()
plt.show()

plt.plot(ccp_alphas, train_acc, label='Training Accuracy')
plt.tight_layout()

plt.plot(ccp_alphas, test_acc,label='Testing Accuracy')
plt.tight_layout()

plt.plot(ccp_alphas, val_acc,label='Validation Accuracy')
plt.tight_layout()

plt.xlabel("Alpha")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Alpha")
plt.legend()
plt.show()

bestvalacc = max(val_acc)
index_bestvalacc = val_acc.index(bestvalacc)

plt.figure(figsize = (15,10))
dec_tree = tree.plot_tree(trees[index_bestvalacc], filled =True, rounded= True)
plt.show()

print(f'\nTraining Accuracy:{train_acc[index_bestvalacc]}\n')
print(f'\nTesting Accuracy:{test_acc[index_bestvalacc]}\n')
print(f'\nValidation Accuracy:{val_acc[index_bestvalacc]}\n')