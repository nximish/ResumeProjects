from operator import contains
import numpy as np
import pandas as pd
import statistics
import pickle
import os
import sys
from sklearn import tree
from sklearn import ensemble
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
# %matplotlib inline

train_path = "D:\\ass3\data\\train.csv"
test_path = "D:\\ass3\data\\test.csv"
val_path = "D:\\ass3\data\\val.csv"


def dataset_1_a():
    global out_str
    out_str = ""
    out_str += 'DATASET-1) (a)'
    
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

    def Accuracy(path):
        global out_str
        if 'train' in path:    
            BestAcc = 100*(model.score(train_inputs,train_target))
            out_str += f'\nAccuracy on Training dataset: {BestAcc} %'
        elif 'test' in path:
            BestAcc = 100*(model.score(test_inputs,test_target))
            out_str += f'\nAccuracy on Training dataset: {BestAcc} %'
        elif 'val' in path:
            BestAcc = 100*(model.score(val_inputs,val_target))
            out_str += f'\nAccuracy on Training dataset: {BestAcc} %'

    Accuracy(train_path)
    Accuracy(test_path)
    Accuracy(val_path)

    #Visualising the Decision Tree

    plt.figure(figsize = (15,10))
    dec_tree = tree.plot_tree(model, filled =True, rounded= True)
    # plt.show()

    with open('2_a.txt', 'w') as f:
        f.write(out_str)


def dataset_1_b():
    global out_str
    out_str = ""
    out_str += 'DATASET-1) (b)'
    
    
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

    dtree = GridSearchCV(model,{'max_depth' : [5,8,10,12,15,18,20],'min_samples_split' : [2,4,6,8,10],'min_samples_leaf' : [1,2,3]},cv=5,return_train_score=False)
    dtree.fit(train_inputs,train_target)
    df = pd.DataFrame(dtree.cv_results_)
    bestmodel = dtree.best_estimator_
    bestmodel.fit(train_inputs,train_target)

    def Accuracy(path):
        global out_str
        if 'train' in path:    
            BestAcc = 100*(bestmodel.score(train_inputs,train_target))
            out_str += f'\nAccuracy on Training dataset: {BestAcc} %'
        elif 'test' in path:
            BestAcc = 100*(bestmodel.score(test_inputs,test_target))
            out_str += f'\nAccuracy on Training dataset: {BestAcc} %'
        elif 'val' in path:
            BestAcc = 100*(bestmodel.score(val_inputs,val_target))
            out_str += f'\nAccuracy on Training dataset: {BestAcc} %'


    Accuracy(train_path)
    Accuracy(test_path)
    Accuracy(val_path)

    #Visualising the Decision Tree

    plt.figure(figsize = (15,10))
    dec_tree = tree.plot_tree(bestmodel, filled =True, rounded= True)
    # plt.show()

    with open('2_b.txt', 'w') as f:
        f.write(out_str)

def dataset_1_c():    
    global out_str
    out_str = ""
    out_str += 'DATASET-1) (c)'
    
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
    # plt.show()

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
    # plt.show()

    plt.plot(ccp_alphas, max_depth, marker="o", drawstyle="steps-post")
    plt.xlabel("Alpha")
    plt.ylabel("Depth of Tree")
    plt.title("Depth vs Alpha")
    plt.tight_layout()
    # plt.show()

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
    # plt.show()

    bestvalacc = max(val_acc)
    index_bestvalacc = val_acc.index(bestvalacc)

    plt.figure(figsize = (15,10))
    dec_tree = tree.plot_tree(trees[index_bestvalacc], filled =True, rounded= True)
    # plt.show()

    out_str += f'\nTraining Accuracy:{train_acc[index_bestvalacc]}'
    out_str += f'\nTesting Accuracy:{test_acc[index_bestvalacc]}'
    out_str += f'\nValidation Accuracy:{val_acc[index_bestvalacc]}'
    with open('2_c.txt', 'w') as f:
        f.write(out_str)


def dataset_1_d():
    global out_str
    out_str = ""
    out_str += 'DATASET-1) (d)'
    
    def Cleansing(path):
        dataframe = pd.read_csv(path)
        i = dataframe[((dataframe.Age == '?') | (dataframe.Shape == '?') | (dataframe.Margin == '?') | (dataframe.Density == '?'))].index
        df = dataframe.drop(i)
        inputs = df.drop(['Severity','BI-RADS assessment'], axis='columns')
        target = df['Severity']
        return(inputs,target)

    #Finding Accuracy


    model = ensemble.RandomForestClassifier(oob_score=True)
    train_inputs = Cleansing(train_path)[0]
    train_target = Cleansing(train_path)[1]
    test_inputs = Cleansing(test_path)[0]
    test_target = Cleansing(test_path)[1]
    val_inputs = Cleansing(val_path)[0]
    val_target = Cleansing(val_path)[1]
    model.fit(train_inputs,train_target)


    randomforest = GridSearchCV(model,{'n_estimators' : [20,40,60,80,100],'min_samples_split' : [2,4,6,8,10],'max_features' : ['sqrt',2,3,4]},return_train_score=False, refit=True,)
    op = randomforest.fit(train_inputs,train_target)
    df = pd.DataFrame(randomforest.cv_results_)
    bestestimator = randomforest.best_estimator_


    def Accuracy(path):
        global out_str
        if 'train' in path:    
            BestAcc = 100*(model.score(train_inputs,train_target))
            out_str += f'\nAccuracy on Training dataset: {BestAcc} %'
        elif 'test' in path:
            BestAcc = 100*(model.score(test_inputs,test_target))
            out_str += f'\nAccuracy on Testing dataset: {BestAcc} %'
        elif 'val' in path:
            BestAcc = 100*(model.score(val_inputs,val_target))
            out_str += f'\nAccuracy on Validation dataset: {BestAcc} %'


    Accuracy(train_path)
    Accuracy(test_path)
    Accuracy(val_path)
    oob = 100*(model.oob_score_)
    out_str += f'\nOut-Of-Bag Accuracy: {oob} %'

    with open('2_d.txt', 'w') as f:
        f.write(out_str)


#Cleansing the data, removing samples that are missing values
def dataset_1_e():
    global out_str
    out_str = ""
    out_str += 'DATASET-1) (e)'

    def Cleansing_Median(path):
        dataframe = pd.read_csv(path)
        i = dataframe[((dataframe.Age == '?') | (dataframe.Shape == '?') | (dataframe.Margin == '?') | (dataframe.Density == '?'))].index
        df = dataframe.drop(i)
        inputs = df.drop(['Severity','BI-RADS assessment'], axis='columns')
        median_inputs = inputs.median()
        dataframe_new = pd.read_csv(path)
        df2 = dataframe_new.replace("?",median_inputs)
        new_inputs = df2.drop(['Severity','BI-RADS assessment'], axis='columns')
        target = df2['Severity']
        return(new_inputs,target)


    def Cleansing_Mode(path):
        dataframe = pd.read_csv(path)
        i = dataframe[((dataframe.Age == '?') | (dataframe.Shape == '?') | (dataframe.Margin == '?') | (dataframe.Density == '?'))].index
        df = dataframe.drop(i)
        inputs = df.drop(['Severity','BI-RADS assessment'], axis='columns')
        mode_inputs= pd.Series([int(statistics.mode(inputs['Age'])), 
                                int(statistics.mode(inputs['Shape'])),
                                int(statistics.mode(inputs['Margin'])),
                                int(statistics.mode(inputs['Density']))], 
                                index=['Age','Shape','Margin','Density'])
        dataframe_new = pd.read_csv(path)
        df2 = dataframe_new.replace("?",mode_inputs)
        new_inputs = df2.drop(['Severity','BI-RADS assessment'], axis='columns')
        target = df2['Severity']
        return(new_inputs,target)

    train_inputs_med = Cleansing_Median(train_path)[0]
    train_target_med = Cleansing_Median(train_path)[1]
    test_inputs_med = Cleansing_Median(test_path)[0]
    test_target_med = Cleansing_Median(test_path)[1]
    val_inputs_med = Cleansing_Median(val_path)[0]
    val_target_med = Cleansing_Median(val_path)[1]

    train_inputs_mode = Cleansing_Mode(train_path)[0]
    train_target_mode = Cleansing_Mode(train_path)[1]
    test_inputs_mode = Cleansing_Mode(test_path)[0]
    test_target_mode = Cleansing_Mode(test_path)[1]
    val_inputs_mode = Cleansing_Mode(val_path)[0]
    val_target_mode = Cleansing_Mode(val_path)[1]

    # Q1(A) 

    def q1a(train_inputs,train_target,test_inputs,test_target,val_inputs,val_target):
        global out_str
        model = tree.DecisionTreeClassifier()
        model.fit(train_inputs,train_target)

        def Accuracy(path):
            global out_str
            if 'train' in path:    
                BestAcc = 100*(model.score(train_inputs,train_target))
                out_str += f'\nAccuracy on Training dataset: {BestAcc} %'
            elif 'test' in path:
                BestAcc = 100*(model.score(test_inputs,test_target))
                out_str += f'\nAccuracy on Testing dataset: {BestAcc} %'
            elif 'val' in path:
                BestAcc = 100*(model.score(val_inputs,val_target))
                out_str += f'\nAccuracy on Validation dataset: {BestAcc} %'

        Accuracy(train_path)
        Accuracy(test_path)
        Accuracy(val_path)

        #Visualising the Decision Tree

        plt.figure(figsize = (15,10))
        dec_tree = tree.plot_tree(model, filled =True, rounded= True)
        # plt.show()


    out_str += '\nDataset-1) (a) MEDIAN MODEL'
    q1a(train_inputs_med,train_target_med,
        test_inputs_med,test_target_med,
        val_inputs_med,val_target_med)


    out_str += '\nDataset-1) (a) MODE MODEL'
    q1a(train_inputs_mode,train_target_mode,
        test_inputs_mode,test_target_mode,
        val_inputs_mode,val_target_mode)
    
    out_str += '\n-------------------------------------------\n' + 'q1(e)(a) done' + '\n-------------------------------------------\n'

    # Q1-(B)

    def q1b(train_inputs,train_target,test_inputs,test_target,val_inputs,val_target):
        global out_str
        model = tree.DecisionTreeClassifier()
        model.fit(train_inputs,train_target)

        dtree = GridSearchCV(model,{'max_depth' : [5,8,10,12,15,18,20],'min_samples_split' : [2,4,6,8,10],'min_samples_leaf' : [1,2,3]},cv=5,return_train_score=False)
        dtree.fit(train_inputs,train_target)
        df = pd.DataFrame(dtree.cv_results_)
        bestmodel = dtree.best_estimator_
        bestmodel.fit(train_inputs,train_target)

        def Accuracy(path):
            global out_str
            if 'train' in path:    
                BestAcc = 100*(bestmodel.score(train_inputs,train_target))
                out_str += f'\nAccuracy on Training dataset: {BestAcc} %'
            elif 'test' in path:
                BestAcc = 100*(bestmodel.score(test_inputs,test_target))
                out_str += f'\nAccuracy on Testing dataset: {BestAcc} %'
            elif 'val' in path:
                BestAcc = 100*(bestmodel.score(val_inputs,val_target))
                out_str += f'\nAccuracy on Validation dataset: {BestAcc} %'


        Accuracy(train_path)
        Accuracy(test_path)
        Accuracy(val_path)

        #Visualising the Decision Tree

        plt.figure(figsize = (15,10))
        dec_tree = tree.plot_tree(bestmodel, filled =True, rounded= True)
        # plt.show()


    out_str += '\nDataset-1) (b) MEDIAN MODEL'
    q1b(train_inputs_med,train_target_med,
        test_inputs_med,test_target_med,
        val_inputs_med,val_target_med)


    out_str += '\nDataset-1) (b) MODE MODEL'
    q1b(train_inputs_mode,train_target_mode,
        test_inputs_mode,test_target_mode,
        val_inputs_mode,val_target_mode)

    out_str += '\n-------------------------------------------\n' + 'q1(e)(b) done' + '\n-------------------------------------------\n'

    # Q1-(C)

    def q1c(train_inputs,train_target,test_inputs,test_target,val_inputs,val_target):
        global out_str

        model = tree.DecisionTreeClassifier()
        model.fit(train_inputs,train_target)


        def Accuracy(model,path):
            global out_str
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
        # plt.show()

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
        # plt.show()

        plt.plot(ccp_alphas, max_depth, marker="o", drawstyle="steps-post")
        plt.xlabel("Alpha")
        plt.ylabel("Depth of Tree")
        plt.title("Depth vs Alpha")
        plt.tight_layout()
        # plt.show()

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
        # plt.show()

        bestvalacc = max(val_acc)
        index_bestvalacc = val_acc.index(bestvalacc)

        plt.figure(figsize = (15,10))
        dec_tree = tree.plot_tree(trees[index_bestvalacc], filled =True, rounded= True)
        # plt.show()

        out_str += f'\nTraining Accuracy:{train_acc[index_bestvalacc]}' + f'\nTesting Accuracy:{test_acc[index_bestvalacc]}' + f'\nValidation Accuracy:{val_acc[index_bestvalacc]}'



    out_str += '\nDataset-1) (c) MEDIAN MODEL'
    q1c(train_inputs_med,train_target_med,
        test_inputs_med,test_target_med,
        val_inputs_med,val_target_med)


    out_str += '\nDataset-1) (c) MODE MODEL'
    q1c(train_inputs_mode,train_target_mode,
        test_inputs_mode,test_target_mode,
        val_inputs_mode,val_target_mode)


    out_str += '\n-------------------------------------------\n' + 'q1(e)(c) done' + '\n-------------------------------------------\n'

    # Q1-(D)

    def q1d(train_inputs,train_target,test_inputs,test_target,val_inputs,val_target):
        global out_str
        model = ensemble.RandomForestClassifier(oob_score=True)
        model.fit(train_inputs,train_target)


        randomforest = GridSearchCV(model,{'n_estimators' : [20,40,60,80,100],'min_samples_split' : [2,4,6,8,10],'max_features' : ['sqrt',2,3,4]},return_train_score=False, refit=True,)
        op = randomforest.fit(train_inputs,train_target)
        df = pd.DataFrame(randomforest.cv_results_)
        bestestimator = randomforest.best_estimator_


        def Accuracy(path):
            global out_str
            if 'train' in path:    
                BestAcc = 100*(model.score(train_inputs,train_target))
                out_str += f'\nAccuracy on Training dataset: {BestAcc} %'
            elif 'test' in path:
                BestAcc = 100*(model.score(test_inputs,test_target))
                out_str += f'\nAccuracy on Testing dataset: {BestAcc} %'
            elif 'val' in path:
                BestAcc = 100*(model.score(val_inputs,val_target))
                out_str += f'\nAccuracy on Validation dataset: {BestAcc} %'
        
        oob = 100*(model.oob_score_)


        Accuracy(train_path)
        Accuracy(test_path)
        Accuracy(val_path)
        out_str += f'Out-Of-Bag Accuracy: {oob} %'



    out_str += '\nDataset-1) (d) MEDIAN MODEL'
    q1d(train_inputs_med,train_target_med,
        test_inputs_med,test_target_med,
        val_inputs_med,val_target_med)


    out_str += '\nDataset-1) (d) MODE MODEL'
    q1d(train_inputs_mode,train_target_mode,
        test_inputs_mode,test_target_mode,
        val_inputs_mode,val_target_mode)


    out_str += '\n-------------------------------------------\n' + 'q1(e)(d) done' + '\n-------------------------------------------\n'

    with open('2_e.txt', 'w') as f:
        f.write(out_str)



def dataset_1_f():
    global out_str
    out_str = ""
    out_str += 'DATASET-1) (f)'

    def Cleansing(path):
        df = pd.read_csv(path)
        # i = dataframe[((dataframe.Age == '?') | (dataframe.Shape == '?') | (dataframe.Margin == '?') | (dataframe.Density == '?'))].index
        # df = dataframe.drop(i)
        df = df.replace('?',np.nan)
        inputs = df.drop(['Severity','BI-RADS assessment'], axis='columns')
        target = df['Severity']
        inputs['Age'] = inputs['Age'].astype(float)
        inputs['Shape'] = inputs['Shape'].astype(float)
        inputs['Margin'] = inputs['Margin'].astype(float)
        inputs['Density'] = inputs['Density'].astype(float)
        # target['Severity'] = target['Severity'].astype(float)
        # print(inputs.dtypes)
        return(inputs,target)

    train_inputs = Cleansing(train_path)[0]
    train_target = Cleansing(train_path)[1]
    test_inputs = Cleansing(test_path)[0]
    test_target = Cleansing(test_path)[1]
    val_inputs = Cleansing(val_path)[0]
    val_target = Cleansing(val_path)[1]

    model = xgboost.XGBClassifier()
    model.fit(train_inputs,train_target)

    xgboosted_dtree = GridSearchCV(model,{'n_estimators' : [10,20,30,40,50],'subsample' : [0.1,0.2,0.3,0.4,0.5,0.6],'max_depth' : [4,5,6,7,8,9,10]},cv=5,return_train_score=False,refit=True)
    xgboosted_dtree.fit(train_inputs,train_target)
    df = pd.DataFrame(xgboosted_dtree.cv_results_)

    bestmodel=xgboosted_dtree.best_estimator_

    #Finding Accuracy

    def Accuracy(path):
        global out_str

        if 'train' in path:    
            BestAcc = 100*(bestmodel.score(train_inputs,train_target))
            out_str += f'\nAccuracy on Training dataset: {BestAcc} %'
        elif 'test' in path:
            BestAcc = 100*(bestmodel.score(test_inputs,test_target))
            out_str += f'\nAccuracy on Testing dataset: {BestAcc} %'
        elif 'val' in path:
            BestAcc = 100*(bestmodel.score(val_inputs,val_target))
            out_str += f'\nAccuracy on Validation dataset: {BestAcc} %'

    Accuracy(train_path)
    Accuracy(test_path)
    Accuracy(val_path)


    with open('2_f.txt', 'w') as f:
        f.write(out_str)





da1 = dataset_1_a()
da2 = dataset_1_b()
da3 = dataset_1_c()
da4 = dataset_1_d()
da5 = dataset_1_e()
da6 = dataset_1_f()