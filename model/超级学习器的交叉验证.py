# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:50:39 2019

@author: Jolin
"""

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score,roc_curve
from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

def superlearner(x_train,y_train,x_test,y_test):
    ensemble = SuperLearner(scorer=roc_auc_score,random_state=42,folds=10,backend="multiprocessing")
    ensemble.add([GaussianNB(),SVC(C=100, probability=True), neighbors.KNeighborsClassifier(n_neighbors=3), LogisticRegression(), MLPClassifier(), GradientBoostingClassifier(n_estimators=100), RandomForestClassifier(random_state=42,n_estimators=100), BaggingClassifier(), tree.DecisionTreeClassifier()],proba=True)
    ensemble.add_meta(LogisticRegression(),proba=True)
    
    ensemble.fit(x_train,y_train)
    preds_prob=ensemble.predict_proba(x_test)
    print('now is here -6\n')
    prob=preds_prob[:, 1]
    preds=[]
    for i in prob:
        if i>=0.5:
            preds.append(1);
        else:
            preds.append(0)
                
    auc_sl=roc_auc_score(y_test,preds_prob[:,1])
    auprc_sl=average_precision_score(y_test,preds_prob[:,1])
    recall_sl=recall_score(y_test,preds)
    acc_sl=accuracy_score(y_test,preds)
    p_sl=precision_score(y_test,preds)
    f1_sl=f1_score(y_test,preds)
#    fpr_sl,tpr_sl,thr_sl=roc_curve(y_test,prob)
    return [auc_sl,auprc_sl,acc_sl,recall_sl,p_sl,f1_sl]

dir_path=r'E:\lml_dataget\v3\1-roll-random_state=0\\'
comb_list=[[[1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
[1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1],
[0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
[0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1],
[1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0]
],[[1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
[0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]
],[[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1],
[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
[1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0]
],[[1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
[1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
[1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]
],[[0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
[1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
[1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
]]
res=[]
f=open(r'E:\lml_dataget\v3\superlearner.txt','w',encoding='utf-8-sig')
for k in tqdm(range(1,6)):
    now_comb=comb_list[0][k-1]
    x_train=pd.read_csv(dir_path+r'x_train_train_'+str(k)+'.csv',header=0,encoding='utf-8-sig')
    y_train=pd.read_csv(dir_path+r'y_train_train_'+str(k)+'.csv',header=0,encoding='utf-8-sig')
    x_test=pd.read_csv(dir_path+r'x_test_'+str(k)+'.csv',header=0,encoding='utf-8-sig')
    y_test=pd.read_csv(dir_path+r'y_test_'+str(k)+'.csv',encoding='utf-8-sig',header=None)
    res_k=superlearner(x_train,y_train,x_test,y_test)
    res.append([now_comb,res_k])
    f.write(str([now_comb,res_k])+'\n')
f.close()
