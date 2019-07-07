# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:52:15 2019

@author: Jolin
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
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

stats_list=['min','max','minmax','mean','std','stdmean','median','qua25','qua75','qua2575','mode','skew','kurt','first']
data=pd.read_csv(r'E:\lml_dataget\v3\all_minmax.csv',header=0,encoding='utf-8-sig')
data_x=data.iloc[:,1:-1]
data_y=data.iloc[:,-1]

sfolder = StratifiedKFold(n_splits=5,random_state=0,shuffle=False)

def simple_statistic(comb):
    resres=[]
    for train, test in tqdm(list(sfolder.split(data_x,data_y))):
#        break
        cofff=['age_interval','admission_type_EMERGENCY','admission_type_ELECTIVE','admission_type_URGENT','aids','hem','mets']
#        stats_list=['min','max','minmax','mean','std','stdmean','median','qua25','qua75','qua2575','mode','skew','kurt','first']
        X_train, X_test = data_x.iloc[train,:], data_x.iloc[test,:]
        Y_train, Y_test = data_y[train], data_y[test]
        x_train,x_val,y_train,y_val=train_test_split(X_train,Y_train,test_size=0.25,random_state=42)
        
        smo=SMOTE(random_state=42,ratio={1:2000})
        x_train_s,y_train_s=smo.fit_sample(x_train,y_train)
        
        ###对遗传算法中的训练集进行重采样，获得新的遗传算法训练集x_train_s
        x_train_s=pd.DataFrame(x_train_s,columns=x_val.columns)
        X_train_s=pd.concat([x_train_s,x_val],axis=0)
        Y_train_s=list(y_train_s)
        Y_train_s.extend(list(y_val))
        Y_train_s=np.array(Y_train_s)
        
        best_combination_nowfold=comb
        for sts in best_combination_nowfold:
            for column in x_train.columns:
                if(sts == column.split('_')[0]):
                    cofff.append(column)
        
        x_train_train=X_train_s[cofff]
        y_train_train=Y_train_s
        x_test=X_test[cofff]
        y_test=Y_test
    
        ensemble = SuperLearner(scorer=roc_auc_score,random_state=42,folds=10,backend="multiprocessing")
        ensemble.add([GaussianNB(),SVC(C=100, probability=True), neighbors.KNeighborsClassifier(n_neighbors=3), LogisticRegression(), MLPClassifier(), GradientBoostingClassifier(n_estimators=100), RandomForestClassifier(random_state=42,n_estimators=100), BaggingClassifier(), tree.DecisionTreeClassifier()],proba=True)
        ensemble.add_meta(LogisticRegression(),proba=True)
        print('now is here -4\n')
        ensemble.fit(x_train_train,y_train_train)
        print('now is here -5\n')
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
        fpr_sl,tpr_sl,thr_sl=roc_curve(y_test,prob)
        print('now is here -7')
        resres.append([best_combination_nowfold,auc_sl,auprc_sl,acc_sl,p_sl,recall_sl,f1_sl,fpr_sl,tpr_sl,thr_sl])
    return resres

simple_statistic_list=[['min', 'std', 'qua25', 'qua75', 'skew', 'first'],['min', 'minmax', 'mean', 'stdmean', 'qua25', 'qua75', 'qua2575', 'mode', 'first']]
abcabc=[]
#f=open(r'E:\lml_dataget\v3\simple_statistic_res.txt','w',encoding='utf-8-sig')
for iii in tqdm(simple_statistic_list):
    aab=simple_statistic(iii)
    abcabc.append([iii,aab])
#    f.write(str([iii,aab])+'\n')
#f.close()
#pd.DataFrame(abcabc).to_csv(r'E:\lml_dataget\v3\simple_statistic.csv',encoding='utf-8-sig')