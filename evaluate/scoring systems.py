# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:30:30 2019

@author: Jolin
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,recall_score,roc_auc_score,roc_curve, accuracy_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
    
def sofa(X_train_s,Y_train_s,X_test,Y_test):
    LR=LogisticRegression()
    LR.fit(X_train_s,Y_train_s)
    preds_sofa=LR.predict(X_test)
    preds_prob_sofa=LR.predict_proba(X_test)
    auc_sofa=roc_auc_score(Y_test,preds_prob_sofa[:,1])
    auprc_sofa=average_precision_score(Y_test,preds_prob_sofa[:,1])
    recall_sofa=recall_score(Y_test,preds_sofa)
    acc=accuracy_score(Y_test,preds_sofa)
    fpr_sofa,tpr_sofa,thr_sofa=roc_curve(Y_test,preds_prob_sofa[:,1])
    return [auc_sofa,auprc_sofa,acc,recall_sofa,fpr_sofa,tpr_sofa,thr_sofa]

def newsapsii(X_train_s,Y_train_s,X_test,Y_test):
    LR=LogisticRegression()
    LR.fit(X_train_s,Y_train_s)
    preds_newsapsii=LR.predict(X_test)
    preds_prob_newsapsii=LR.predict_proba(X_test)
    auc_newsapsii=roc_auc_score(Y_test,preds_prob_newsapsii[:,1])
    auprc_newsapsii=average_precision_score(Y_test,preds_prob_newsapsii[:,1])
    recall_newsapsii=recall_score(Y_test,preds_newsapsii)
    acc=accuracy_score(Y_test,preds_newsapsii)
    fpr_newsapsii,tpr_newsapsii,thr_newsapsii=roc_curve(Y_test,preds_prob_newsapsii[:,1])
    return [auc_newsapsii,auprc_newsapsii,acc,recall_newsapsii,fpr_newsapsii,tpr_newsapsii,thr_newsapsii]

def mods(X_train_s,Y_train_s,X_test,Y_test):
    LR=LogisticRegression()
    LR.fit(X_train_s,Y_train_s)
    preds_mods=LR.predict(X_test)
    preds_prob_mods=LR.predict_proba(X_test)
    auc_mods=roc_auc_score(Y_test,preds_prob_mods[:,1])
    auprc_mods=average_precision_score(Y_test,preds_prob_mods[:,1])
    acc=accuracy_score(Y_test,preds_mods)
    recall_mods=recall_score(Y_test,preds_mods)
    fpr_mods,tpr_mods,thr_mods=roc_curve(Y_test,preds_prob_mods[:,1])
    return [auc_mods,auprc_mods,acc,recall_mods,fpr_mods,tpr_mods,thr_mods]

def sapsii(data_x,data_y):
    data_use=data_x[['sapsii']]
    label=data_y
    data_use['trans']= data_use.apply(lambda x: -7.7631+0.0737*x+0.9971*np.log2(1+x))
    data_use['preds_prob']=2**data_use['trans']/(1+2**data_use['trans'])
    preds_prob_sapsii=data_use['preds_prob']
    preds_sapsii=preds_prob_sapsii.apply(lambda x: 1 if x >=0.5 else 0)
    auc_sapsii=roc_auc_score(label,preds_prob_sapsii)
    auprc_sapsii=average_precision_score(label,preds_prob_sapsii)
    acc=accuracy_score(label,preds_sapsii)
    recall_sapsii=recall_score(label,preds_sapsii)
    fpr_sapsii,tpr_sapsii,thr_sapsii=roc_curve(label,preds_prob_sapsii)
    return [auc_sapsii,auprc_sapsii,acc,recall_sapsii,fpr_sapsii,tpr_sapsii,thr_sapsii]


data=pd.read_csv(r'E:\lml_dataget\v3\fortableone\score.csv',header=0,encoding='utf-8-sig')
data_x=data.iloc[:,3:]
minmax=MinMaxScaler()
data_x=pd.DataFrame(minmax.fit_transform(data_x),columns=data.columns[3:])
data_y=data['hospital_expire_flag']

sfolder = StratifiedKFold(n_splits=5,random_state=0,shuffle=False)

aaasofaa=[]
aaanewsapsiii=[]
aaamodss=[]
for train, test in tqdm(list(sfolder.split(data_x,data_y))):
#    break
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
    
    ###sofa
    x_train_train=X_train_s.iloc[:,1:7]
    y_train_train=Y_train_s
    x_test_test=X_test.iloc[:,1:7]
    y_test_test=Y_test
    res_sofa=sofa(x_train_train,y_train_train,x_test_test,y_test_test)
    aaasofaa.append(res_sofa)
    
    ###newsapsii
    x_train_train=X_train_s.iloc[:,9:24]
    y_train_train=Y_train_s
    x_test_test=X_test.iloc[:,9:24]
    y_test_test=Y_test
    res_newsapsii=newsapsii(x_train_train,y_train_train,x_test_test,y_test_test)
    aaanewsapsiii.append(res_newsapsii)
    
    ###mlods
    x_train_train=X_train_s.iloc[:,25:-1]
    y_train_train=Y_train_s
    x_test_test=X_test.iloc[:,25:-1]
    y_test_test=Y_test
    res_mods=mods(x_train_train,y_train_train,x_test_test,y_test_test)
    aaamodss.append(res_mods)
    
    ###sapsii
aaasapsiii=sapsii(data_x,data_y)
a_mods=pd.DataFrame(aaamodss,columns=['auroc','auprc','acc','recall','fpr','tpr','thr'])
a_sofa=pd.DataFrame(aaasofaa,columns=['auroc','auprc','acc','recall','fpr','tpr','thr'])
#a_sapsii=pd.DataFrame(aaasapsiii,columns=['auroc','auprc','recall','fpr','tpr','thr'])
a_newsapsii=pd.DataFrame(aaanewsapsiii,columns=['auroc','auprc','acc','recall','fpr','tpr','thr'])
a_mods_mean=a_mods.mean()
a_newsapsii_mean=a_newsapsii.mean()
a_sofa_mean=a_sofa.mean()

a_mods_fpr=(pd.DataFrame([list(i) for i in a_mods['fpr']])).mean()
a_mods_tpr=(pd.DataFrame([list(i) for i in a_mods['tpr']])).mean()
a_mods_thr=(pd.DataFrame([list(i) for i in a_mods['thr']])).mean()

a_sofa_fpr=(pd.DataFrame([list(i) for i in a_sofa['fpr']])).mean()
a_sofa_tpr=(pd.DataFrame([list(i) for i in a_sofa['tpr']])).mean()
a_sofa_thr=(pd.DataFrame([list(i) for i in a_sofa['thr']])).mean()

a_sapsii_fpr=aaasapsiii[3]
a_sapsii_tpr=aaasapsiii[4]
a_sapsii_thr=aaasapsiii[5]

a_newsapsii_fpr=(pd.DataFrame([list(i) for i in a_newsapsii['fpr']])).mean()
a_newsapsii_tpr=(pd.DataFrame([list(i) for i in a_newsapsii['tpr']])).mean()
a_newsapsii_thr=(pd.DataFrame([list(i) for i in a_newsapsii['thr']])).mean()
    

    
    
    
    
    
    

