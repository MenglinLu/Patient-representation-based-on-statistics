# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 14:55:53 2018

@author: Jolin
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score,recall_score,accuracy_score,precision_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LassoLarsIC
from sklearn.metrics import roc_auc_score
from pyearth import Earth
from sklearn.model_selection import cross_validate
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

indiv=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
data_all=pd.read_csv('E:\\sepsis\\data\\final3_b.csv',header=0)
data_all=data_all.fillna(data_all.mean())
dataset=data_all.iloc[:,1:-7]
stats_list=['min','max','minmax','mean','std','stdmean','median','qua25','qua75','qua2575','mode','skew','kurt','first']
cof=['age','gender_F','gender_M','admission_type_EMERGENCY','admission_type_ELECTIVE','admission_type_URGENT','AIDS','HEM','METS']
id1 = [i for i,x in enumerate(indiv) if x==1]
stats=[stats_list[i] for i in id1]
for sts in stats:
    for column in dataset.columns:
        if(sts == column.split('_')[0]):
            cof.append(column);
data_x=dataset[cof]
data_y=data_all['hosp_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
ensemble = SuperLearner(scorer=roc_auc_score,random_state=42,folds=10,backend="multiprocessing")
ensemble.add([GaussianNB(),SVC(C=100, probability=True), neighbors.KNeighborsClassifier(n_neighbors=3), LogisticRegression(), MLPClassifier(), GradientBoostingClassifier(n_estimators=100), RandomForestClassifier(random_state=42,n_estimators=100), BaggingClassifier(), tree.DecisionTreeClassifier()],proba=True)
ensemble.add_meta(LogisticRegression(),proba=True)
ensemble.fit(x_train, y_train)
preds_prob=ensemble.predict_proba(x_test)
prob=preds_prob[:, 1]
preds=[]
for i in prob:
    if i>=0.5:
        preds.append(1);
    else:
        preds.append(0)
        
auc_sl=roc_auc_score(y_test,preds_prob[:,1])
auprc_sl=average_precision_score(y_test,preds_prob[:,1])
recall_sl=recall_score(preds,y_test)

#超级学习器灵敏度和特异度
fpr_sl,tpr_sl,thr_sl=roc_curve(y_test,prob)

#SOFA评分
from sklearn.preprocessing import MinMaxScaler
#data_all=pd.read_csv('C:\\Users\\Angela Du\\Desktop\\sepsis\\data\\mods.csv',header=0)
#data_x=data_all.iloc[:,3:-1]
data_all=pd.read_csv('E:\\sepsis\\data\\baseline.csv',header=0)
data_x=data_all.iloc[:,2:8]
min_max=MinMaxScaler()
data_x=min_max.fit_transform(data_x)
data_y=data_all['expire_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
LR=LogisticRegression()
LR.fit(x_train,y_train)
preds_sofa=LR.predict(x_test)
preds_prob_sofa=LR.predict_proba(x_test)
auc_sofa=roc_auc_score(y_test,preds_prob_sofa[:,1])
auprc_sofa=average_precision_score(y_test,preds_prob_sofa[:,1])
recall_sofa=recall_score(preds_sofa,y_test)
fpr_sofa,tpr_sofa,thr_sofa=roc_curve(y_test,preds_prob_sofa[:,1])

#New SAPSII
from sklearn.preprocessing import MinMaxScaler
#data_all=pd.read_csv('C:\\Users\\Angela Du\\Desktop\\sepsis\\data\\mods.csv',header=0)
#data_x=data_all.iloc[:,3:-1]
data_all=pd.read_csv('E:\\sepsis\\data\\baseline.csv',header=0)
data_x=data_all.iloc[:,9:-1]
min_max=MinMaxScaler()
data_x=min_max.fit_transform(data_x)
data_y=data_all['expire_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
LR=LogisticRegression()
LR.fit(x_train,y_train)
preds_newsapsii=LR.predict(x_test)
preds_prob_newsapsii=LR.predict_proba(x_test)
auc_newsapsii=roc_auc_score(y_test,preds_prob_newsapsii[:,1])
auprc_newsapsii=average_precision_score(y_test,preds_prob_newsapsii[:,1])
recall_newsapsii=recall_score(preds_newsapsii,y_test)
fpr_newsapsii,tpr_newsapsii,thr_newsapsii=roc_curve(y_test,preds_prob_newsapsii[:,1])

#MODS
from sklearn.preprocessing import MinMaxScaler
data_all=pd.read_csv(r'E:\sepsis\data\mods.csv',header=0)
data_x=data_all.iloc[:,3:-1]
min_max=MinMaxScaler()
data_x=min_max.fit_transform(data_x)
data_y=data_all['expire_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
LR=LogisticRegression()
LR.fit(x_train,y_train)
preds_mods=LR.predict(x_test)
preds_prob_mods=LR.predict_proba(x_test)
auc_mods=roc_auc_score(y_test,preds_prob_mods[:,1])
auprc_mods=average_precision_score(y_test,preds_prob_mods[:,1])
recall_mods=recall_score(preds_mods,y_test)
fpr_mods,tpr_mods,thr_mods=roc_curve(y_test,preds_prob_mods[:,1])

#SAPSII
data_all=pd.read_csv('E:\\sepsis\\data\\baseline.csv',header=0)
data_use=data_all[['sapsii']]
label=data_all['expire_flag']
data_use['trans']= data_use.apply(lambda x: -7.7631+0.0737*x+0.9971*np.log2(1+x))
data_use['preds_prob']=2**data_use['trans']/(1+2**data_use['trans'])
preds_prob_sapsii=data_use['preds_prob']
#preds_prob=data_use.apply(lambda x: -7.7631+0.0737*x+0.9971*np.log10(1+x))
preds_sapsii=preds_prob_sapsii.apply(lambda x: 1 if x >=0.5 else 0)
auc_sapsii=roc_auc_score(label,preds_prob_sapsii)
auprc_sapsii=average_precision_score(label,preds_prob_sapsii)
recall_sapsii=recall_score(label,preds_sapsii)
fpr_sapsii,tpr_sapsii,thr_sapsii=roc_curve(label,preds_prob_sapsii)

#绘制ROC曲线
fig=plt.figure(figsize=(7,7))
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.plot(fpr_sl_all,tpr_sl_all,color='mediumpurple', linestyle='-', lw=2,label='0.9444 Top 1 by Super Learner')
plt.plot(fpr_newsapsii,tpr_newsapsii,color='c', linestyle='--', lw=2, label='0.7236 New SAPSII')
plt.plot(fpr_sapsii,tpr_sapsii,color='springgreen', linestyle='-.', lw=2, label='0.6845 SAPSII')
plt.plot(fpr_sofa,tpr_sofa,color='coral', linestyle='--', lw=2, label='0.6513 SOFA')
plt.plot(fpr_mods,tpr_mods,color='m', linestyle=':', lw=2, label='0.6351 MODS')
plt.xlim([-0.02,1.0])
plt.ylim([0,1.05])
plt.xlabel('False Positive Rate',fontsize=12)
plt.ylabel('True Positive Rate',fontsize=12)
plt.legend(loc="lower right",fontsize=8)
plt.show()
fig.savefig(r'C:\Users\Angela Du\Desktop\new eps\fig7.eps')


##常用统计特征组合

data_all=pd.read_csv('E:\\sepsis\\data\\final3_b.csv',header=0)
data_all=data_all.fillna(data_all.mean())
dataset=data_all.iloc[:,1:-7]
stats_list=['min','max','minmax','mean','std','stdmean','median','qua25','qua75','qua2575','mode','skew','kurt','first']
cof=['age','gender_F','gender_M','admission_type_EMERGENCY','admission_type_ELECTIVE','admission_type_URGENT','AIDS','HEM','METS']
#最小值
indiv=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
id1 = [i for i,x in enumerate(indiv) if x==1]
stats=[stats_list[i] for i in id1]
for sts in stats:
    for column in dataset.columns:
        if(sts == column.split('_')[0]):
            cof.append(column);
data_x=dataset[cof]
data_y=data_all['hosp_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
ensemble = SuperLearner(scorer=roc_auc_score,random_state=42,folds=10,backend="multiprocessing")
ensemble.add([GaussianNB(),SVC(C=100, probability=True), neighbors.KNeighborsClassifier(n_neighbors=3), LogisticRegression(), MLPClassifier(), GradientBoostingClassifier(n_estimators=100), RandomForestClassifier(random_state=42,n_estimators=50), BaggingClassifier(), tree.DecisionTreeClassifier()],proba=True)
ensemble.add_meta(LogisticRegression(),proba=True)
ensemble.fit(x_train, y_train)
preds_prob_min=ensemble.predict_proba(x_test)
prob_min=preds_prob_min[:, 1]
preds_min=[]
for i in prob:
    if i>=0.5:
        preds_min.append(1);
    else:
        preds_min.append(0)
        
auc_sl_min=roc_auc_score(y_test,prob_min)
auprc_sl_min=average_precision_score(y_test,prob_min)
recall_sl_min=recall_score(preds_min,y_test)
fpr_sl_min,tpr_sl_min,thr_sl_min=roc_curve(y_test,prob_min)

#最大值
indiv=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
id1 = [i for i,x in enumerate(indiv) if x==1]
stats=[stats_list[i] for i in id1]
for sts in stats:
    for column in dataset.columns:
        if(sts == column.split('_')[0]):
            cof.append(column);
data_x=dataset[cof]
data_y=data_all['hosp_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
ensemble = SuperLearner(scorer=roc_auc_score,random_state=42,folds=10,backend="multiprocessing")
ensemble.add([GaussianNB(),SVC(C=100, probability=True), neighbors.KNeighborsClassifier(n_neighbors=3), LogisticRegression(), MLPClassifier(), GradientBoostingClassifier(n_estimators=100), RandomForestClassifier(random_state=42,n_estimators=50), BaggingClassifier(), tree.DecisionTreeClassifier()],proba=True)
ensemble.add_meta(LogisticRegression(),proba=True)
ensemble.fit(x_train, y_train)
preds_prob_max=ensemble.predict_proba(x_test)
prob_max=preds_prob_max[:, 1]
preds_max=[]
for i in prob:
    if i>=0.5:
        preds_max.append(1);
    else:
        preds_max.append(0)
        
auc_sl_max=roc_auc_score(y_test,preds_prob_max[:,1])
auprc_sl_max=average_precision_score(y_test,preds_prob_max[:,1])
recall_sl_max=recall_score(preds_max,y_test)
fpr_sl_max,tpr_sl_max,thr_sl_max=roc_curve(y_test,prob_max)

#均值
indiv=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
id1 = [i for i,x in enumerate(indiv) if x==1]
stats=[stats_list[i] for i in id1]
for sts in stats:
    for column in dataset.columns:
        if(sts == column.split('_')[0]):
            cof.append(column);
data_x=dataset[cof]
data_y=data_all['hosp_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3,random_state=42)
ensemble = SuperLearner(scorer=roc_auc_score,random_state=42,folds=10,backend="multiprocessing")
ensemble.add([GaussianNB(),SVC(C=100, probability=True), neighbors.KNeighborsClassifier(n_neighbors=3), LogisticRegression(), MLPClassifier(), GradientBoostingClassifier(n_estimators=100), RandomForestClassifier(random_state=42,n_estimators=50), BaggingClassifier(), tree.DecisionTreeClassifier()],proba=True)
ensemble.add_meta(LogisticRegression(),proba=True)
ensemble.fit(x_train, y_train)
preds_prob_mean=ensemble.predict_proba(x_test)
prob_mean=preds_prob_mean[:, 1]
preds_mean=[]
for i in prob:
    if i>=0.5:
        preds_mean.append(1);
    else:
        preds_mean.append(0)
        
auc_sl_mean=roc_auc_score(y_test,preds_prob_mean[:,1])
auprc_sl_mean=average_precision_score(y_test,preds_prob_mean[:,1])
recall_sl_mean=recall_score(preds_mean,y_test)
fpr_sl_mean,tpr_sl_mean,thr_sl_mean=roc_curve(y_test,prob_mean)

#第一次测量值
indiv=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
id1 = [i for i,x in enumerate(indiv) if x==1]
stats=[stats_list[i] for i in id1]
for sts in stats:
    for column in dataset.columns:
        if(sts == column.split('_')[0]):
            cof.append(column);
data_x=dataset[cof]
data_y=data_all['hosp_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
ensemble = SuperLearner(scorer=roc_auc_score,random_state=42,folds=10,backend="multiprocessing")
ensemble.add([GaussianNB(),SVC(C=100, probability=True), neighbors.KNeighborsClassifier(n_neighbors=3), LogisticRegression(), MLPClassifier(), GradientBoostingClassifier(n_estimators=50), RandomForestClassifier(random_state=42,n_estimators=50), BaggingClassifier(), tree.DecisionTreeClassifier()],proba=True)
ensemble.add_meta(LogisticRegression(),proba=True)
ensemble.fit(x_train, y_train)
preds_prob_first=ensemble.predict_proba(x_test)
prob_first=preds_prob_first[:, 1]
preds_first=[]
for i in prob:
    if i>=0.5:
        preds_first.append(1);
    else:
        preds_first.append(0)
        
auc_sl_first=roc_auc_score(y_test,preds_prob_first[:,1])
auprc_sl_first=average_precision_score(y_test,preds_prob_first[:,1])
recall_sl_first=recall_score(preds_first,y_test)
fpr_sl_first,tpr_sl_first,thr_sl_first=roc_curve(y_test,prob_first)

#最小值，最大值
indiv=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
id1 = [i for i,x in enumerate(indiv) if x==1]
stats=[stats_list[i] for i in id1]
for sts in stats:
    for column in dataset.columns:
        if(sts == column.split('_')[0]):
            cof.append(column);
data_x=dataset[cof]
data_y=data_all['hosp_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
ensemble = SuperLearner(scorer=roc_auc_score,random_state=42,folds=10,backend="multiprocessing")
ensemble.add([GaussianNB(),SVC(C=100, probability=True), neighbors.KNeighborsClassifier(n_neighbors=3), LogisticRegression(), MLPClassifier(), GradientBoostingClassifier(n_estimators=50), RandomForestClassifier(random_state=42,n_estimators=50), BaggingClassifier(), tree.DecisionTreeClassifier()],proba=True)
ensemble.add_meta(LogisticRegression(),proba=True)
ensemble.fit(x_train, y_train)
preds_prob_min_max=ensemble.predict_proba(x_test)
prob_min_max=preds_prob_min_max[:, 1]
preds_min_max=[]
for i in prob:
    if i>=0.5:
        preds_min_max.append(1);
    else:
        preds_min_max.append(0)
        
auc_sl_min_max=roc_auc_score(y_test,preds_prob_min_max[:,1])
auprc_sl_min_max=average_precision_score(y_test,preds_prob_min_max[:,1])
recall_sl_min_max=recall_score(preds_min_max,y_test)
fpr_sl_min_max,tpr_sl_min_max,thr_sl_min_max=roc_curve(y_test,prob_min_max)

#最大值，最小值，均值
indiv=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
id1 = [i for i,x in enumerate(indiv) if x==1]
stats=[stats_list[i] for i in id1]
for sts in stats:
    for column in dataset.columns:
        if(sts == column.split('_')[0]):
            cof.append(column);
data_x=dataset[cof]
data_y=data_all['hosp_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
ensemble = SuperLearner(scorer=roc_auc_score,random_state=42,folds=10,backend="multiprocessing")
ensemble.add([GaussianNB(),SVC(C=100, probability=True), neighbors.KNeighborsClassifier(n_neighbors=3), LogisticRegression(), MLPClassifier(), GradientBoostingClassifier(n_estimators=100), RandomForestClassifier(random_state=42,n_estimators=250), BaggingClassifier(), tree.DecisionTreeClassifier()],proba=True)
ensemble.add_meta(LogisticRegression(),proba=True)
ensemble.fit(x_train, y_train)
preds_prob_min_max_mean=ensemble.predict_proba(x_test)
prob_min_max_mean=preds_prob_min_max_mean[:, 1]
preds_min_max_mean=[]
for i in prob:
    if i>=0.5:
        preds_min_max_mean.append(1);
    else:
        preds_min_max_mean.append(0)
        
auc_sl_min_max_mean=roc_auc_score(y_test,preds_prob_min_max_mean[:,1])
auprc_sl_min_max_mean=average_precision_score(y_test,preds_prob_min_max_mean[:,1])
recall_sl_min_max_mean=recall_score(preds_min_max_mean,y_test)
fpr_sl_min_max_mean,tpr_sl_min_max_mean,thr_sl_min_max_mean=roc_curve(y_test,prob_min_max_mean)

#最大值、最小值、均值、方差
indiv=[1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
id1 = [i for i,x in enumerate(indiv) if x==1]
stats=[stats_list[i] for i in id1]
for sts in stats:
    for column in dataset.columns:
        if(sts == column.split('_')[0]):
            cof.append(column);
data_x=dataset[cof]
data_y=data_all['hosp_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
ensemble = SuperLearner(scorer=roc_auc_score,random_state=42,folds=10,backend="multiprocessing")
ensemble.add([GaussianNB(),SVC(C=100, probability=True), neighbors.KNeighborsClassifier(n_neighbors=3), LogisticRegression(), MLPClassifier(), GradientBoostingClassifier(n_estimators=100), RandomForestClassifier(random_state=42,n_estimators=250), BaggingClassifier(), tree.DecisionTreeClassifier()],proba=True)
ensemble.add_meta(LogisticRegression(),proba=True)
ensemble.fit(x_train, y_train)
preds_prob_min_max_mean_std=ensemble.predict_proba(x_test)
prob_min_max_mean_std=preds_prob_min_max_mean_std[:, 1]
preds_min_max_mean_std=[]
for i in prob:
    if i>=0.5:
        preds_min_max_mean_std.append(1);
    else:
        preds_min_max_mean_std.append(0)
        
auc_sl_min_max_mean_std=roc_auc_score(y_test,preds_prob_min_max_mean_std[:,1])
auprc_sl_min_max_mean_std=average_precision_score(y_test,preds_prob_min_max_mean_std[:,1])
recall_sl_min_max_mean_std=recall_score(preds_min_max_mean_std,y_test)
fpr_sl_min_max_mean_std,tpr_sl_min_max_mean_std,thr_sl_min_max_mean_std=roc_curve(y_test,prob_min_max_mean_std)

##全部
indiv=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
id1 = [i for i,x in enumerate(indiv) if x==1]
stats=[stats_list[i] for i in id1]
for sts in stats:
    for column in dataset.columns:
        if(sts == column.split('_')[0]):
            cof.append(column);
data_x=dataset[cof]
data_y=data_all['hosp_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
ensemble = SuperLearner(scorer=roc_auc_score,folds=10,backend="multiprocessing")
ensemble.add([GaussianNB(),SVC(C=100, probability=True), neighbors.KNeighborsClassifier(n_neighbors=3), LogisticRegression(), MLPClassifier(), GradientBoostingClassifier(n_estimators=100), RandomForestClassifier(random_state=42,n_estimators=250), BaggingClassifier(), tree.DecisionTreeClassifier()],proba=True)
ensemble.add_meta(LogisticRegression(),proba=True)
ensemble.fit(x_train, y_train)
preds_prob_all=ensemble.predict_proba(x_test)
prob_all=preds_prob_all[:, 1]
preds_all=[]
for i in prob:
    if i>=0.5:
        preds_all.append(1);
    else:
        preds_all.append(0)
        
auc_sl_all=roc_auc_score(y_test,preds_prob_all[:,1])
auprc_sl_all=average_precision_score(y_test,preds_prob_all[:,1])
recall_sl_all=recall_score(preds_all,y_test)
fpr_sl_all,tpr_sl_all,thr_sl_all=roc_curve(y_test,prob_all)

fig=plt.figure(figsize=(7,7))
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.plot(fpr_sl_all,tpr_sl_all,color='mediumpurple', linestyle='-', lw=2,label='0.9444 min,max,std,stdmean,'+'\n'+'median,skew,first(Top 1)')
plt.plot(fpr_sl_min_max_mean_std,tpr_sl_min_max_mean_std,color='sandybrown', linestyle='--', lw=2, label='0.9358 min,max,mean,std(Top 6)')
plt.plot(fpr_sl,tpr_sl,color='blue', linestyle='--', lw=2, label='0.9157 all')
plt.plot(fpr_sl_min_max_mean,tpr_sl_min_max_mean,color='coral', linestyle='--', lw=2, label='0.9053 min,max')
plt.plot(fpr_sl_mean,tpr_sl_mean,color='c', linestyle='--', lw=2, label='0.9028 max')
plt.plot(fpr_sl_min,tpr_sl_min,color='lightblue', linestyle='--', lw=2, label='0.9007 min,max,mean')
plt.plot(fpr_sl_first,tpr_sl_first,color='green', linestyle='-.', lw=2, label='0.8987 mean')
plt.plot(fpr_sl_max,tpr_sl_max,color='m', linestyle=':', lw=2, label='0.8967 min')
plt.plot(fpr_sl_min_max,tpr_sl_min_max,color='red', linestyle='--', lw=2, label='0.8906 first')

plt.xlim([-0.02,1.0])
plt.ylim([0,1.05])
plt.xlabel('False Positive Rate',fontsize=12)
plt.ylabel('True Positive Rate',fontsize=12)
plt.legend(loc="lower right",fontsize=8)
plt.show()
fig.savefig(r'C:\Users\Angela Du\Desktop\new eps\fig8.eps')




#绘制校准曲线

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
#ax2 = plt.subplot2grid((3, 1), (2, 0))
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

clf_score = brier_score_loss(y_test, preds_prob_newsapsii[:,1], pos_label=1)
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, preds_prob_newsapsii[:,1], n_bins=10)
ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s (%1.3f)" % ('min,max,mean,std', clf_score))
#ax2.hist(prob_min_max_mean_std, range=(0, 1), bins=10, label='min,max,mean,std',histtype="step", lw=2)
ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

#ax2.set_xlabel("Mean predicted value")
#ax2.set_ylabel("Count")
#ax2.legend(loc="upper center", ncol=2)
plt.tight_layout()
plt.show()