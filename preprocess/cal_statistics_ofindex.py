# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:25:43 2019

@author: Jolin
"""

import pandas as pd
import os
import datetime
#import numpy as np

file_list=os.listdir(r'E:\lml_dataget\v3\data_corhort')
for file in file_list[7:]:
    index_name=file.split('_')[1].replace('.csv','')
    df_file=pd.read_csv(r'E:\lml_dataget\v3\data_corhort\\'+file,header=0)
    df_file['timetime']=[datetime.datetime.strptime(df_file['charttime'][i], "%Y/%m/%d %H:%M") for i in range(len(df_file))]
    df_file=df_file.sort_values('timetime')
    if('icustay_id' in df_file.columns):
        df_file=df_file.drop(['icustay_id'],axis=1)
    df_file=df_file.drop(['subject_id','itemid','timetime','valueuom_'+index_name,'charttime', 'intime', 'interval_chart'],axis=1)
    
    r=df_file.groupby('hadm_id')
    res=pd.DataFrame(r.min())
    column=index_name
    res.columns=['min_'+str(column)]
    res['max_'+column]=r.max()
    res['minmax_'+column]=r.max()-r.min()
    res['mean_'+column]=r.mean()
    res['std_'+column]=r.std()
    res['stdmean_'+column]=r.std()/r.mean()
    res['median_'+column]=r.median()
    res['qua25_'+column]=r.quantile(0.25)
    res['qua75_'+column]=r.quantile(0.75)
    res['qua2575_'+column]=r.quantile(0.75)-r.quantile(0.25)
    res['mode_'+column]=r.apply(lambda df:(df['valuenum_'+column]).mode()[0])
    res['skew_'+column]=r.apply(lambda df:(df['valuenum_'+column]).skew())                                                                 
    res['kurt_'+column]=r.apply(lambda df:(df['valuenum_'+column]).kurt())    
    aaa=r.head(1).sort_values(['hadm_id'])
    aaa.index=aaa['hadm_id']
    res['first_'+column]=aaa['valuenum_'+column]
    res.to_csv('E:\\lml_dataget\\v3\\stats_res\\stats_'+str(column)+'.csv')