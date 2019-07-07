# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:59:04 2019

@author: eileenlu
"""

import os
import pandas as pd

file_list=os.listdir(r'E:\lml_dataget\v3\data')
patient_list=pd.read_csv(os.path.join(r'E:\lml_dataget\v3','patientlist.csv'),header=0)
for file in file_list:
    index_name=file.split('_')[1].replace('.csv','')
    df_file=pd.read_csv(r'E:\lml_dataget\v3\data\\'+file,header=0)
    group_df_file=df_file.groupby(['hadm_id'])
    group_count=pd.DataFrame(group_df_file.count().iloc[:,0]).reset_index()
    group_count.columns=['hadm_id',index_name+'_count']
    patient_list=pd.merge(left=patient_list,right=group_count,how='left')

patient_list=patient_list.fillna(0)

patient_list1=patient_list.iloc[:,14:]
patient_list1.index=patient_list['hadm_id']
patient_list2=patient_list1.drop(['bilirubin_count'],axis=1)
patient_list2['missing_rate']=1-(patient_list2 >= 1).sum(axis=1)/14
patient_list3=patient_list2[patient_list2['missing_rate']<0.1]

patient_liat11=patient_list3[patient_list3['gcseyes_count']>3]
patient_liat11=patient_liat11[patient_liat11['gcsmotor_count']>3]
patient_liat11=patient_liat11[patient_liat11['gcsverbal_count']>3]
patient_liat11=patient_liat11[patient_liat11['hr_count']>3]
patient_liat11=patient_liat11[patient_liat11['t_count']>3]
patient_liat11=patient_liat11[patient_liat11['sbp_count']>3]
patient_liat11=patient_liat11[patient_liat11['urine_count']>3]
patient_liat11=patient_liat11[patient_liat11['bicarbonate_count']>1]
patient_liat11=patient_liat11[patient_liat11['wbc_count']>1]
patient_liat11=patient_liat11[patient_liat11['sodium_count']>1]
patient_liat11=patient_liat11[patient_liat11['potassium_count']>1]
patient_liat11=patient_liat11[patient_liat11['bun_count']>1]
patient_liat11=patient_liat11[patient_liat11['fio2_count']>1]
patient_liat12=patient_liat11[patient_liat11['po2_count']>1]

patient_liat11.to_csv('E:\\lml_dataget\\lml_dataget\\delete_po2_patientlist.csv')
patient_liat12.to_csv('E:\\lml_dataget\\v3\po2fio2_than_patientlist_v3.csv')

#aaa=pd.read_csv(r'E:\\lml_dataget\\v3\data\first24h_sbp_1.csv',header=0)

#
#
#patient_list=patient_liat11
#
#
#
#delta_list=[1,2,3,4,5]
#res=[]
#for delta in delta_list:
#    res_delta=(patient_list.iloc[:,7:] >= delta).sum(axis=0).reset_index()
#    res.append(list(res_delta.iloc[:,1]))
#    
#res_df=pd.DataFrame(res,index=['完全缺失率','至少含一个值','至少含两个值','至少含三个值','至少含四个值'],columns=res_delta['index'])
#res_df_missing=1-res_df/len(patient_list)
#
#res_df_missing.to_csv(r'E:\lml_dataget\lml_dataget\res_missing_index.csv',encoding='utf-8-sig')
#
#res_df.to_csv(r'E:\lml_dataget\lml_dataget\res_miss_count_index.csv',encoding='utf-8-sig')
#
#
#
#res_p=[]
#for delta in delta_list:
#    res_delta_p=(patient_list.iloc[:,7:] >= delta).sum(axis=1).reset_index()
#    res_p.append(list(res_delta_p.iloc[:,1]))
#    
#res_df_p=pd.DataFrame(res_p,index=['完全缺失率','至少含一个值','至少含两个值','至少含三个值','至少含四个值'],columns=patient_list['hadm_id'])
#res_df_missing_p=1-res_df_p/15
#
#res_df_p_t=res_df_p.transpose()
#res_df_missing_p_t=res_df_missing_p.transpose()
#res_df_missing_p_t.to_csv(r'E:\lml_dataget\lml_dataget\res_missing_patient.csv',encoding='utf-8-sig')
#
#res_df_p_t.to_csv(r'E:\lml_dataget\lml_dataget\res_miss_count_patient.csv',encoding='utf-8-sig')
#
#
#patient_liat11=patient_list[patient_list['gcseyes_1_count']>2]
#patient_liat11=patient_liat11[patient_liat11['t_f1_count']>2]
#patient_liat11=patient_liat11[patient_liat11['bicarbonate_count']>1]
#
#patient_list=patient_liat11





#########################取最终患者列表的数据集
patientlist=pd.read_csv(r'E:/lml_dataget/v3/po2fio2_than_patientlist_v3.csv',header=0)
for file1 in file_list:
    file_df=pd.read_csv(r'E:/lml_dataget/v3/data/'+file1,header=0)
    file_df_merge=pd.merge(left=patientlist,right=file_df,how='left',on=['hadm_id'])
    file_df_merge.to_csv(r'E:/lml_dataget/v3/data_corhort/'+file1,index=False)
    

    