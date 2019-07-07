# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:53:48 2019

@author: cherry chen
"""

import pandas as pd
import psycopg2
import os

class get_data:
    def __init__(self):
        self.code_dict_chartevents={
                   'gcsverbal_1':723,
                   'gcsverbal_2':223900,
                   'gcsmotor_1':454,
                   'gcsmotor_2':223901,
                   'gcseyes_1':184,
                   'gcseyes_2':220739,
                   'sbp_1':51,
                   'sbp_2':442,
                   'sbp_3':455,
                   'sbp_4':6701,
                   'sbp_5':220050,
                   'sbp_6':220179,
                   'hr_1':211,
                   'hr_2':220045,
                   't_c1':676,
                   't_c2':223762,
                   't_f1':678,
                   't_f2':223761,
                   'fio2_1':190,
                   'fio2_2':3420,
                   'fio2_3':3422,
                   'fio2_4':223835}
        self.code_dict_outputevents={
                   'urine_1':40055,
                   'urine_2':40069,
                   'urine_3':226559,
                   'urine_4':226560}
        self.code_dict_labevents={
                   'po2':50821,
                   'bun':51006,
                   'wbc_1':51301,
                   'wbc_2':51300,
                   'potassium_1':50822,
                   'potassium_2':50971,
                   'sodium_1':950824,
                   'sodium_2':50983,
                   'bilirubin':50885,
                   'bicarbonate':50882}
        self.dir_path=os.path.dirname(os.path.realpath(__file__))

    def select_request(self,host,port,db_name,user_name,pwd, sql_str):
        con=psycopg2.connect(host=host,port=port,dbname=db_name,user=user_name,password=pwd)
        cur=con.cursor()
        cur.execute(sql_str)
        rows=cur.fetchall()
        columns_name=[desc[0] for desc in cur.description]
        res_df=pd.DataFrame(rows)
        if(len(rows)>0):
            res_df.columns=columns_name
        return res_df
    
if __name__=='__main__':
    db_name='mimic'
    host='localhost'
    port='5432'
    user_name='postgres'
    pwd='postgre'
    gd=get_data()
    a=gd.select_request(host,port,db_name,user_name,pwd,'select * from mimiciii.a_patientlist_diag')
    a=a.fillna(0)
    a.to_csv(os.path.join('E:','/lml_dataget/v3/diag.csv'),index=False, encoding='utf-8-sig')
    
#    for code in gd.code_dict_chartevents:
#        sql_str_chartevents="select subject_id, hadm_id, icustay_id, itemid, charttime, intime, valuenum as valuenum_"+code.split('_')[0]+", valueuom as valueuom_"+code.split('_')[0]+", interval_chart from (select subject_id, hadm_id, icustay_id, itemid, valuenum, valueuom, charttime, intime, extract(day from age(t.charttime, t.intime)) as interval_chart from (select a.subject_id, a.hadm_id, a.intime, b.icustay_id, b.itemid, b.charttime, b.valuenum, b.valueuom from mimiciii.a_patients_v3 a left join mimiciii.chartevents b on a.icustay_id=b.icustay_id) t) tt where itemid=" +str(gd.code_dict_chartevents[code])+" and interval_chart=0 and intime<charttime and valuenum is not null"
#        a=gd.select_request(host,port,db_name,user_name,pwd,sql_str_chartevents)
#        a.to_csv(os.path.join('E:','/lml_dataget/v3/first24h_'+code+'.csv'),index=False, encoding='utf-8-sig')
#    
#    for code in gd.code_dict_labevents:
#        sql_str_labevents="select subject_id, hadm_id, itemid, charttime, intime, value as value_"+code.split('_')[0]+", valuenum as valuenum_"+code.split('_')[0]+", valueuom as valueuom_"+code.split('_')[0]+", interval_chart from (select subject_id, hadm_id, itemid, value, valuenum, valueuom, charttime, intime, extract(day from age(t.charttime, t.intime)) as interval_chart from (select a.subject_id, a.hadm_id, a.intime, b.itemid, b.charttime, b.value, b.valuenum, b.valueuom from mimiciii.a_patients_v3 a left join mimiciii.labevents b on a.hadm_id=b.hadm_id) t) tt where itemid=" +str(gd.code_dict_labevents[code])+" and interval_chart=0 and intime<charttime and value is not null"
#        a=gd.select_request(host,port,db_name,user_name,pwd,sql_str_labevents)
#        a.to_csv(os.path.join('E:','/lml_dataget/v3/first24h_'+code+'.csv'),index=False, encoding='utf-8-sig')
#        
#    for code in gd.code_dict_outputevents:
#        sql_str_outputevents="select subject_id, hadm_id, itemid, charttime, intime, value as valuenum_"+code.split('_')[0]+", valueuom as valueuom_"+code.split('_')[0]+", interval_chart from (select subject_id, hadm_id, itemid, value, valueuom, charttime, intime, extract(day from age(t.charttime, t.intime)) as interval_chart from (select a.subject_id, a.hadm_id, a.intime, b.itemid, b.charttime, b.value, b.valueuom from mimiciii.a_patients_v3 a left join mimiciii.outputevents b on a.hadm_id=b.hadm_id) t) tt where itemid=" +str(gd.code_dict_outputevents[code])+" and interval_chart=0 and intime<charttime and value is not null"
#        a=gd.select_request(host,port,db_name,user_name,pwd,sql_str_outputevents)
#        a.to_csv(os.path.join('E:','/lml_dataget/v3/first24h_'+code+'.csv'),index=False, encoding='utf-8-sig')

