# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:16:16 2019

@author: Jolin
"""

import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd

def ga_main(kfold,x_train_s,y_train_s,x_val,y_val):
    population_size=10
    chromosome_length=14
    cof=['age_interval','admission_type_EMERGENCY','admission_type_ELECTIVE','admission_type_URGENT','aids','hem','mets']
    stats_list=['min','max','minmax','mean','std','stdmean','median','qua25','qua75','qua2575','mode','skew','kurt','first']
    pc=0.6
    pm=0.1
    population=pop=species_origin(population_size=population_size,chromosome_length=chromosome_length)  
    now_best_indiv=list()  #迄今为止最好的个体
    now_best_fitness=[random.random()*5 for _ in range(50)]  #迄今为止最好的适应度
    all_best_fitness=list()  #存储每一代最好的适应度
    all_best_indiv=list()  #存储每一代最好的个体
    n=1;
    f_res_n=[]
    f_res_nowbest=[]
    f=open(r'E:/lml_dataget/v3/random80/guocheng_'+str(kfold)+'.txt','a',encoding='utf-8')
    while (np.std(now_best_fitness[-50:])>0.001):
            #计算种群的适应度
        function1=function(population=population,x_train_s=x_train_s,y_train_s=y_train_s,x_val=x_val,y_val=y_val,cof=cof,stats_list=stats_list)
        fitness1=fitness(function1)
            #求种群中最好的个体以及对应的适应度值
        best_individual,best_fitness=best(population,fitness1)
        all_best_indiv.append(best_individual)
        all_best_fitness.append(best_fitness)
        nowbestfit=max(all_best_fitness)
        nowbestindv=all_best_indiv[all_best_fitness.index(max(all_best_fitness))]
        now_best_fitness.append(nowbestfit)
        now_best_indiv.append(nowbestindv) 
        f.write('----------第'+str(n)+'次的结果--------------'+':::'+str([best_individual,best_fitness])+'\n')
        f.write('----------目前最好的结果--------------'+':::'+str([nowbestindv,nowbestfit])+'\n')
        print('----------第'+str(n)+'次的结果--------------'+'\n')
        print(str([best_individual,best_fitness]));
        print('----------目前最好的结果--------------'+'\n')
        print(str([nowbestindv,nowbestfit]));
        f_res_n.append([n,best_individual,best_fitness])
        f_res_nowbest.append([n,nowbestindv,nowbestfit])
        selection(population,fitness1,pop)#选择
        crossover(population,pc,pop)#交配
        mutation(population,pm)#变异
        n=n+1;
        if(n>300):
            break;
    f_res_n_df=pd.DataFrame(f_res_n,columns=['no','best_individual','best_fitness'])
    f_res_nowbest_df=pd.DataFrame(f_res_nowbest,columns=['no','best_individual_now','best_fitness_now'])
    f_res_n_df.to_csv(r'E:\lml_dataget\v3\random80\\'+str(kfold)+'_flod_res.csv',index=False,encoding='utf-8-sig')
    f_res_nowbest_df.to_csv(r'E:\lml_dataget\v3\random80\\'+str(kfold)+'_flod_res_nowbest.csv',index=False,encoding='utf-8-sig')
    f.close()
    return [f_res_n,f_res_nowbest]

def species_origin(population_size,chromosome_length):  
    population=[[]]  
    while(len(population)<=population_size):
        temporary=[]
        for j in range(chromosome_length):  
            temporary.append(random.randint(0,1))
#            if(sum(temporary)==7):
        population.append(temporary)  
                #将染色体添加到种群中  
    return population[1:]
    
def function(population,x_train_s,y_train_s,x_val,y_val,cof,stats_list):
    function1=[]
    coff=cof
    for ii in range(len(population)):
        indiv=population[ii]
        id1 = [i for i,x in enumerate(indiv) if x==1]
        stats=[stats_list[i] for i in id1]
        for sts in stats:
            for column in x_train_s.columns:
                if(sts == column.split('_')[0]):
                    coff.append(column)
            
        x_train_ss=pd.DataFrame(x_train_s,columns=x_train_s.columns)[coff]
        y_train_ss=y_train_s
        x_vall=x_val[coff]
        y_vall=y_val
        
        rf=RandomForestClassifier(random_state=42,n_estimators=50)
        rf.fit(x_train_ss,y_train_ss)
        preds_rf=rf.predict_proba(x_vall)
        res=roc_auc_score(y_vall,preds_rf[:,1])
        function1.append(res*100)
    return function1
    
def fitness(function1):
    fitness1=[]
    mf=0
    for i in range(len(function1)):
        if(function1[i]+mf>0):
            temporary=mf+function1[i]
        else:
            temporary=0.0
            # 如果适应度小于0,则定为0
        fitness1.append(temporary)
            #将适应度添加到列表中
    return fitness1
    
    #计算适应度和
def sum(fitness1):
    total=0
    for i in range(len(fitness1)):
        total+=fitness1[i]
    return total
     
    #计算适应度斐波纳挈列表，这里是为了求出累积的适应度
def cumsum(fitness1):
    for i in range(len(fitness1)-2,-1,-1):
            # range(start,stop,[step])
            # 倒计数
        total=0
        j=0
        while(j<=i):
            total+=fitness1[j]
            j+=1
        #这里是为了将适应度划分成区间
        fitness1[i]=total
        fitness1[len(fitness1)-1]=1
    
    #3.选择种群中个体适应度最大的个体
def selection(population,fitness1,pop):
    new_fitness=[]
        #单个公式暂存器
    total_fitness=sum(fitness1)
        #将所有的适应度求和
    for i in range(len(fitness1)):
        new_fitness.append(fitness1[i]/total_fitness)
        #将所有个体的适应度概率化,类似于softmax
    cumsum(new_fitness)
        #将所有个体的适应度划分成区间
    ms=[]
        #存活的种群
    pop_len=len(population)
        #求出种群长度
        #根据随机数确定哪几个能存活
     
    for i in range(pop_len):
        ms.append(random.random())
        # 产生种群个数的随机值
    ms.sort()
        # 存活的种群排序
    fitin=0
    newin=0
    new_pop=population
     
        #轮盘赌方式
    while newin<pop_len:
        if(ms[newin]<new_fitness[fitin]):
            new_pop[newin]=pop[fitin]
            newin+=1
        else:
            fitin+=1
    population=new_pop
    
def crossover(population,pc,pop):
    #pc是概率阈值，选择单点交叉还是多点交叉，生成新的交叉个体，这里没用
    pop_len=len(population)
     
    for i in range(pop_len-1):
        cpoint=random.randint(0,len(population[0]))
            #在种群个数内随机生成单点交叉点
        temporary1=[]
        temporary2=[]
     
        temporary1.extend(pop[i][0:cpoint])
        temporary1.extend(pop[i+1][cpoint:len(population[i])])
            #将tmporary1作为暂存器，暂时存放第i个染色体中的前0到cpoint个基因，
            #然后再把第i+1个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
     
        temporary2.extend(pop[i+1][0:cpoint])
        temporary2.extend(pop[i][cpoint:len(pop[i])])
            # 将tmporary2作为暂存器，暂时存放第i+1个染色体中的前0到cpoint个基因，
            # 然后再把第i个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
        pop[i]=temporary1
        pop[i+1]=temporary2
            # 第i个染色体和第i+1个染色体基因重组/交叉完成
    
    #step4：突变
def mutation(population,pm):
        # pm是概率阈值
    px=len(population)
        # 求出种群中所有种群/个体的个数
    py=len(population[0])
        # 染色体/个体中基因的个数
    for i in range(px):
        if(random.random()<pm):
            #如果小于阈值就变异
            mpoint=random.randint(0,py-1)
                # 生成0到py-1的随机数
            if(population[i][mpoint]==1):
                #将mpoint个基因进行单点随机变异，变为0或者1
                population[i][mpoint]=0
            else:
                population[i][mpoint]=1
    
    
    #寻找最好的适应度和个体
def best(population,fitness1):
     
    px=len(population)
    bestindividual=population[0]
    bestfitness=fitness1[0]
     
    for i in range(1,px):
       # 循环找出最大的适应度，适应度最大的也就是最好的个体
       if(fitness1[i]>bestfitness):
     
           bestfitness=fitness1[i]
           bestindividual=population[i]
     
    return [bestindividual,bestfitness]
       