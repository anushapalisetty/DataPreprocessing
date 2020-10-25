#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:15:56 2019

@author: anusha
"""


import pandas as pd

##read dataframe
df=pd.read_csv("Others.csv")
df_cat=df.astype('category')

##calculate summary table values
df_trans=df_cat.describe().transpose()
miss_val=df_cat.isnull().sum()
miss_val_per=miss_val/df_trans['count']*100
df_trans["% Miss."]=miss_val_per
mode_per=df_trans['freq']/df_trans['count']*100
df_trans["Mode %"]=mode_per
df_trans_rename=df_trans.rename(columns={'count':'Count','unique':'Card.','top':'Mode','freq':'Mode Freq.' })
print(df_trans_rename)


##calculate 2nd mode and its frequency
count=pd.DataFrame()
df_val=pd.DataFrame()
for (name,series) in df_cat.iteritems():
    count = df_cat[name].value_counts().nlargest(n=2).iloc[[1]]
    df_val=df_val.append(count)
df_val.sum().to_csv('mode.csv',header=False)
df_re=pd.read_csv('mode.csv',names=['2nd Mode','2nd Mode Freq.'])
df_re_n=df_re.rename(index={0:"Attr 0",1:"Attr 1",2:"Attr 2",3:"Attr 3",4:"Labels"})
result=pd.concat([df_trans_rename, df_re_n], axis=1, ignore_index=False)

##calculating 2nd mode freq percentile
mode2_per=result['2nd Mode Freq.']/result['Count']*100
result['2nd Mode %']=mode2_per

##selecting column in order of the summary table report
df_summ=result[['Count','% Miss.','Card.','Mode','Mode Freq.','Mode %','2nd Mode','2nd Mode Freq.','2nd Mode %']]
print(df_summ)






