#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:34:47 2019

@author: anusha
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing


open("Quantitative.csv")
df_quan=pd.read_csv("Quantitative.csv")

df_out=pd.DataFrame()

df_desc=df_quan.describe()
df_trans=df_desc.transpose()


##Method to find outliers
df_out['3rd_max']=abs(df_trans['75%']-df_trans['max'])
df_out['median_3rd']=abs(df_trans['50%']-df_trans['75%'])
df_out['diff']=df_out['3rd_max'] - df_out['median_3rd']



df_out['1st_min']=abs(df_trans['25%']-df_trans['min'])
df_out['median_1st']=abs(df_trans['50%']-df_trans['25%'])
df_out['diff_min']=df_out['1st_min'] - df_out['median_1st']

df_out["outliers"]=((df_out['3rd_max'] > df_out['median_3rd']) | (df_out['1st_min']>df_out['median_1st']))
print(df_out)

#setting clamp thresholds
IQR=df_trans['75%']-df_trans['25%']

df_clamp=pd.DataFrame()
df_clamp['Lower']=df_trans['25%']-1.5*IQR
df_clamp['Upper']=df_trans['75%']+1.5*IQR


attr='_ClampedValues'



for name,series in df_quan.iteritems():
    df_quan[name+''+attr]=df_quan[name]
    #print(df_quan[attr+'_'+name].head())
    for i in df_quan[name].index.values:
        if df_quan.loc[i,name] < df_clamp.loc[name,'Lower']:
            df_quan.loc[i,name+''+attr]=df_clamp.loc[name,'Lower']
        elif df_quan.loc[i,name]>df_clamp.loc[name,'Upper']:
            df_quan.loc[i,name+''+attr]=df_clamp.loc[name,'Upper']
        else:
            df_quan.loc[i,name]=df_quan.loc[i,name]

df_qc=pd.DataFrame()

#All clamped data into one dataframe
df_qc=df_quan.iloc[:,9:18]


##Normalisation

##preparing columns for clamp normalization
df_qnc=pd.DataFrame()
attr1="_ClampedNormalizedValues"

n=[]
for name,series in df_qc.iteritems():
    k=name.split('_')[0]
    n.append(k+''+attr1)




#applying min max normalization
min_max=preprocessing.MinMaxScaler()
norm_data=pd.DataFrame(min_max.fit_transform(df_qc), columns = n)

print(norm_data.describe())



###Box plots
for (name,series) in norm_data.iteritems():
        plt.boxplot(norm_data[name])
        plt.title("Box plot of :"+name)
        plt.xlabel(name)
        plt.ylabel("Value")
        plt.show()
        
####Scatter matrix
        
df_spm=norm_data
sns.pairplot(df_spm)


#####Q_transferrres file

df_final=df_quan.join(norm_data)


df_transf=pd.DataFrame()
df_transf=df_final[['Attr 4','Attr 4_ClampedValues','Attr 4_ClampedNormalizedValues','Attr 5','Attr 5_ClampedValues','Attr 5_ClampedNormalizedValues','Attr 6','Attr 6_ClampedValues','Attr 6_ClampedNormalizedValues','Attr 7','Attr 7_ClampedValues','Attr 7_ClampedNormalizedValues','Attr 8','Attr 8_ClampedValues','Attr 8_ClampedNormalizedValues','Attr 9','Attr 9_ClampedValues','Attr 9_ClampedNormalizedValues','Attr 10','Attr 10_ClampedValues','Attr 10_ClampedNormalizedValues','Attr 11','Attr 11_ClampedValues','Attr 11_ClampedNormalizedValues','Attr 12','Attr 12_ClampedValues','Attr 12_ClampedNormalizedValues']]
df_transf.to_csv("QTransferred.csv",index=False)


        
        


#
