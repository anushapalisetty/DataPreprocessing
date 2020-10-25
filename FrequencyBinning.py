#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:28:40 2019

@author: anusha
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_quan=pd.read_csv("Quantitative.csv")

df_s=pd.DataFrame()
df_mean=pd.DataFrame()
df_binned=pd.DataFrame()
attName='_BIN'
df_bord=pd.DataFrame()
l=list()

##Sorting the data and returning the bins with mean values
      
def print_mean(name):
    df_sort=pd.DataFrame()
    a=[]
    df_sort["Column"]=df_quan[name].sort_values()
    for i in range(0,1000):
        a.append(i)
    df_sort["temp"+name]=df_sort.index.values
    df_sort.index=a 
    for i in range(0,len(df_quan),50):
        bin_data=list()
        for j in range(0,50):
            bin_data.append(df_sort.loc[i+j,"Column"])
        min_val=min(bin_data)
        max_val=max(bin_data)
        mean=(max_val+min_val)/2
        for k in range(i,i+50):
            df_sort.loc[k,name+attName]=mean
    df_sort.index=df_sort["temp"+name]
    df_s=df_sort.sort_index()
    df_sort.drop(columns=["Column",name+attName,"temp"+name])
    return df_s[name+attName]


##Calculating bin borders to plot equal frequency binning
    
def print_borders(name):
    df_sort=pd.DataFrame()
    a=[]
    b=list()
    df_sort["Column"]=df_quan[name].sort_values()
    for i in range(0,1000):
        a.append(i)
    df_sort["temp"+name]=df_sort.index.values
    df_sort.index=a 
    for i in range(0,len(df_quan),50):
        bin_data=list()
        for j in range(0,50):
            bin_data.append(df_sort.loc[i+j,"Column"])
        min_val=min(bin_data)
        b.append(min_val)
    return b

##main function which calls mean and border function
for name,series in df_quan.iteritems():
    df_mean[name+attName]=print_mean(name)

for name,series in df_quan.iteritems():
    df_bord[name+attName]=print_borders(name)


##plot equal frequency binning
plt.hist(df_mean["Attr 12_BIN"],bins=df_bord["Attr 12_BIN"])
plt.title("Equal Frequency Binning of Attr 12")
plt.xlabel("Attr 12")
plt.ylabel("Frequency")


###Join binned data with quantiative dataframe
df_join=df_quan.join(df_mean)

sns.heatmap(df_mean.corr(),annot=True,cmap='Blues')


#generate name columns
n=[]
for name,series in df_quan.iteritems():
        n.append(name)
        n.append(name+'_BIN')

df_binned=df_join[n]
print(df_binned)

###read to csv file
df_binned.to_csv("QuantitativeBinned.csv",index=False)


            
            






