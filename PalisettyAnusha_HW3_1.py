#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import seaborn as sns
import numpy as np

df=pd.read_csv('cancerData.csv')

print(df.shape) 


sns.set(style='ticks')
sns.pairplot(df,hue='Class',palette='husl')


ax=df['Class'].value_counts().plot(kind='bar')
ax.set_title('Class Values')
ax.set_xlabel('Values of Class')
ax.set_ylabel('Frequency') 

class_mapping={labels:idx for idx,labels in enumerate(np.unique(df['Class']))}
df['Class']=df['Class'].map(class_mapping)


n=[]
for name,series in df.loc[:,'radius1':'fractal dimension3'].iteritems():
    k=name
    n.append(k)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df.iloc[:,1:31])
norm_data=pd.DataFrame(scaler.fit_transform(df.iloc[:,1:31]),columns=n)
norm_data.describe().to_csv('norm1.csv')


scaler=MinMaxScaler(feature_range=(0,3))
scaler.fit(df.loc[:,'radius1':'fractal dimension3'])
norm_data_r=pd.DataFrame(scaler.fit_transform(df.loc[:,'radius1':'fractal dimension3']),columns=n)
norm_data_r['Class']=df['Class']
norm_data_r.to_csv('cancerNormalized.csv',index=False)
norm_data_r.describe().to_csv('norm3.csv')


#%%

