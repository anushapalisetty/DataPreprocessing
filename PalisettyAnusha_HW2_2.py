#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:02:13 2019

@author: anusha
"""


import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns

open("Quantitative.csv")
df_quan=pd.read_csv("Quantitative.csv")


##calculating unique values
df_c=df_quan.nunique()

##calculating missing values
df_m=df_quan.isnull().sum()



#Summary Table
df_desc=df_quan.describe()
print(df_desc)
df_trans=df_desc.transpose()
df_trans_rename=df_trans.rename(columns={'count':'Count','min':'Min','std':'Std. Dev.','25%':'1st Qrt.','mean':'Mean','50%':'Median','75%':'3rd Qrt.','max':'Max' })
df_trans_rename['Card.']=df_c
df_trans_rename['% Miss.']=df_m/df_trans['count']*100
df_summ=df_trans_rename[['Count','% Miss.','Card.','Min','1st Qrt.','Mean','Median','3rd Qrt.','Max','Std. Dev.']]
df_summ.to_csv("Summary_Cat.csv")



##Histogram plots
ax=df_quan["Attr 4"].plot.hist(bins=20)
ax.set_title("Attribute: Attr 4- Histogram with 20 bins")
ax.set_xlabel("Attr 4")
ax.set_ylabel("Counts")
fig=ax.figure
fig.set_size_inches(5,3)
fig.tight_layout(pad=1)
fig.savefig("Attr 4.png",dpi=600)
plt.close(fig)

ax=df_quan["Attr 5"].plot.hist(bins=50)
ax.set_title("Attribute: Attr 5- Histogram with 50 bins")
ax.set_xlabel("Attr 5")
ax.set_ylabel("Counts")
fig=ax.figure
fig.set_size_inches(5,3)
fig.tight_layout(pad=1)
fig.savefig("Attr 5.png",dpi=600)
plt.close(fig)

ax=df_quan["Attr 6"].plot.hist(bins=60)
ax.set_title("Attribute: Attr 6- Histogram with 60 bins")
ax.set_xlabel("Attr 6")
ax.set_ylabel("Counts")
fig=ax.figure
fig.set_size_inches(5,3)
fig.tight_layout(pad=1)
fig.savefig("Attr 6.png",dpi=600)
plt.close(fig)

ax=df_quan["Attr 7"].plot.hist(bins=50)
ax.set_title("Attribute: Attr 7- Histogram with 50 bins")
ax.set_xlabel("Attr 7")
ax.set_ylabel("Counts")
fig=ax.figure
fig.set_size_inches(5,3)
fig.tight_layout(pad=1)
fig.savefig("Attr 7.png",dpi=600)
plt.close(fig)

ax=df_quan["Attr 8"].plot.hist(bins=30)
ax.set_title("Attribute: Attr 8- Histogram with 30 bins")
ax.set_xlabel("Attr 8")
ax.set_ylabel("Counts")
fig=ax.figure
fig.set_size_inches(5,3)
fig.tight_layout(pad=1)
fig.savefig("Attr 8.png",dpi=600)
plt.close(fig)

ax=df_quan["Attr 9"].plot.hist(bins=30)
ax.set_title("Attribute: Attr 9- Histogram with 30 bins")
ax.set_xlabel("Attr 9")
ax.set_ylabel("Counts")
fig=ax.figure
fig.set_size_inches(5,3)
fig.tight_layout(pad=1)
fig.savefig("Attr 9.png",dpi=600)
plt.close(fig)

ax=df_quan["Attr 10"].plot.hist(bins=30)
ax.set_title("Attribute: Attr 10- Histogram with 30 bins")
ax.set_xlabel("Attr 10")
ax.set_ylabel("Counts")
fig=ax.figure
fig.set_size_inches(5,3)
fig.tight_layout(pad=1)
fig.savefig("Attr 10.png",dpi=600)
plt.close(fig)

ax=df_quan["Attr 11"].plot.hist(bins=30)
ax.set_title("Attribute: Attr 11- Histogram with 30 bins")
ax.set_xlabel("Attr 11")
ax.set_ylabel("Counts")
fig=ax.figure
fig.set_size_inches(5,3)
fig.tight_layout(pad=1)
fig.savefig("Attr 11.png",dpi=600)
plt.close(fig)


ax=df_quan["Attr 12"].plot.hist(bins=60)
ax.set_title("Attribute: Attr 12- Histogram with 60 bins")
ax.set_xlabel("Attr 12")
ax.set_ylabel("Counts")
fig=ax.figure
fig.set_size_inches(5,3)
fig.tight_layout(pad=1)
fig.savefig("Attr 12.png",dpi=600)
plt.close(fig)




#violin plots
for i, col in enumerate(df_quan.columns):
    plt.figure(i)
    sns.violinplot(x=df_quan[col],palette="Blues")
    plt.title('Violin Plot for :'+col)
    
    
#Creating scatter plots
sns.pairplot(df_quan)


#covraiance   heatmaps
print(df_quan.cov())
sns.heatmap(df_quan.cov().astype(float), annot=True,cmap="Greens")
plt.title("Covariance Heatmap")


# Correlation and heatmaps

print(df_quan.corr())
fig = plt.figure() # width x height
ax1 = fig.add_subplot(1, 1, 1) # row, column, position
sns.heatmap(df_quan.corr(),ax=ax1, annot=True,cmap="Blues")
plt.title("Correlation Heatmaps")







    







