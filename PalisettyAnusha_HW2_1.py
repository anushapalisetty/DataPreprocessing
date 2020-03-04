# -*- coding: utf-8 -*-
"""
Spyder Editor

author:Anusha
"""

import pandas as pd

open("dataPrep.csv")

###Categorical data
df=pd.read_csv("dataPreP.csv")

df_cat=df.select_dtypes(include=['object'])

df_cat.to_csv("Others.csv",index=False)


###numeric data
df_num=df._get_numeric_data()

df_num.to_csv("Quantitative.csv",index=False)



