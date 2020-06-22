

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


pd.reset_option('expand_frame_repr')

############ Method 1###############
df_o=pd.read_csv('cancerData.csv')
class_mapping={labels:idx for idx,labels in enumerate(np.unique(df_o['Class']))}
df_o['Class']=df_o['Class'].map(class_mapping)
print(df_o.describe())

df_g=pd.DataFrame()
df_g['radius']=(df_o['radius1']+df_o['radius2']+df_o['radius3'])/3
df_g['texture']=(df_o['texture1']+df_o['texture2']+df_o['texture3'])/3
df_g['perimeter']=(df_o['perimeter1']+df_o['perimeter2']+df_o['perimeter3'])/3
df_g['area']=(df_o['area1']+df_o['area2']+df_o['area3'])/3
df_g['smoothness']=(df_o['smoothness1']+df_o['smoothness2']+df_o['smoothness3'])/3
df_g['compactness']=(df_o['compactness1']+df_o['compactness2']+df_o['compactness3'])/3
df_g['concavity']=(df_o['concavity1']+df_o['concavity2']+df_o['concavity3'])/3
df_g['concave points']=(df_o['concave points1']+df_o['concave points2']+df_o['concave points3'])/3
df_g['symmetry']=(df_o['symmetry1']+df_o['symmetry2']+df_o['symmetry3'])/3
df_g['fractal dimension']=(df_o['fractal dimension1']+df_o['fractal dimension2']+df_o['fractal dimension3'])/3
df_g['Class']=df_o['Class']
print(df_g.shape)
print(df_g.head())

n=[]
for name,series in df_g.loc[:,'radius':'fractal dimension'].iteritems():
    k=name
    n.append(k)

scaler=MinMaxScaler(feature_range=(0,3))
scaler.fit(df_g.loc[:,'radius':'fractal dimension'])
norm_data_r=pd.DataFrame(scaler.fit_transform(df_g.loc[:,'radius':'fractal dimension']),columns=n)
norm_data_r['Class']=df_g['Class']

X=norm_data_r.iloc[:,1:10].values
y=norm_data_r['Class'].values



######################################Method 2######################
df_d=pd.read_csv('cancerData.csv')
class_mapping={labels:idx for idx,labels in enumerate(np.unique(df_d['Class']))}
df_d['Class']=df_d['Class'].map(class_mapping)


df_d = df_d.drop(['symmetry1', 'compactness2', 'fractal dimension1', 'texture2','symmetry2' ], axis=1)
print(df_d.head())


n=[]
for name,series in df_d.loc[:,'radius1':'fractal dimension3'].iteritems():
    k=name
    n.append(k)

scaler=MinMaxScaler(feature_range=(0,3))
scaler.fit(df_d.loc[:,'radius1':'fractal dimension3'])
norm_data_r1=pd.DataFrame(scaler.fit_transform(df_d.loc[:,'radius1':'fractal dimension3']),columns=n)
norm_data_r1['Class']=df_d['Class']

X=norm_data_r1.iloc[:,0:25].values
y=norm_data_r1['Class'].values
sss = StratifiedShuffleSplit(n_splits=2,test_size=0.33, random_state=0)

for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


###################Method 3#######################################

df_c=pd.read_csv('cancerData.csv')
class_mapping={labels:idx for idx,labels in enumerate(np.unique(df_d['Class']))}
df_c['Class']=df_c['Class'].map(class_mapping)
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df_c.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,cmap='GnBu')

df_c = df_c.drop(['area2','perimeter2','perimeter3','radius3','area1','symmetry1', 'compactness2', 'fractal dimension1', 'texture2','symmetry2','smoothness2','perimeter1' ], axis=1)
print(df_c.head())

n=[]
for name,series in df_c.loc[:,'radius1':'fractal dimension3'].iteritems():
    k=name
    n.append(k)

scaler=MinMaxScaler(feature_range=(0,3))
scaler.fit(df_c.loc[:,'radius1':'fractal dimension3'])
norm_data_r2=pd.DataFrame(scaler.fit_transform(df_c.loc[:,'radius1':'fractal dimension3']),columns=n)
norm_data_r2['Class']=df_d['Class']

X=norm_data_r2.iloc[:,0:10].values
y=norm_data_r2['Class'].values
sss = StratifiedShuffleSplit(n_splits=2,test_size=0.33, random_state=0)

for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]



############################Method 4####################################

df_train = pd.read_csv('stratified_training.csv')
df_test = pd.read_csv('stratified_testing.csv')

X_train = df_train.iloc[:,:30]
X_test = df_test.iloc[:,:30]
y_train = df_train.iloc[:,30]
y_test = df_test.iloc[:,30]

# prepare models
models = []
models.append(('Logistic Regression', LogisticRegression(solver='lbfgs',C=100,max_iter=5000)))
models.append(('KNN', KNeighborsClassifier(n_neighbors=8,p=1,metric='minkowski',weights='distance')))
models.append(('DecisionTree', DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=0)))
models.append(('RAN', RandomForestClassifier(criterion='entropy',n_estimators=1000,random_state=0)))
models.append(('Random Forest', GaussianNB()))
models.append(('Support Vector Machine', SVC(gamma='scale',kernel='rbf',C=10)))

for name,model in models:
    model.fit(X_train,y_train)    
    y_pred=model.predict(X_test)
    correct = (y_test == y_pred).sum()
    incorrect = (y_test != y_pred).sum()
    accuracy = correct / (correct + incorrect) * 100
    print(name,accuracy)






