
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('cancerNormalized.csv')
pd.set_option('expand_frame_repr', False)

X=df.iloc[:,0:30].values
y=df['Class'].values


sss = StratifiedShuffleSplit(n_splits=2,test_size=0.33, random_state=0)
sss.split(X, y)


for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

n=[]
for name,series in df.loc[:,'radius1':'fractal dimension3'].iteritems():
    k=name
    n.append(k)

df_train=pd.DataFrame(data=X_train,columns=n)
df_train['Class']=pd.DataFrame(data=y_train)
df_test=pd.DataFrame(data=X_test,columns=n)
df_test['Class']=pd.DataFrame(data=y_test)

df_train.to_csv('stratified_training.csv',index=False)
df_test.to_csv('stratified_testing.csv',index=False)

from sklearn.tree import DecisionTreeClassifier

resultsEntropy=pd.DataFrame(columns=['LevelLimit' , 'Score for Training' , 'Score for Testing',
                                     'Accuracy','Mean Square Error'])
resultsGini=pd.DataFrame(columns=['LevelLimit' , 'Score for Training' , 'Score for Testing',
                                     'Accuracy','Mean Square Error'])
from sklearn import metrics

for treeDepth in range(1,6):
    dct=DecisionTreeClassifier(criterion="entropy",max_depth=treeDepth,random_state=0)
    dct=dct.fit(X_train, y_train)
    y_pred=dct.predict(X_test)
    mserr=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    scoreTrain=dct.score(X_train,y_train)
    scoreTest=dct.score(X_test,y_test)
    correct = (y_test == y_pred).sum()
    incorrect = (y_test != y_pred).sum()
    accuracy = correct / (correct + incorrect) * 100
    export_graphviz(dct,out_file='tree'+str(treeDepth)+'.dot',filled=True, rounded=True,  
                     special_characters=True,feature_names=n)
    resultsEntropy.loc[treeDepth]=[treeDepth,scoreTrain,scoreTest,accuracy,mserr]
    
    
for treeDepth in range(1,6):
    dct=DecisionTreeClassifier(criterion="gini",max_depth=treeDepth,random_state=0)
    dct=dct.fit(X_train, y_train)
    y_pred=dct.predict(X_test)
    mserr=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    scoreTrain=dct.score(X_train,y_train)
    scoreTest=dct.score(X_test,y_test)
    correct = (y_test == y_pred).sum()
    incorrect = (y_test != y_pred).sum()
    accuracy = correct / (correct + incorrect) * 100
    resultsGini.loc[treeDepth]=[treeDepth,scoreTrain,scoreTest,accuracy,mserr]

from sklearn.tree import export_graphviz
dcy=DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=0)
dcy=dcy.fit(X_train, y_train)
export_graphviz(dcy,out_file='tree.dot',filled=True, rounded=True,  
                     special_characters=True,feature_names=n)

print(resultsEntropy)
resultsEntropy.pop('LevelLimit')
resultsEntropy.pop('Mean Square Error')
resultsEntropy.pop('Accuracy')
ax=resultsEntropy.plot()
ax.set_title('Decision Tree Plot using Entropy')
ax.set_xlabel('Depth of tree')
ax.set_ylabel('Score')

print('Gini',resultsGini)
resultsGini.pop('LevelLimit')
resultsGini.pop('Mean Square Error')
resultsGini.pop('Accuracy')
ax=resultsGini.plot()
ax.set_title('Decision Tree Plot using Gini')
ax.set_xlabel('Depth of tree')
ax.set_ylabel('Score')
                     
                     



    






