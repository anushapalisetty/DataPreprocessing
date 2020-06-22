
#%%
import pandas as pd
import numpy as np

df=pd.read_csv('cancerNormalized.csv')

X=df.iloc[:,0:30].values
y=df['Class'].values

df_train = pd.read_csv('stratified_training.csv')
df_test = pd.read_csv('stratified_testing.csv')

X_train = df_train.iloc[:,:30]
X_test = df_test.iloc[:,:30]
y_train = df_train.iloc[:,30]
y_test = df_test.iloc[:,30]



from sklearn.ensemble import RandomForestClassifier
results=pd.DataFrame(columns=['Count of Trees','Score for training','Score for testing','Root Mean Square Error'])
indexR=1
indexG=1

resultsgini=pd.DataFrame(columns=['Count of Trees','Score for training','Score for testing','Root Mean Square Error'])


feat_labels=df.columns[:31]
forest=RandomForestClassifier(n_estimators=10000,n_jobs=-1,random_state=0)
forest.fit(X_train,y_train)
importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print(feat_labels[indices[f]],importances[indices[f]])

from sklearn import metrics
for size in range(1,100,10):
    forest=RandomForestClassifier(criterion='entropy',n_estimators=size,max_depth=3,random_state=0)
    forest.fit(X_train,y_train)
    scoreTrain=forest.score(X_train,y_train)
    scoreTest=forest.score(X_test,y_test)
    y_pred=forest.predict(X_test)
    mserr=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    results.loc[indexR]=[size,scoreTrain,scoreTest,mserr]
    indexR=indexR+1
    
for size in range(1,100,10):
    forest=RandomForestClassifier(criterion='gini',n_estimators=size,max_depth=3,random_state=0)
    forest.fit(X_train,y_train)
    scoreTrain=forest.score(X_train,y_train)
    scoreTest=forest.score(X_test,y_test)
    y_pred=forest.predict(X_test)
    mserr=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    resultsgini.loc[indexG]=[size,scoreTrain,scoreTest,mserr]
    indexG=indexG+1


print(results)
results.pop('Count of Trees')
results.pop('Root Mean Square Error')
ax=results.plot()
ax.set_title('RandomForest for Entropy')
ax.set_xlabel('Index')
ax.set_ylabel('Score') 


print('gini',resultsgini)
resultsgini.pop('Count of Trees')
resultsgini.pop('Root Mean Square Error')
ax=resultsgini.plot()
ax.set_title('RandomForest for Gini')
ax.set_xlabel('Index')
ax.set_ylabel('Score') 