
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

df=pd.read_csv('cancerNormalized.csv')
pd.set_option('expand_frame_repr', False)

df_train = pd.read_csv('stratified_training.csv')
df_test = pd.read_csv('stratified_testing.csv')

X_train = df_train.iloc[:,:30]
X_test = df_test.iloc[:,:30]
y_train = df_train.iloc[:,30]
y_test = df_test.iloc[:,30]

from sklearn.neighbors import KNeighborsClassifier
resultsKnn=pd.DataFrame(columns=['KNN' , 'Score for Training' , 'Score for Testing','Root Mean Square Error'])

for knnCount in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=knnCount,p=2,metric='minkowski',weights='distance')
    knn=knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    mserr=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    scoreTrain=knn.score(X_train,y_train)
    scoreTest=knn.score(X_test,y_test)
    resultsKnn.loc[knnCount]=[knnCount,scoreTrain,scoreTest,mserr]
    


###Cross Validation
from sklearn.model_selection import cross_val_score
X=df.iloc[:,0:30].values
y=df['Class'].values
neighbors = [x for x in range(1,20) if x % 2 != 0]
cv_scores = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
MSE = [1-x for x in cv_scores]
optimal_k_index = MSE.index(min(MSE))
optimal_k = neighbors[optimal_k_index]
print(optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
mserr=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
scoreTrain=knn.score(X_train,y_train)
scoreTest=knn.score(X_test,y_test)
print(scoreTest)

##Display Results for our KNN classifier
print(resultsKnn)
resultsKnn.to_csv('knn_r.csv')
resultsKnn.pop('KNN')
resultsKnn.pop('Root Mean Square Error')
ax=resultsKnn.plot()
ax.set_title('KNN for P=1')
ax.set_xlabel('Values of K')
ax.set_ylabel('Score')  








