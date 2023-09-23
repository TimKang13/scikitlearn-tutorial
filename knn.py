import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from UCI Machine Learning Repository


#data is now tabular data from car.data
data = pd.read_csv('car.data')
#originally didn't have headings of features, so added them manually in car.data
#print(data.head())

#selecting feature columns we want to see
X = data[[ 
    'buying', 
    'maint',
    'safety'
]].values
y = data[['class']]  #label column
#print(X, y)

#converting data to numbers // using LabelEncoder
Le = LabelEncoder()
for i in range(len(X[0])): # traversing, number of features in X
    X[:, i] = Le.fit_transform(X[:, i])  
#print(X) #we can see we turned string tabular data into number vector

#converting // using dictionary
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping)  #see key, convert to value
y = np.array(y)
#print(y)

#create model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights ='uniform')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
knn.fit(X_train, y_train) #trains the model

prediction = knn.predict(X_test) #tests for test features 
accuracy = metrics.accuracy_score(y_test, prediction)  #accuracy, by looking at prediction by features and the actual label
print("predictions: ", prediction)
print("accuracy: ", accuracy)

a=1727
print("actual value: ", y[a])
print("predicted value: ", knn.predict(X)[a])


    

