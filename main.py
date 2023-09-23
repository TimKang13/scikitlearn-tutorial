from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

iris = datasets.load_iris()
#split it in features and labels
X = iris.data  #features in 2D array
y = iris.target  #labels in 1D array

classes = ['Iris Setosa', 'Iris Verisicolour', 'Iris Virginica']

#print(X.shape)   
#print(y.shape)
#train with 80%
#predict with remaining 20%
#level of accuracy

#test_size = 0.2 --> 20% is test case
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2)  #X is parameter for feature data,  y is parameter for label data. 
#sizes of each train and test sets for features and labels
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

model = svm.SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("predictions", predictions)
print("actual", y_test)
#prints the names
#for i in range(len(predictions)):
    #print(classes[predictions[i]])

print(acc)








