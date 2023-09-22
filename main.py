from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
#split it in features and labels
X = iris.data  #features in 2D array
y = iris.target  #labels in 1D array


#print(X.shape)   
#print(y.shape)

#hours of study vs good/bad grades
#10 different students
#train with 8
#predict with remaining 2
#level of accuracy

#test_size = 0.2 --> 20% is test case
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2)  #X is parameter for feature data,  y is parameter for label data. 
#sizes of each train and test sets for features and labels
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)








