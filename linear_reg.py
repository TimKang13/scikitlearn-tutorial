from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt  

housing = fetch_california_housing()
X = housing.data
y = housing.target



print(X.shape)

print(y.shape)

#algorithm
l_reg = linear_model.LinearRegression()

#visualize with pyplot of matplotlib
plt.scatter(X.T[1], y)
plt.show

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Predictions: ", predictions)
print("R^2 value: ", l_reg.score(X,y))
print("coedd: ", l_reg.coef_)
print("intercept: ", l_reg.intercept_)