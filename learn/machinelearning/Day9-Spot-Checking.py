# Hi, you cannot possibly know which algorithm will perform best on your data beforehand.
#
# You have to discover it using a process of trial and error. I call this spot-checking algorithms. The scikit-learn library provides an interface to many machine learning algorithms and tools to compare the estimated accuracy of those algorithms.
#
# In this lesson, you must practice spot-checking different machine learning algorithms.
# Spot-check linear algorithms on a dataset (e.g. linear regression, logistic regression and linear discriminate analysis).
# Spot-check some nonlinear algorithms on a dataset (e.g. KNN, SVM and CART).
# Spot-check some sophisticated ensemble algorithms on a dataset (e.g. random forest and stochastic gradient boosting).
# For example, the snippet below spot-checks the K-Nearest Neighbors algorithm on the Boston House Price dataset.
# KNN Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
url = "https://goo.gl/sXleFv"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())