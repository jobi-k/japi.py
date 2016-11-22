# Hi, once you have found one or two algorithms that perform well on your dataset, you may want to improve the performance of those models.

# One way to increase the performance of an algorithm is to tune it's parameters to your specific dataset.

# The scikit-learn library provides two ways to search for combinations of parameters for a machine learning algorithm:
#     Tune the parameters of an algorithm using a grid search that you specify.
# Tune the parameters of an algorithm using a random search.
# Your goal in today's lesson is to practice each search method.
#
# The snippet below uses is an example of using a grid search for the Ridge Regression algorithm on the Pima Indians onset of diabetes dataset.
# # Grid Search for Algorithm Tuning
from pandas import read_csv
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)