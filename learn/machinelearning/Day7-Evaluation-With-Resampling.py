# The dataset used to train an algorithm cannot be used to give you reliable
# estimates of the accuracy of the model on new data. This is a big problem because
# the whole idea of creating the model is to make predictions on new data.

# You can use statistical methods called resampling methods to split your training
# dataset up into subsets, some are used to train the model and others are held back and
# used to estimate the accuracy of the model on unseen data. 
# Split a dataset into training and test sets.
# Estimate the accuracy of an algorithm using k-fold cross validation.
# Estimate the accuracy of an algorithm using leave one out cross validation.
# The snippet below uses scikit-learn to estimate the accuracy of the Logistic Regression
# algorithm on the Pima Indians onset of diabetes dataset using 10-fold cross validation.

# Tips to remember
#   This requires anaconda 4.2.0+ or scikit-learn 0.18+
#   White space matters and is used for indicating code blocks.
#

# Evaluate using Cross Validation
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# split dataset into training and test sets using kfold
# estimate accuracy using k-fold cross validation
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
