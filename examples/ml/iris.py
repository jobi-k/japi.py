# iris is a common data set used for demonstrating the
# concepts of machine learning in python.
# the example is based on a supervised learning pattern where
# both the input and output is known.

# ~~~~~ imports ~~~~~
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#    load dataset
datafile = 'iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(datafile, names=names)

#    shape our dataset
#    this should tell us the number of rows and columns (150,5)
print(dataset.shape)

#    peek at the top of the data. aka: show the head of the top 20 records
#    should include the headers and look like
#    sepal-length   sepal-width  petal-length   petal-width      class
# 0      5.1             3.5         1.4             0.2         Iris-Setosa
# 1      4.9             3.0         1.4             0.2         Iris-Setosa
# ....
# 19     5.1             3.8         1.5             0.3         Iris-Setosa
print(dataset.head(20))

#    summarize our attributes
#    total count, mean, min and max values
print(dataset.describe())

#    identify the distribution of each class (flower type)
#    for this example we have three groups distributed evenly
print(dataset.groupby('class').size())

#    python libs for machine learning are awesome.  check this out
#    visualization of our data - univariate plots (or each individual variable)
dataset.plot(kind='box', subplots=True, sharex=False, sharey=False)
plt.show()
#    note that the above is a modal popup.
#    to continue execution you must close it
#    how about a histogram
dataset.hist()
plt.show()

#    visualization continued with multivariate plots
#    in other words, the interactions between our variables
#    using a scatter plot matrix
scatter_matrix(dataset)
plt.show()


#    so that was a ton of fun
#    now lets evaluate some of the available machine learning algorithms
#    step 1 - create a validation dataset
#    split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#    step 2 - create the test harness
#    split out dataset into 10 parts, train on 9 and test on 1
#    repleat for all combinations
num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'

#    step 3 - build models for evaluating six different algorithms
#    model 1 - Logistic Regression (LR)
#    model 2 - Linear Discriminant Analysis (LDA)
#    model 3 - K-Nearest Neighbors (KNN)
#    model 4-  Classification and Regression Trees (CART)
#    model 5 - Gaussian Naive Bayes (NB)
#    model 6 - Support Vector Machines (SVM)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#    evaluate each model
results = []
names = []
print('** model accuracy results **')
for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#    compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#    make predictions
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print('end')
