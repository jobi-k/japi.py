
# ~~~~~ imports ~~~~~
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import pickle
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
#    credit-ecoa (0 = Individual, 1 =  Joint)
#    credit-opened-after-2010 (0 = No, 1 = Yes)
datafile = 'credit.data'
datanames = ['credit-ecoa', 'credit-balance', 'credit-30daylate', 'credit-opened-after-2010',  'credit-class']
dataset = pandas.read_csv(datafile, names=datanames)

#    shape our dataset
#    this should tell us the number of rows and columns (150,5)
print(dataset.shape)

#    peek at the top of the data. aka: show the head of the top 20 records
print(dataset.head(20))

#    summarize our attributes
classgroup = dataset.groupby('credit-class')
print('*** Group by credit-class ***')
print(classgroup.describe())

ecoagroup = dataset.groupby('credit-ecoa')
print('*** Group by credit-ecoa ***')
print(ecoagroup.describe())



#    identify the distribution of each credit-class
print(classgroup.size())

#    python libs for machine learning are awesome.  check this out
#    visualization of our data - univariate plots (or each individual variable)
dataset.plot(kind='box', subplots=True, sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

#    visualization continued with multivariate plots
#    in other words, the interactions between our variables
#    using a scatter plot matrix
scatter_matrix(dataset)
plt.show()

#    now lets evaluate some of the available machine learning algorithms
#    step 1 - create a validation dataset
#    split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.80      #80 percent of data set
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
highest = 0.0
highestname = ""
print('** model accuracy results **')
for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

    #    determine the best model thus far
    if (highest < cv_results.mean()):
        highest = cv_results.mean()
        highestname = name

#    compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#
#    save probably a better way to do this and the remaining logic
msg = "best model = %s" % (highestname)
msg2 = "best model mean %f" % (highest)
print (msg)
print (msg2)

#    remember when using pickle to serialize an object you need
#    to use the same version of python and associated libs to deserialize
#    add these to the file name for better identification

version = '0'
if highestname == 'CART':
    model =  DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    filename = 'credit_cart_model_v0_.sav'
    pickle.dump(model, open(filename, 'wb'))
elif highestname == 'LR':
    model =  LogisticRegression()
    model.fit(X_train, Y_train)
    filename = 'credit_lr_model_v0_.sav'
    pickle.dump(model, open(filename, 'wb'))
elif highestname == 'LDA':
    model =  LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)
    filename = 'credit_lda_model_v0_.sav'
    pickle.dump(model, open(filename, 'wb'))
elif highestname == 'KNN':
    model =  KNeighborsClassifier()
    model.fit(X_train, Y_train)
    filename = 'credit_knn_model_v0_.sav'
    pickle.dump(model, open(filename, 'wb'))
elif highestname == 'NB':
    model =  GaussianNB()
    model.fit(X_train, Y_train)
    filename = 'credit_NB_model_v0_.sav'
    pickle.dump(model, open(filename, 'wb'))
elif highestname == 'SVM':
    model =  SVC()
    model.fit(X_train, Y_train)
    filename = 'credit_knn_model_v0_.sav'
    pickle.dump(model, open(filename, 'wb'))


#    TBD - create model README.TXT
print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))

filename = 'credit_cart_model_v0_.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_validation, Y_validation)
print (result)


print('end')