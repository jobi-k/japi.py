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
datafile = 'music.data'
header = ['song-name', 'song-published', 'song-author', 'song-origin', 'song-class']
dataset = pandas.read_csv(datafile, names=header)

#    shape our dataset
print(dataset.shape)

#    peek at the top of the data. aka: show the head of the top 20 records
print(dataset.head(20))

#    summarize our attributes
print(dataset.describe())

#    identify the distribution of each class (flower type)
print(dataset.groupby('song-class').size())

#    visualize data set
dataset.plot(kind='box', subplots=True, sharex=False, sharey=False)
plt.show()

print('end')