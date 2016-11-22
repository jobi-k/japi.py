# Today's lesson focus' on gaining a better understanding
# of how to preform pre-processing of your data using scikit-learn.
# The purpose of this is to get your data into the best possible
# shape/structure before submitting to a modeling algorithm.

# Reference - Thanks to Jason@machinelearning.com for putting
# together a program for individuals to self-master machine
# learning.  www.machinelearningmastery.com.  The below information
# is from his day 6 - Prepare for Modeling by Pre-Processing Data email


# Tips to remember
# scikit-learn has two standard idioms for data transformation
# 1. Fit and Multiple Transform
# 2. Combined Fit-And-Transform
# There are many techniques that can be applied
#   standardized numerical data (mean 0, deviation of 1)
#   normalize numerical data (range 0-1)
#   binarizing

# Lesson Agenda

from sklearn.preprocessing import StandardScaler
import pandas
import numpy

url = "https://goo.gl/vhm1eU"
header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# standardized data (0 mean, 1 standard deviation)
dataframe = pandas.read_csv(url, names=header)
array = dataframe.values;

# seperate array
X = array[:, 0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

#summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])
