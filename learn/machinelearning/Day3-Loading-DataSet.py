# Today's lesson focus' on loading data sets
# using files in the comma separated value format.
# CSV is the most commonly used format for representing
# data sets for machine learning.  There is a
# repository of test data sets available from UCI.
# http://archive.ics.uci.edu/ml/?__s=oszdtfazd9zgbvvpibcs

# Tips to remember
# NumPy (numpy.org) is Python package for scientific computing

# Lesson Agenda
#  1. Loading CSV formatted files using the standard CSV.Reader() library
#  2. Loading CSV formatted files using numpy.loadtxt() from the NumPy library
#  3. Loading CSV formatted files using pandas.read_csv() from the pandas library

import pandas
url = "https://goo.gl/vhm1eU"
headers = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=headers)

# Will show the number of rows and columns for the given data set
print(data.shape)

print('end')