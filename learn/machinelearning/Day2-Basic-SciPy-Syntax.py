# Today's lesson focus' on the basic syntax of
# Python.  With special focus on the machine
# learning library called SciPy.

# Reference - Thanks to Jason@machinelearning.com for putting
# together a program for individuals to self-master machine
# learning.  www.machinelearningmastery.com.  The below information
# is from his day 6 - Prepare for Modeling by Pre-Processing Data email


# Tips to remember
#   The has symbol is for comments (Duh!)
#   White space matters and is used for indicating code blocks.
#

# Lesson Agenda
#  1.  Assignment of variables
#  2.  Working with lists
#  3.  Flow Control
#  4.  NumPy Arrays
#  5.  Simply Plots with MatPlotLib
#  6.  Working with Pandas Series and DataFrames

import numpy
import pandas


myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)
