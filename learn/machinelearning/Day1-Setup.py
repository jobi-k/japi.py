# Display versions of python and any associated libraries

# Reference - Thanks to Jason@machinelearning.com for putting
# together a program for individuals to self-master machine
# learning.  www.machinelearningmastery.com.  The below information
# is from his day 6 - Prepare for Modeling by Pre-Processing Data email

# if you have anconda installed you should run the following from your command line
# to ensure the latest of the sklearn package is also installed.
# conda update scikit-learn


import sys
print('Python: {}'.format(sys.version))
import scipy
print('scipy: {}'.format(scipy.__version__))
import numpy
print('numpy: {}'.format(numpy.__version__))
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
import pandas
print('pandas: {}'.format(pandas.__version__))
import sklearn
print('sklearn: {}'.format(sklearn.__version__))