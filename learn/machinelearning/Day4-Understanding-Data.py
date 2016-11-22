# Today's lesson focus' on gaining a better understanding
# of your data set using descriptive statistics.

# Reference - Thanks to Jason@machinelearning.com for putting
# together a program for individuals to self-master machine
# learning.  www.machinelearningmastery.com.  The below information
# is from his day 6 - Prepare for Modeling by Pre-Processing Data email


# Tips to remember
# The pandas library provides several helper functions

# Lesson Agenda
# 1. Understand data using the head() helper function
# 2. Review dimensions data using the shaper() helper function
# 3. Visualize the data types for each attribute using the dtypes property
# 4. Calculate pair-wise correlation between variables using corr()

# Example is dependent on the Pima Indians onset of diabetes data set

import pandas

url = "https://goo.gl/vhm1eU"
header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=header)
description = data.describe()
print(description)