import pandas
import matplotlib.pyplot as plt
import numpy as np
import linear_regression

data = pandas.read_csv('abalone.csv', '\t', header=None)
print data.shape

X, y = data.values[0:99, 0:-1], data.values[0:99, -1]
print X.shape, y.shape
y_hat = linear_regression.lwlr_test(data.values[100:199, 0:-1], X, y, 0.1)
print np.square(y_hat - data.values[100:199, -1]).sum()
