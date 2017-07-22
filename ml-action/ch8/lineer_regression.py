import pandas
import matplotlib.pyplot as plt
import numpy as np

data = pandas.read_csv('ex0.csv', sep='\t', header=None)
print data.shape
plt.scatter(data[1], data[2])
plt.show()

