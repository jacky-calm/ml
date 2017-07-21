from math import log
import numpy as np
import pandas as pd

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return np.array(dataSet), labels

def entropy(data):
    print data
    n = len(data)
    labels = data[:, -1]
    print labels
    labelsV = pd.Categorical(labels).codes
    print labelsV
    p = np.array(np.bincount(labelsV)/float(n))
    print p
    p = -p*np.log2(p)
    print p
    return p.sum()


data, labels = createDataSet()

print np.split(data, )
print entropy(data)
