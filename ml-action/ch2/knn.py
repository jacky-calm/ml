import numpy as np
import operator
import os
import matplotlib.pyplot as plt

def createDataSet():
    X = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    y = ['A', 'A', 'B', 'B']
    return X, y

def norm(X):
    return X / (X.max(axis=0) - X.min(axis=0))

def classifyKnn(testX, trainX, trainY, k):
    testY = []
    print testX.shape
    for X in testX:
        dataSetSize = trainX.shape[0]
        broadcast = np.broadcast_to(X, (dataSetSize, len(X)))
        diff = broadcast - trainX
        diffSquare = diff ** 2
        distance = diffSquare.sum(axis=1)
        sortedIndex = distance.argsort()
        classCount = {}
        for i in range(k):
            label = trainY[sortedIndex[i]]
            classCount[label] = classCount.get(label, 0) + 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        testY.append(sortedClassCount[0][0])
    return np.array(testY)


def loadData(file):
    with open(file) as f:
        X = np.array([line.strip().split('\t') for line in f])
    return np.array(X[:, 0:2], dtype=float), np.array(X[:, 3], dtype=int)

def show(X, y):
    fig = plt.figure()
    plot = fig.add_subplot(231)
    plot.scatter(y, X[:, 0], s=15.0 * y, c=y, label='f0')
    plot = fig.add_subplot(232)
    plot.scatter(y, X[:, 1], s=15.0 * y, c=y, label='f1')
    plot = fig.add_subplot(233)
    plot.scatter(y, X[:, 2], s=15.0 * y, c=y, label='f2')

    plot = fig.add_subplot(234)
    plot.scatter(X[:, 0], X[:, 1], s=15.0 * y, c=y)
    plot = fig.add_subplot(235)
    plot.scatter(X[:, 0], X[:, 2], s=15.0 * y, c=y)
    plot = fig.add_subplot(236)
    plot.scatter(X[:, 1], X[:, 2], s=15.0 * y, c=y)

    plt.show()

# print os.getcwd()

X, y = loadData('ch2/datingTestSet2.txt')
X = norm(X)
print X.shape, type(X)
print y.shape
# show(X, y)
testRatio = 0.1
testCount = int(X.shape[0]*testRatio)
trainX = X[0:X.shape[0]-testCount, :]
trainY = y[0:X.shape[0]-testCount]
testX = X[X.shape[0]-testCount:, :]
testY = y[X.shape[0]-testCount:]
k = 10

y_hat = classifyKnn(testX, trainX, trainY, k)

print zip(testY, y_hat)
err = np.array(testY == y_hat)
print err, err.mean()

# dataSet, y = createDataSet()
# print dataSet
# print y
# print classifyKnn([1.0, 1.2], dataSet, y, 3)

