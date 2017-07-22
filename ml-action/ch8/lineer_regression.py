import pandas
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = pandas.read_csv('ex0.csv', sep='\t', header=None)
    print data.shape

    plt.scatter(data[1], data[2])

    X = np.mat(data[[0, 1]].values)
    y = np.mat(data[2].values)

    print X.shape, y.shape
    xTx = X.T * X
    print xTx

    if np.linalg.det(xTx) == 0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (X.T * y.T)
    print ws
    y_hat = X * ws
    plt.plot(data[1], y_hat, 'r')

    plt.show()


if __name__ == "__main__":
    main()
