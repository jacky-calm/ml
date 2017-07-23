import pandas
import matplotlib.pyplot as plt
import numpy as np


def local_weighted_linear_regression(xi, X, y, k):
    # print "x, xi, yi", xi.shape
    m = X.shape[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = xi - X[j, :]
        weights[j, j] = np.exp(diff * diff.T / (-2 * np.square(k)))
    xTwx = X.T * (weights * X)
    if np.linalg.det(xTwx) == 0:
        print "This xTwx matrix is singular, cannot do inverse"
        return
    wl = xTwx.I * (X.T * (weights * y.T))
    return (xi * wl)[0, 0]


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
    ws = xTx.I * (X.T * y.T)  # weights calculated by standard linear regression
    print ws
    y_hat = X * ws
    plt.plot(data[1], y_hat, 'r')

    y_hat_l = np.array([local_weighted_linear_regression(xi, X, y, 0.03) for xi in X])
    sorted_index = np.argsort(data[1])

    plt.plot(data[1][sorted_index], y_hat_l[sorted_index], 'b')

    plt.show()


if __name__ == "__main__":
    main()
