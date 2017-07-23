import pandas
import matplotlib.pyplot as plt
import numpy as np


def local_weighted_linear_regression(xi, X, y, k):
    # print "x, xi, yi", np.array(xi).shape, X.shape, xi
    m = X.shape[0]
    y_mat = np.mat(y)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = xi - X[j, :]
        # print "diff: ", diff.shape, diff.T.shape, diff * diff.T, diff, diff.T
        weights[j, j] = np.exp(np.dot(diff, diff.T) / (-2 * np.square(k)))
    # print "weights: ", weights.shape
    xTwx = X.T * (weights * X)
    if np.linalg.det(xTwx) == 0:
        print "This xTwx matrix is singular, cannot do inverse"
        return
    wl = xTwx.I * (X.T * (weights * y_mat.T))
    return (xi * wl)[0, 0]


def lwlr_test(test_x, train_x, train_y, k=1.0):
    # print "test_x: ", test_x.shape, train_x.shape
    return np.array([local_weighted_linear_regression(xi, train_x, train_y, k) for xi in test_x])


def plot_sorted(x, y, c='r'):
    sorted_index = np.argsort(x)
    plt.plot(x[sorted_index], y[sorted_index], c)


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

    y_hat_l = lwlr_test(X, X, y, 0.03)
    plot_sorted(data[1], y_hat_l, c='b')

    plt.show()


if __name__ == "__main__":
    main()
