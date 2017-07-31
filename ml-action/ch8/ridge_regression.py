import pandas
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True, linewidth=120)


def ridge_regress(xM, yM, lam=0.2):
    xTx = xM.T * xM
    denom = xTx + np.eye(xM.shape[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print "singular matrix"
        return
    ws = denom.I * (xM.T * yM)
    return ws


def ridge_test(xA, yA):
    xM, yM = np.mat(xA), np.mat(yA).T
    yMean = np.mean(yM, axis=0)
    yM = yM - yMean
    xMean = np.mean(xM, axis=0)
    xVar = np.var(xM, axis=0)
    xM = (xM - xMean) / xVar
    numTest = 30
    wM = np.zeros((numTest, xM.shape[1]))
    for i in range(numTest):
        ws = ridge_regress(xM, yM, np.exp(i - 10))
        wM[i, :] = ws.T
    return wM


def main():
    data = pandas.read_csv('abalone.csv', sep='\t', header=None).values
    xA = data[:, 0:-1]
    yA = data[:, -1]
    weights = ridge_test(xA, yA)
    print weights

    ax = plt.plot(range(-10, 20), weights)
    plt.xlabel('log(lambda)')
    plt.ylabel('weights')
    plt.title('ridge regression shrinkage')
    plt.legend(ax)
    plt.show()


if __name__ == '__main__':
    main()
