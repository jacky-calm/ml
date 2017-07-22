from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


def std(x, mu):
    return np.sqrt(((x-mu) ** 2).mean(axis=0))


def gaussian(x, mu, sigma):
    print x, mu, sigma
    print x - mu, np.square(x - mu)
    return 1 / np.sqrt(2 * np.pi) * np.exp(-np.square(x - mu) / (2 * np.square(sigma)))


def train_nb(x_train, y_train):
    print [x_train[(y_train == i)].shape for i in range(3)]
    mu = [x_train[(y_train == i)].mean(axis=0) for i in range(3)]
    sigma = [std(x_train[(y_train == i)], mu[i]) for i in range(3)]
    sigma2 = [np.std(x_train[(y_train == i)], axis=0) for i in range(3)]
    print "sigma, sigma2: ===== "
    print sigma
    print sigma2
    print "sigma, sigma2: ===== "
    return mu, sigma


def main():
    iris = load_iris()
    x, y = iris.data[:, :2], iris.target

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=1)
    print x.shape, x_train.shape, x_valid.shape

    mu, sigma = train_nb(x_train, y_train)
    print mu
    print sigma
    p = [gaussian(x_valid[0], mu[i], sigma[i]) for i in range(3)]
    print "gaussian ===== ", p


if __name__ == "__main__":
    main()
