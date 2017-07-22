from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def std(x, mu):
    return np.sqrt(((x - mu) ** 2).mean(axis=0))


def gaussian(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-np.square(x - mu) / (2 * np.square(sigma)))


def train_nb(x_train, y_train):
    # print [x_train[(y_train == i)].shape for i in range(3)]
    mu = [x_train[(y_train == i)].mean(axis=0) for i in range(3)]
    sigma = [std(x_train[(y_train == i)], mu[i]) for i in range(3)]
    # sigma2 = [np.std(x_train[(y_train == i)], axis=0) for i in range(3)]
    # print "sigma, sigma2: ===== "
    # print sigma
    # print sigma2
    # print "sigma, sigma2: ===== "
    return mu, sigma


def classify(x, mu, sigma, pOc):
    p = [np.log(gaussian(x, mu[i], sigma[i])).sum() + pOc[i] for i in range(3)]
    # print "p of c given x ===== ", p, np.argmax(p)
    return np.argmax(p)



def main():
    iris = load_iris()
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    for x0 in range(3):
        for x1 in range(x0+1, 4):
            x, y = iris.data[:, (x0, x1)], iris.target

            x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=1)
            print x.shape, x_train.shape, x_valid.shape

            mu, sigma = train_nb(x_train, y_train)
            # print "mu: ", mu
            # print "sigma: ", sigma

            pOc = [(y_train == i).mean() for i in range(3)]
            # print "pOc ====== ", pOc
            y_predict = [classify(v, mu, sigma, pOc) for v in x_valid]
            accuracy = (y_predict == y_valid).mean()

            error = x_valid[y_predict != y_valid]
            print "x0, x1, accuracy: ", accuracy
            if x0 == 0:
                plt.subplot(3, 1, x1)
                plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train+10, marker=".")
                plt.scatter(x_valid[:, 0], x_valid[:, 1], c=y_valid+100, marker="^")
                plt.scatter(error[:, 0], error[:, 1], c='r', marker="o")
                plt.xlabel(feature_names[x0])
                plt.ylabel(feature_names[x1])
                plt.title("accuracy {}".format(accuracy))

    plt.show()

if __name__ == "__main__":
    main()
