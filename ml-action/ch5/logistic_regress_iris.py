from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


data, target = load_iris(True)

x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8, random_state=1)
weights = np.ones(data.shape[1])
alpha = 0.1

for i in range(x_train.shape[0]):
    pick = x_train[i]
    h = sigmoid(np.dot(weights, pick))
    error = h - y_train[i]
    weights = weights - alpha * error * pick

    print weights
