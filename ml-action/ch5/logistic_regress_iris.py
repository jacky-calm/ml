from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


data, target = load_iris(True)
data, target = data[target != 2], target[target != 2]  # only choose two classes of iris

data = np.append(np.ones((data.shape[0], 1)), data, axis=1)

x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8, random_state=1)

print x_train.shape

weights = np.ones(data.shape[1])
alpha = 0.1

w_trace, h_trace, error_trace, y_train_trace = [], [], [], []
for i in range(200):
    random_i = int(np.random.uniform(0, x_train.shape[0]))
    pick = x_train[random_i]
    h = sigmoid(np.dot(weights, pick))
    h_trace.append(h)
    error = h - y_train[random_i]
    y_train_trace.append(y_train[random_i])
    error_trace.append(error)
    weights = weights - alpha * error * pick
    w_trace.append(weights)

plt.subplot(311)
plt.plot(w_trace)
plt.title('weights')

plt.subplot(312)
plt.plot(error_trace)
plt.title('errors')

plt.tight_layout()
plt.show()
